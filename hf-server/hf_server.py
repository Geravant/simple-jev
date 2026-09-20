"""Single-file Hugging Face classifier server using the shared common modules.

Run directly from a checkout (the sibling common/ folder is required)::

    python hf-server/hf_server.py --model /path/to/model --device cpu --dtype float32

Or install with pip install -e './hf-server[test]' and run::

    simple-jev --model organization/model --device auto
    python -m hf_server --model organization/model --device auto

Request flow:
    ClassifierRequest -> PromptCompiler -> HFBackend -> common.build_response

common owns validation, versioned classifier wording, label semantics, and answer
math. This file owns role assembly (plain text, or text plus images rendered
through the model's processor), native chat/tokenizer boundaries, shared-prefix
inference, queue/cancellation controls, usage accounting, HTTP, and startup.
Model weights load only when load_service/main is called.

The sections below follow the data flow and retain the implementation notes for
cache ownership, per-row logit selection, and asynchronous cleanup. Tests live in
tests/; no separate simple_jev package or duplicate classifier template is needed.
"""

import argparse
import asyncio
import base64
import copy
import inspect
import io
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError

# Direct script execution puts hf-server/, not the checkout root, on sys.path.
# Prefer the sibling common source when running from this repo; an installed
# wheel instead imports its bundled common package through normal resolution.
_checkout_root = Path(__file__).resolve().parent.parent
if (_checkout_root / "common" / "prompt_builder.py").is_file():
    sys.path.insert(0, str(_checkout_root))

from common import (
    ClassifierRequest,
    PromptPlan,
    build_response,
    prepare_prompt,
)
from common.prompt_builder import DEFAULT_TEMPLATE_VERSION, canonical

# Shared plan to native chat and tokens


@dataclass(frozen=True)
class Branch:
    """One compiled question, identified by its plan-local branch ID.

    token_ids encodes the complete prompt through the incomplete assistant answer
    prefix. output_ids contains the next-token vocabulary IDs, in the same order
    as the shared question's output_labels. Messages/prefix remain available for
    inspection; they are not reconstructed from tokens during inference.

    render_only compilation leaves both ID lists empty and is not executable.
    frozen prevents attribute reassignment, not mutation of the contained lists.
    """

    branch_id: str
    token_ids: list[int]
    output_ids: list[int]
    messages: list[dict]
    answer_prefix: str


@dataclass
class CompiledRequest:
    """Bind the semantic prompt plan to its HF-specific executable branches.

    Keep this pair together until response scoring: branch IDs and output order
    must be interpreted against the plan that created them. Branch order initially
    matches plan order; the backend may reorder execution for efficient padding.

    media holds the processor tensors for image input (pixel_values plus whatever
    the model family needs: image_grid_thw, mm_token_type_ids, image_position_ids)
    for the shared prefix forward only. media_length is the token length of the
    prompt those per-token tensors were built for; media_span is the first token
    index after the last image expansion, so the backend can confirm every image
    lies inside the shared prefix. None/0 means text only.
    """

    plan: PromptPlan
    branches: list[Branch]
    media: dict | None = None
    media_length: int = 0
    media_span: int = 0


def common_prefix(sequences):
    """Return the longest identical token prefix of a nonempty sequence group.

    The caller validates nonempty prompts. An empty shared prefix is valid.
    Comparing actual IDs is essential: matching rendered string fragments alone
    does not prove that the tokenizer produced reusable prefix tokens.
    """
    first = sequences[0]
    end = min(map(len, sequences))
    for other in sequences[1:]:
        for i in range(end):
            if first[i] != other[i]:
                end = i
                break
    return first[:end]


def decode_image(part, max_bytes):
    """Return a PIL image for one image content part, or raise ValueError.

    Accepts the hosted API shape {"type": "image_url", "image_url": {"url": ...}}
    (the url may also be given directly as a string) and {"type": "image",
    "url": ...}. Only data: URIs are read: the server never fetches remote URLs,
    so a request cannot make it reach other hosts. Decoded bytes are capped
    before PIL sees them; PIL's own decompression-bomb limit applies after.
    """
    from PIL import Image  # optional dependency: only image requests need it

    url = part.get("image_url") if part.get("type") == "image_url" else part.get("url")
    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError("Images must be data: URIs (image_url.url or url)")
    header, _, payload = url.partition(",")
    if not header.endswith(";base64") or not header[5:].startswith("image/"):
        raise ValueError("Image data URI must be data:image/<type>;base64,...")
    # Reject oversized payloads from the base64 length before decoding anything.
    if len(payload) * 3 // 4 > max_bytes:
        raise ValueError(f"Image exceeds {max_bytes} bytes")
    try:
        raw = base64.b64decode(payload, validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:  # noqa: BLE001 — any decode failure is a client error
        raise ValueError(f"Image could not be decoded: {type(exc).__name__}") from exc
    return image.convert("RGB")


def split_at_images(ids, image_token_id):
    """Split token IDs into text segments around each run of image placeholders.

    Adjacent placeholders (two images with no text between them) form one run,
    so every inner segment is nonempty. n runs give n + 1 segments.
    """
    segments, current, in_run = [], [], False
    for token in ids:
        if token == image_token_id:
            if not in_run:
                segments.append(current)
                current = []
            in_run = True
        else:
            current.append(token)
            in_run = False
    segments.append(current)
    return segments


def image_expansions(raw_ids, expanded_ids, image_token_id):
    """Recover what the processor substituted for each placeholder run.

    raw_ids is the tokenizer's encoding of the rendered chat, where the template
    left one placeholder token per image. expanded_ids is the processor's encoding
    of the same text with the images, where each placeholder became a run of
    soft-token IDs (plus any begin/end markers the family adds). Returns
    (shared_segments, expansions): the text segments before each run and the run
    itself, so other branches that share the same context can be expanded without
    running the image processor again. The final text segment is excluded: it is
    the only part that differs between branches. Raises ValueError when the two
    encodings cannot be aligned, which means the processor changed text outside
    the placeholders and per-branch expansion would be unsafe.
    """
    segments = split_at_images(raw_ids, image_token_id)
    runs = len(segments) - 1
    if runs == 0:
        raise ValueError("Chat template produced no image placeholder for the images")
    tail = segments[-1]
    if tail and expanded_ids[-len(tail):] != tail:
        raise ValueError("Processor changed the prompt text after the images")
    body = expanded_ids[: len(expanded_ids) - len(tail)]
    if body[: len(segments[0])] != segments[0]:
        raise ValueError("Processor changed the prompt text before the images")
    cursor = len(segments[0])
    expansions = []
    for k in range(1, runs + 1):
        if k == runs:
            end = len(body)
        else:
            # The next shared segment is nonempty and made of text tokens; its
            # first occurrence after at least one substituted token ends this run.
            segment = segments[k]
            end = -1
            for start in range(cursor + 1, len(body) - len(segment) + 1):
                if body[start : start + len(segment)] == segment:
                    end = start
                    break
            if end < 0:
                raise ValueError("Processor changed the prompt text between images")
        run = body[cursor:end]
        if image_token_id not in run:
            raise ValueError("Processor did not expand an image placeholder")
        expansions.append(run)
        cursor = end + (len(segments[k]) if k < runs else 0)
    return segments[:-1], expansions


def apply_image_expansions(raw_ids, shared_segments, expansions, image_token_id):
    """Rebuild one branch's expanded IDs from its raw IDs and the shared expansions.

    Every branch must carry the same shared context (same segments before each
    image run): images are part of the context, never of the per-question suffix.
    """
    segments = split_at_images(raw_ids, image_token_id)
    if segments[:-1] != shared_segments:
        raise ValueError("Images must be in the shared context, not in a question")
    expanded = []
    for segment, run in zip(shared_segments, expansions):
        expanded += segment + run
    return expanded + segments[-1]


class PromptCompiler:
    """Render shared classifier plans with a model's native tokenizer template."""

    def __init__(
        self,
        tokenizer,
        max_tokens=16384,
        version=DEFAULT_TEMPLATE_VERSION,
        *,
        processor=None,
        max_images=4,
        max_image_bytes=10_000_000,
    ):
        """Store the renderer, complete-prompt token limit, and shared version.

        Version validation is performed by common.prepare_prompt during compile.
        This class does not load a tokenizer or model on its own. processor is
        the model's multimodal processor; without it image content is rejected.
        max_images caps images per request and max_image_bytes caps each decoded
        image payload.
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.version = version
        self.processor = processor
        self.max_images = max_images
        self.max_image_bytes = max_image_bytes

    @property
    def accepts_images(self):
        """True when image content parts can be compiled."""
        return self.processor is not None

    def _image_token_id(self):
        """The placeholder token ID the chat template emits per image."""
        token_id = getattr(self.processor, "image_token_id", None)
        if token_id is None:
            token_id = self.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        return token_id

    def _chat_message(self, message, images):
        """Return a template-ready message, collecting decoded images in order.

        String content passes through. A content list keeps text parts as
        {"type": "text"} and turns each image part into the {"type": "image"}
        placeholder that HF chat templates expand; the decoded image goes to
        images. System/developer turns must be plain text: templates reject
        images there and the classifier instructions are merged into that turn.
        """
        content = message.content
        if isinstance(content, str):
            return {"role": message.role, "content": content}
        if content is None:
            raise ValueError("Chat message content is required")
        if message.role in {"system", "developer"}:
            raise ValueError("System messages must be plain text")
        parts = []
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind == "text":
                if not isinstance(part.get("text"), str):
                    raise ValueError("Text content parts need a string 'text'")
                parts.append({"type": "text", "text": part["text"]})
            elif kind in {"image", "image_url"}:
                if self.processor is None:
                    raise ValueError("This server was started without image support")
                if len(images) >= self.max_images:
                    raise ValueError(f"At most {self.max_images} images per request")
                images.append(decode_image(part, self.max_image_bytes))
                parts.append({"type": "image"})
            else:
                raise ValueError("Content parts must be text, image or image_url")
        return {"role": message.role, "content": parts}

    def _process_images(self, text, images):
        """Run the processor once on the first branch: expanded IDs plus tensors.

        The template already supplies special tokens, so the processor's tokenizer
        must not add BOS/EOS again. Everything except input_ids/attention_mask is
        model input for the prefix forward (pixel_values and family extras).
        """
        processed = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            add_special_tokens=False,
        )
        ids = processed["input_ids"][0].tolist()
        media = {
            k: v for k, v in processed.items() if k not in {"input_ids", "attention_mask"}
        }
        return ids, media

    def compile(self, request: ClassifierRequest, *, render_only=False):
        """Validate text input and compile one branch per shared-plan question.

        request may be a ClassifierRequest or an input dictionary. render_only
        returns roles/content and the shared plan for diagnostics/tests; it skips
        chat-template tokenization, token limits, and output-boundary checks.
        Such a result must not be sent to HFBackend.score.

        Unsupported media/tools, malformed token boundaries, and overlong prompts
        raise ValueError. Caller data is preserved when merging system messages.
        """
        if not isinstance(request, ClassifierRequest):
            request = ClassifierRequest.model_validate(request)
        # The shared schema allows richer contexts for other integrations. This
        # adapter narrows that contract: text chat, plus images when a processor
        # is configured. Tools, media options and tool/function turns stay out.
        if request.tools or request.mm_processor_kwargs or request.media_io_kwargs:
            raise ValueError("HF reference does not support tools or media options")
        if request.messages and any(
            m.model_extra or m.role in {"tool", "function"} for m in request.messages
        ):
            raise ValueError("HF reference accepts system/developer/user/assistant chat only")
        images = []
        chat = None
        if request.messages is not None:
            # New dictionaries: adding classifier instructions never mutates the
            # caller's conversation. Images are decoded once here, in order.
            chat = [self._chat_message(m, images) for m in request.messages]

        plan = prepare_prompt(request, version=self.version)
        system = plan.system_prompt_prefix + plan.prefix_instruction
        rendered = []   # (question, messages, text, raw_ids, output_ids)
        for question in plan.questions:
            content = plan.suffix_instruction + question.instruction
            # State is JSON-serialized, including quotes around string states.
            # For chat, preserve turn boundaries and merge only a leading system
            # turn; the final selected question is always a new user message.
            if chat is None:
                messages = [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": f"State:\n{canonical(request.state)}\n\n" + content,
                    },
                ]
            else:
                messages = [dict(m) for m in chat]
                if messages[0]["role"] == "system":
                    messages[0]["content"] = system + "\n" + messages[0]["content"]
                else:
                    messages.insert(0, {"role": "system", "content": system})
                messages.append({"role": "user", "content": content})

            text, ids, output_ids = "", [], []
            if not render_only:
                # Render first, then append incomplete JSON to the open assistant
                # position. Do not create a completed assistant message or add
                # a closing brace/EOS before the next-token scoring position.
                text = (
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    + question.answer_prefix
                )
                # The template already supplies special tokens. Adding another
                # BOS/EOS during encode would alter the intended model input.
                # With images, this encoding still holds one placeholder token
                # per image; the processor's expansion is applied below.
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if not ids or len(ids) > self.max_tokens:
                    raise ValueError(
                        f"Branch for {question.question_id!r} must contain 1–{self.max_tokens} tokens"
                    )
                # Derive IDs at the actual rendered boundary, not from isolated
                # label encoding. Check every branch; templates/context can affect it.
                # The boundary is text-only, so the tokenizer decides it even
                # when images precede it.
                for label in question.output_labels:
                    extended = self.tokenizer.encode(
                        text + label, add_special_tokens=False
                    )
                    if len(extended) != len(ids) + 1 or extended[:-1] != ids:
                        raise ValueError(
                            f"Answer label {label!r} is not single-token stable"
                        )
                    output_ids.append(extended[-1])
                if len(set(output_ids)) != len(output_ids):
                    raise ValueError("Output labels must map to distinct token IDs")
            rendered.append((question, messages, text, ids, output_ids))

        media, media_length, media_span = None, 0, 0
        if images and not render_only:
            # The image processor runs once, on the first branch. Its expansion
            # of each placeholder is then replayed onto every branch, which is
            # valid because images live in the shared context that all branches
            # have in common; apply_image_expansions verifies that.
            image_token_id = self._image_token_id()
            expanded, media = self._process_images(rendered[0][2], images)
            shared, expansions = image_expansions(rendered[0][3], expanded, image_token_id)
            media_length = len(expanded)
            media_span = sum(map(len, shared)) + sum(map(len, expansions))
            for i, (question, messages, text, ids, output_ids) in enumerate(rendered):
                ids = apply_image_expansions(ids, shared, expansions, image_token_id)
                if len(ids) > self.max_tokens:
                    raise ValueError(
                        f"Branch for {question.question_id!r} must contain 1–{self.max_tokens} tokens"
                    )
                rendered[i] = (question, messages, text, ids, output_ids)

        branches = [
            Branch(question.branch_id, ids, output_ids, messages, question.answer_prefix)
            for question, messages, _, ids, output_ids in rendered
        ]
        return CompiledRequest(plan, branches, media, media_length, media_span)


# Inference result


@dataclass
class BackendResult:
    """Results keyed by plan-local branch ID, independent of batch execution order.

    HF stores one 1-D CPU float32 tensor per branch, ordered exactly like the
    corresponding shared question's output_labels. The broad Any annotation also
    permits label/float maps from test backends, accepted by common's scorer.

    metrics contains internal timing/token/batch counters. branch_output_tokens
    is zero for HF logits-only inference. It does not supply input usage; the
    service computes the unique token-prefix union from the compiled prompts.
    """

    # Compact tensors in each plan question's output_labels order, keyed by ID.
    logits: dict[str, Any]
    metrics: dict


# Logical input-token accounting


def unique_prompt_tokens(sequences):
    """Return the number of distinct token-tree edges across all sequences.

    Example: [1, 2, 3] and [1, 2, 4] count as four tokens, not six. Duplicate
    paths add nothing; a path that ends inside another adds no new suffix. Empty
    input returns zero. Sorting creates a new list and leaves caller order intact.

    Lexicographic neighbors share the greatest already-counted prefix for each
    next sequence. Subtract that overlap from its length instead of building an
    explicit trie. No global cache or backend warmup information is consulted.
    """
    # Lexicographically adjacent sequences share the maximum previously seen
    # prefix. Count each token-tree edge once, including hierarchical prefixes.
    total = 0
    previous = []
    for sequence in sorted(sequences):
        shared = 0
        for left, right in zip(previous, sequence):
            if left != right:
                break
            shared += 1
        total += len(sequence) - shared
        previous = sequence
    return total


# Shared-prefix model execution


class HFBackend:
    """Own one inference model and serialize all forwards through a thread lock."""

    def __init__(self, model, *, max_batch_size=32, max_batch_tokens=32768):
        """Set evaluation mode and bounds on padded suffix batches.

        max_batch_size caps rows; max_batch_tokens caps rows times suffix width.
        Neither bounds the prefix prefill or cache memory. The tokenizer/compiler
        enforces the full model-context limit separately.
        """
        if max_batch_size < 1 or max_batch_tokens < 1:
            raise ValueError("Batch limits must be positive")
        self.model = model.eval()
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        # Processors return float32 pixel tensors; the vision tower runs in the
        # weight dtype (bfloat16 in production).
        self._dtype = next(model.parameters()).dtype
        # Protect the model even if cancellation returns before its thread exits.
        self._lock = threading.Lock()
        # Some model classes accept selected sequence positions to avoid
        # materializing all sequence logits. Fall back to full logits otherwise.
        self._last_logits = (
            "logits_to_keep" in inspect.signature(model.forward).parameters
        )

    async def score(self, compiled):
        """Run blocking model work in a worker thread with cooperative cancellation.

        Cancelling an asyncio task cannot interrupt an in-flight tensor kernel.
        Set a stop flag for the worker to check before its next batch, then let
        cancellation propagate. The model lock remains held until the worker exits.
        """
        stop = threading.Event()
        try:
            return await asyncio.to_thread(self._score, compiled, stop)
        except asyncio.CancelledError:
            stop.set()
            raise

    def _prefix_media(self, compiled, prefix_length, device):
        """Model inputs for the prefix forward from the compiled media tensors.

        Per-token tensors (mm_token_type_ids, token_type_ids) were built for the
        first branch's full prompt and are cut to the prefix. Floating tensors
        take the weight dtype; everything else only moves to the embedding device.
        """
        import torch

        inputs = {}
        for name, value in (compiled.media or {}).items():
            if isinstance(value, torch.Tensor):
                if (
                    name.endswith("token_type_ids")
                    and value.ndim == 2
                    and value.shape == (1, compiled.media_length)
                ):
                    value = value[:, :prefix_length]
                if value.is_floating_point():
                    value = value.to(device=device, dtype=self._dtype)
                else:
                    value = value.to(device)
            inputs[name] = value
        return inputs

    def _reset_multimodal_state(self):
        """Clear rope_deltas cached on M-RoPE models by a previous request.

        Those models derive positions from the stored deltas whenever a cache
        is present and no positions are given; a stale value from an earlier
        image request must never leak into the next prefix forward.
        """
        for module in (self.model, getattr(self.model, "model", None)):
            if module is not None and hasattr(module, "rope_deltas"):
                module.rope_deltas = None

    def _score(self, compiled, stop):
        """Execute one request; all model/cache work stays under the same lock."""
        import torch

        with self._lock, torch.inference_mode():
            if stop.is_set():
                raise asyncio.CancelledError()
            start = time.perf_counter()
            sequences = [b.token_ids for b in compiled.branches]
            if not sequences or any(not ids for ids in sequences):
                raise ValueError("Expected nonempty scoring prompts")
            # Leave at least one suffix token, including for identical prompts.
            prefix = common_prefix(sequences)[: min(map(len, sequences)) - 1]
            # Inputs start on the embedding device; a dispatched/sharded model
            # may move later activations using its own Transformers hooks.
            device = self.model.get_input_embeddings().weight.device
            extra = {"logits_to_keep": 1} if self._last_logits else {}
            cache = None
            rope_deltas = None
            forwards = 0
            # Images are consumed by the prefix forward only: their soft tokens
            # then sit in the seed cache like any other context. A prefix that
            # ends inside an image would split its tokens across forwards.
            if compiled.media and len(prefix) < compiled.media_span:
                raise ValueError("Images must lie entirely within the shared prompt prefix")
            if prefix:
                # This is one unchunked forward. The returned cache is the seed;
                # no suffix batch may mutate it or reuse another batch's cache.
                ids = torch.tensor([prefix], device=device)
                media = self._prefix_media(compiled, len(prefix), device)
                self._reset_multimodal_state()
                out = self.model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=True,
                    **media,
                    **extra,
                )
                cache = out.past_key_values
                if cache is None or not hasattr(cache, "reorder_cache"):
                    raise ValueError(
                        "Model must expose a reorderable Transformers cache"
                    )
                # M-RoPE families (Qwen3.5, Qwen3-VL) report the offset between
                # token index and text position after multimodal content.
                rope_deltas = getattr(out, "rope_deltas", None)
                del out
                forwards += 1
            lengths = [len(ids) - len(prefix) for ids in sequences]
            if max(lengths) > self.max_batch_tokens:
                raise ValueError("A question suffix exceeds max_batch_tokens")
            # Longest first reduces padding. Results carry branch IDs, so map
            # insertion/execution order need not equal the original question order.
            pending = sorted(
                range(len(sequences)), key=lambda i: lengths[i], reverse=True
            )
            results = {}
            sizes = []
            padded_tokens = 0
            while pending:
                if stop.is_set():
                    raise asyncio.CancelledError()
                # Every row uses width slots, including right padding. Account
                # for padded work, not just the sum of unpadded suffix lengths.
                width = lengths[pending[0]]
                limit = min(self.max_batch_size, self.max_batch_tokens // width)
                batch, pending = pending[:limit], pending[limit:]
                # Forward mutates its cache. Never let a branch mutate the seed.
                branch_cache = copy.deepcopy(cache)
                if branch_cache is not None:
                    # The seed has one row. Repeated index zero broadcasts that
                    # row to the batch via the cache's supported reorder API.
                    branch_cache.reorder_cache(
                        torch.zeros(len(batch), dtype=torch.long, device=device)
                    )
                # Padding follows the scored position, never enters a real token's
                # causal context, and is excluded from attention. Discard this cache.
                ids = torch.zeros((len(batch), width), dtype=torch.long, device=device)
                mask = torch.zeros(
                    (len(batch), len(prefix) + width), dtype=torch.long, device=device
                )
                for row, i in enumerate(batch):
                    ids[row, : lengths[i]] = torch.tensor(
                        sequences[i][len(prefix) :], device=device
                    )
                    mask[row, : len(prefix) + lengths[i]] = 1
                # Position IDs continue after the shared prefix. Attention masks
                # include both prefix and suffix; real tokens cannot attend to
                # right padding. Padded token ID zero is just unused storage.
                positions = (
                    torch.arange(len(prefix), len(prefix) + width, device=device)
                    .unsqueeze(0)
                    .expand(len(batch), -1)
                )
                if rope_deltas is not None:
                    # After an image, M-RoPE text positions continue from the
                    # highest multimodal position, not from the token count.
                    # Give the three rotary axes the same shifted positions, as
                    # the model does itself during incremental decoding.
                    positions = positions.unsqueeze(0).expand(3, -1, -1) + rope_deltas.to(
                        device
                    ).reshape(1, -1, 1)
                last = torch.tensor([lengths[i] - 1 for i in batch], device=device)
                # Unequal lengths mean different final positions per row.
                # logits_to_keep accepts one position set shared across rows;
                # inverse maps each row's last position into that returned set.
                keep, inverse = torch.unique(last, sorted=True, return_inverse=True)
                branch_extra = {"logits_to_keep": keep} if self._last_logits else {}
                out = self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=branch_cache,
                    use_cache=True,
                    **branch_extra,
                )
                selected = out.logits[
                    torch.arange(len(batch), device=device),
                    inverse if self._last_logits else last,
                ]
                # Gather each branch's permitted tokens before leaving the batch.
                # Compact CPU tensors avoid retaining full vocabulary/GPU buffers.
                for row, i in enumerate(batch):
                    branch = compiled.branches[i]
                    results[branch.branch_id] = (
                        selected[row, branch.output_ids].float().cpu()
                    )
                del out, branch_cache, selected
                forwards += 1
                sizes.append(len(batch))
                padded_tokens += len(batch) * width
            return BackendResult(
                results,
                {
                    "backend": "transformers",
                    "prefill_strategy": "shared_prefix",
                    "prefix_tokens": len(prefix),
                    "suffix_batch_sizes": sizes,
                    "engine_forwards": forwards,
                    # These are distinct accounting views, not interchangeable:
                    # branch_prompt_tokens repeats shared context per branch;
                    # computed_prompt_tokens includes padding but prefixes once;
                    # logical_prefill_tokens omits padding, still counting overlap
                    # beyond the one shared prefix separately per branch.
                    "branch_prompt_tokens": sum(map(len, sequences)),
                    "computed_prompt_tokens": len(prefix) + padded_tokens,
                    "logical_prefill_tokens": len(prefix) + sum(lengths),
                    "padded_suffix_tokens": padded_tokens,
                    "branch_output_tokens": 0,
                    "scored_positions": len(sequences),
                    "backend_seconds": time.perf_counter() - start,
                },
            )


# Request admission and response assembly


class OverloadedError(Exception):
    """Admission capacity is exhausted; the HTTP layer translates this to 429."""


class DecisionService:
    """Manage one configured model, its compiler/backend, and request lifecycle.

    Use on one asyncio event loop: the admission counter is intentionally updated
    without awaiting between its capacity check and increment. Compiler/backend
    dependencies allow service tests to run without model weights.
    """

    def __init__(
        self,
        model,
        compiler,
        backend,
        *,
        metadata=None,
        concurrency=4,
        queue_size=16,
        max_request_branches=100,
        model_aliases=(),
        advanced_metrics=None,
    ):
        """Configure admission and diagnostic output for a loaded model.

        concurrency counts active coroutine slots; queue_size adds waiting slots.
        Supply a positive concurrency and nonnegative queue size. model_aliases
        permits additional names for the same loaded model, not dynamic loading.
        advanced_metrics explicitly overrides the environment flag when provided;
        otherwise 1/true/yes/on enable ENABLE_OPEN_JEV_ADVANCED_METRICS.
        """
        if max_request_branches < 1:
            raise ValueError("max_request_branches must be positive")
        self.max_request_branches = max_request_branches
        self.model_aliases = {model, *model_aliases}
        self.model = model
        self.compiler = compiler
        self.backend = backend
        self.metadata = metadata or {}
        self.advanced_metrics = (
            os.environ.get("ENABLE_OPEN_JEV_ADVANCED_METRICS", "").strip().lower()
            in {"1", "true", "yes", "on"}
            if advanced_metrics is None
            else advanced_metrics
        )
        self._semaphore = asyncio.Semaphore(concurrency)
        self._capacity = concurrency + queue_size
        self._inflight = 0

    async def classify(self, request):
        """Return a complete response or raise validation/overload/cancellation.

        Validate and admit before expensive work. Admission is released in finally
        on success, failure, or cancellation. The queue timer ends when the active
        slot is acquired; total_seconds spans admitted work through response build.
        These timings use a monotonic clock and are not distributed trace spans.
        """
        if not isinstance(request, ClassifierRequest):
            request = ClassifierRequest.model_validate(request)
        if request.model not in self.model_aliases:
            raise ValueError(f"Loaded model is {self.model!r}")
        # v1 has exactly one inference branch per question; candidate count no
        # longer expands requests. The shared schema separately caps 256 questions.
        branches = len(request.questions)
        if branches > self.max_request_branches:
            raise ValueError(
                f"Request has {branches} scoring branches; maximum is {self.max_request_branches}"
            )
        if self._inflight >= self._capacity:
            raise OverloadedError("Scoring queue is full")
        # No await between this check/increment pair: other tasks on the same
        # loop cannot interleave admission and oversubscribe the capacity.
        self._inflight += 1
        start = time.perf_counter()
        try:
            async with self._semaphore:
                queued = time.perf_counter() - start
                # Tokenization can be expensive and must not block cancellation/HTTP.
                if hasattr(self.compiler, "compile_async"):
                    compiled = await self.compiler.compile_async(request)
                else:
                    images_ok = getattr(self.compiler, "accepts_images", False)
                    if request.messages and any(
                        (not isinstance(m.content, str) and not images_ok)
                        or m.role in {"tool", "function"}
                        for m in request.messages
                    ):
                        raise ValueError(
                            "Multimodal/tool chat requires a native renderer; this backend accepts text chat only"
                        )
                    compiled = await asyncio.to_thread(self.compiler.compile, request)
                # The compiler retains the shared plan alongside executable IDs.
                # Never reconstruct label meaning from model output or batch order.
                result = await self.backend.score(compiled)
                response = build_response(
                    compiled.plan,
                    result.logits,
                    # Logical prefix-union accounting excludes batch padding and
                    # repeated context, not a sum of the backend's forward sizes.
                    input_tokens=unique_prompt_tokens(
                        [b.token_ids for b in compiled.branches]
                    ),
                    output_tokens=result.metrics.get("branch_output_tokens", 0),
                    advanced=self.advanced_metrics,
                )
                # Common handles answer filtering and authoritative version data;
                # HF only adds backend-specific metadata and execution timings.
                if self.advanced_metrics:
                    response["metadata"] = {
                        **self.metadata,
                        **response["metadata"],
                        "usage_accounting": "unique_token_prefixes_and_engine_leaf_outputs",
                    }
                    response["metrics"] = {
                        **result.metrics,
                        "queue_seconds": queued,
                        "total_seconds": time.perf_counter() - start,
                    }
                return response
        finally:
            # Release service admission even if a cancelled worker is finishing.
            # HFBackend's lock still prevents overlapping access to the model.
            self._inflight -= 1


# HTTP routes and errors


def validation_response(message=None, errors=()):
    """Format explicit text or Pydantic errors without returning input payloads.

    Keep at most ten structured details and report the count of additional
    errors. Convert location tuples into readable dotted paths with array indices,
    omitting the leading transport-specific 'body' component. The first location
    becomes the top-level param; errors without a location use null.
    """
    details = []
    for error in errors[:10]:
        path = ""
        for part in error.get("loc", ()):
            if part == "body" and not path:
                continue
            if isinstance(part, int):
                path += f"[{part}]"
            else:
                path += ("." if path else "") + str(part)
        details.append(
            {"param": path or None, "message": error["msg"], "type": error["type"]}
        )
    if message is None:
        message = (
            "; ".join(
                f"{e['param']}: {e['message']}" if e["param"] else e["message"]
                for e in details
            )
            or "Invalid classifier request"
        )
        if len(errors) > len(details):
            message += f"; {len(errors) - len(details)} additional validation errors"
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": 422,
                "param": details[0]["param"] if details else None,
                "details": details,
            }
        },
    )


class ClassifierRoute(APIRoute):
    """Normalize errors only for classifier routes, leaving host handlers alone.

    Request parsing can fail before the endpoint function runs, so normalization
    belongs around FastAPI's generated route handler as well as in the endpoint.
    """

    def get_route_handler(self):
        """Wrap FastAPI parsing/execution while preserving non-422 exceptions."""
        handler = super().get_route_handler()

        async def validated(request):
            """Intercept classifier validation errors before they leave this route."""
            try:
                return await handler(request)
            except RequestValidationError as exc:
                # Exclude input values and exception contexts from public errors.
                return validation_response(errors=exc.errors())
            except HTTPException as exc:
                if exc.status_code == 422:
                    return validation_response(message=str(exc.detail))
                raise

        return validated


def create_app(service):
    """Build a standalone app around an already-created service.

    The CLI loads the model before calling this function. Readiness therefore
    reports that configured service, not a new inference probe on every request.
    """
    app = FastAPI(title="Simple-JEV", version="0.1.0")

    @app.get("/health")
    async def health():
        """Return the configured model identifier without invoking inference."""
        return {"status": "ready", "model": service.model}

    attach_routes(app, lambda request: service)
    return app


def attach_routes(app, get_service):
    """Attach classifier endpoints; the alias stays out of generated OpenAPI.

    get_service is synchronous and request-scoped, allowing an embedding app to
    select its service without changing the classifier handler's implementation.
    """
    router = APIRouter(route_class=ClassifierRoute)

    @router.post("/v1/classifier")
    @router.post("/v1/systemone", include_in_schema=False)
    async def classify(body: ClassifierRequest, request: Request):
        """Run one service task and cancel it if the HTTP client disconnects."""
        task = asyncio.create_task(get_service(request).classify(body))
        try:
            # Wait with a timeout rather than awaiting task directly, so a long
            # tokenization/model pass does not prevent disconnect checks.
            while not task.done():
                await asyncio.wait({task}, timeout=0.1)
                if await request.is_disconnected():
                    task.cancel()
                    raise HTTPException(499, "Client disconnected")
            return await task
        except OverloadedError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "1"}) from exc
        except ValidationError as exc:
            return validation_response(errors=exc.errors())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            # Always observe the child task's completion/exception. Cancelling
            # its coroutine does not forcibly stop an active model worker thread;
            # HFBackend implements cooperative stopping and a model lock.
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    app.include_router(router)


# Model loading and command-line entry point


def load_service(
    model_name,
    *,
    revision=None,
    device="auto",
    dtype="bfloat16",
    max_model_len=16384,
    max_batch_size=32,
    max_batch_tokens=32768,
    max_request_branches=100,
    images=True,
    max_images=4,
    model_aliases=(),
):
    """Load a model and return a ready-to-use service, without starting HTTP.

    device is passed to Transformers as device_map; dtype selects a torch dtype.
    max_model_len limits each complete compiled prompt. max_batch_tokens limits
    padded suffix tokens per batch, not shared-prefix prefill or total KV memory.
    max_request_branches caps questions admitted in a single request. images
    loads the checkpoint's processor when its config has a vision tower, which
    enables image content parts; max_images caps images per request.
    model_aliases are extra names accepted in the request "model" field, for
    hosts that mount the weights under a path such as /repository.

    The loader sets service concurrency to one: separate requests are serialized,
    while branches within a request are batched. The backend's thread lock also
    prevents overlap if cancellation releases admission before a forward ends.
    """
    # Heavy dependencies are local to loading, so CLI help and source inspection
    # do not initialize a model or import the Transformers model classes.
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    config = AutoConfig.from_pretrained(model_name, revision=revision)
    # These checkpoint families use the image/text auto-loader even for text
    # scoring. Image input additionally needs the checkpoint's processor.
    loader = (
        AutoModelForImageTextToText
        if config.model_type in {"gemma4", "qwen3_5", "qwen3_5_moe"}
        else AutoModelForCausalLM
    )
    model = loader.from_pretrained(
        model_name,
        revision=revision,
        dtype=getattr(torch, dtype),
        device_map=device,
    )
    processor = None
    if images and getattr(config, "vision_config", None) is not None:
        from transformers import AutoProcessor

        try:
            processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        except Exception as exc:  # noqa: BLE001 — degrade to text, say why
            print(f"Image input disabled: processor failed to load ({exc})", file=sys.stderr)
    # PromptCompiler's default comes from common.DEFAULT_TEMPLATE_VERSION.
    # Keep one compiler/backend pair for the service's loaded model/tokenizer.
    compiler = PromptCompiler(
        tokenizer, max_tokens=max_model_len, processor=processor, max_images=max_images
    )
    backend = HFBackend(
        model,
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
    )
    return DecisionService(
        model_name,
        compiler,
        backend,
        concurrency=1,
        max_request_branches=max_request_branches,
        model_aliases=model_aliases,
        metadata={"backend": "transformers", "model_revision": revision},
    )


def main():
    """Parse process settings, load the service, then run its ASGI application.

    host/port are Uvicorn settings; the other arguments configure loading and
    inference. Model loading happens before the listener starts accepting work.
    --help exits during parsing and therefore does not load any weights.
    """
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--max-request-branches", type=int, default=100)
    parser.add_argument(
        "--no-images",
        dest="images",
        action="store_false",
        help="do not load the processor; reject image content even for vision models",
    )
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument(
        "--model-alias",
        dest="model_aliases",
        action="append",
        default=[],
        metavar="NAME",
        help="extra name accepted in the request model field (repeatable)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = vars(parser.parse_args())
    host, port, model = args.pop("host"), args.pop("port"), args.pop("model")
    uvicorn.run(create_app(load_service(model, **args)), host=host, port=port)


if __name__ == "__main__":
    main()
