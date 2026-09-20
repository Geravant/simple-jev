"""Image input: placeholder expansion, compile validation, and cached scoring.

The compile tests use a byte tokenizer whose chat template emits one <img>
placeholder per image and a fake processor that expands it the way HF
processors do (begin marker, soft tokens, end marker) and returns per-token
type IDs plus a pixel tensor. The backend tests build tiny randomly initialized
Qwen3.5 and Gemma4 vision-language models and compare shared-prefix scoring,
with the image consumed by the prefix forward, against independent complete
forwards. They establish execution equivalence, not answer quality.
"""

import base64
import importlib.util
import io

import numpy as np
import pytest

from hf_server import (
    Branch,
    CompiledRequest,
    PromptCompiler,
    apply_image_expansions,
    image_expansions,
    split_at_images,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None
    or importlib.util.find_spec("PIL") is None,
    reason="Install HF optional dependencies",
)

# Token IDs above the ASCII range so text bytes never collide with them.
IMG, BOI, EOI = 250, 251, 252


class ImageTokenizer:
    """Byte tokenizer whose template renders image parts as one <img> token."""

    image_token = "<img>"

    def encode(self, text, **kwargs):
        ids = []
        for i, piece in enumerate(text.split("<img>")):
            if i:
                ids.append(IMG)
            ids += list(piece.encode("ascii"))
        return ids

    def convert_tokens_to_ids(self, token):
        return IMG if token == "<img>" else None

    def apply_chat_template(self, messages, **kwargs):
        lines = []
        for m in messages:
            content = m["content"]
            if isinstance(content, list):
                content = "".join(
                    "<img>" if p["type"] == "image" else p["text"] for p in content
                )
            lines.append(f"{m['role']}: {content}")
        return "\n".join(lines) + "\nassistant: "


class FakeProcessor:
    """Expand <img> to BOI + (width // 4) soft tokens + EOI, like HF processors.

    Returns input_ids, attention_mask, mm_token_type_ids and a pixel tensor;
    media_factory can add family-specific tensors for a real tiny model.
    """

    image_token_id = IMG

    def __init__(self, media_factory=None):
        self.calls = 0
        self.media_factory = media_factory

    def __call__(self, text, images, return_tensors, add_special_tokens):
        import torch

        assert return_tensors == "pt" and add_special_tokens is False
        assert len(text) == 1
        self.calls += 1
        ids, types = [], []
        for i, piece in enumerate(text[0].split("<img>")):
            if i:
                soft = images[i - 1].width // 4
                ids += [BOI] + [IMG] * soft + [EOI]
                types += [0] + [1] * soft + [0]
            piece_ids = list(piece.encode("ascii"))
            ids += piece_ids
            types += [0] * len(piece_ids)
        out = {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([types]),
            "pixel_values": torch.zeros(len(images), 3),
        }
        if self.media_factory:
            out.update(self.media_factory(images))
        return out


def data_uri(width=16, height=8, kind="PNG"):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buffer, kind)
    mime = "image/png" if kind == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode()


def image_part(width=16, shape="image_url"):
    if shape == "image_url":
        return {"type": "image_url", "image_url": {"url": data_uri(width)}}
    return {"type": "image", "url": data_uri(width)}


QUESTIONS = {
    "color": {
        "type": "choice",
        "instructions": "Color?",
        "criteria": {"red": None, "blue": None},
    },
    "yes": {"type": "noul", "instructions": "Red?"},
}


def request(parts, system="Look.", questions=QUESTIONS):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": parts})
    return {"model": "test", "messages": messages, "questions": questions}


# Placeholder expansion helpers


def test_expansions_are_recovered_and_replayed():
    raw = [1, 2, IMG, 3, 4, IMG, IMG, 5, 6]
    first = [BOI, IMG, IMG, EOI]
    second = [BOI, IMG, EOI, BOI, IMG, IMG, IMG, EOI]
    expanded = [1, 2, *first, 3, 4, *second, 5, 6]
    assert split_at_images(raw, IMG) == [[1, 2], [3, 4], [5, 6]]
    shared, expansions = image_expansions(raw, expanded, IMG)
    assert shared == [[1, 2], [3, 4]]
    assert expansions == [first, second]
    # Another branch with the same context and a different question tail.
    other = [1, 2, IMG, 3, 4, IMG, IMG, 9, 9, 9]
    assert apply_image_expansions(other, shared, expansions, IMG) == [
        1, 2, *first, 3, 4, *second, 9, 9, 9,
    ]
    with pytest.raises(ValueError, match="shared context"):
        apply_image_expansions([1, IMG, 3, 4, IMG, IMG, 9], shared, expansions, IMG)
    with pytest.raises(ValueError, match="no image placeholder"):
        image_expansions([1, 2, 3], expanded, IMG)
    with pytest.raises(ValueError, match="after the images"):
        image_expansions(raw, expanded[:-1] + [7], IMG)
    with pytest.raises(ValueError, match="did not expand"):
        image_expansions([1, IMG, 2], [1, 2], IMG)


# Compile path


def test_images_are_expanded_on_every_branch_with_one_processor_call():
    processor = FakeProcessor()
    compiler = PromptCompiler(ImageTokenizer(), processor=processor)
    body = request([{"type": "text", "text": "Photo:"}, image_part(16), {"type": "text", "text": " what?"}])
    original = [dict(m) for m in body["messages"]]
    compiled = compiler.compile(body)

    assert processor.calls == 1
    assert body["messages"] == original, "caller messages must not be mutated"
    assert set(compiled.media) == {"mm_token_type_ids", "pixel_values"}
    assert tuple(compiled.media["pixel_values"].shape) == (1, 3)
    first, second = compiled.branches
    run = [BOI, IMG, IMG, IMG, IMG, EOI]
    for branch in (first, second):
        ids = branch.token_ids
        start = ids.index(BOI)
        assert ids[start : start + len(run)] == run
        assert ids.count(IMG) == 4, "the single placeholder was replaced, not kept"
        assert ids[: compiled.media_span] == first.token_ids[: compiled.media_span]
        assert len(branch.output_ids) == len(set(branch.output_ids))
    assert compiled.media_span == first.token_ids.index(EOI) + 1
    assert compiled.media_length == len(first.token_ids)
    assert tuple(compiled.media["mm_token_type_ids"].shape) == (1, compiled.media_length)
    # The question tails differ; both carry the rendered question text.
    assert first.token_ids[compiled.media_span :] != second.token_ids[compiled.media_span :]
    assert bytes(first.token_ids[compiled.media_span :]).endswith(b'{"answer": "')
    # Template-ready messages keep the image placeholder and merge the system turn.
    plan_only = compiler.compile(body, render_only=True)
    assert plan_only.media is None
    user = plan_only.branches[0].messages[1]["content"]
    assert user[1] == {"type": "image"} and user[0]["type"] == "text"
    assert plan_only.branches[0].messages[0]["content"].endswith("\nLook.")


def test_image_and_image_url_parts_and_two_images():
    processor = FakeProcessor()
    compiler = PromptCompiler(ImageTokenizer(), processor=processor)
    body = request([image_part(16, "image_url"), {"type": "text", "text": "|"}, image_part(8, "image")])
    compiled = compiler.compile(body)
    ids = compiled.branches[0].token_ids
    assert ids.count(IMG) == 4 + 2 and ids.count(BOI) == 2
    assert tuple(compiled.media["pixel_values"].shape) == (2, 3)
    # Adjacent images with no text between them form one run.
    body = request([image_part(8), image_part(8)])
    ids = compiler.compile(body).branches[0].token_ids
    assert ids.count(BOI) == 2 and ids.count(IMG) == 4


@pytest.mark.parametrize(
    "compiler_kwargs, parts, system, message",
    [
        ({}, [image_part()], "Look.", "without image support"),
        ({"processor": FakeProcessor(), "max_images": 1}, [image_part(), image_part()], "Look.", "At most 1 images"),
        ({"processor": FakeProcessor()}, [{"type": "audio", "audio": "x"}], "Look.", "text, image or image_url"),
        ({"processor": FakeProcessor()}, [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}], "Look.", "data: URIs"),
        ({"processor": FakeProcessor()}, [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}], "Look.", "could not be decoded"),
        ({"processor": FakeProcessor(), "max_image_bytes": 10}, [image_part()], "Look.", "exceeds 10 bytes"),
        ({"processor": FakeProcessor()}, [{"type": "text", "text": "x"}], [{"type": "text", "text": "sys"}], "must be plain text"),
    ],
)
def test_image_requests_are_validated(compiler_kwargs, parts, system, message):
    compiler = PromptCompiler(ImageTokenizer(), **compiler_kwargs)
    with pytest.raises(ValueError, match=message):
        compiler.compile(request(parts, system=system))


# Tiny vision-language models: cached scoring equals complete forwards


def tiny_vlm(family):
    """Randomly initialized tiny VLM plus a builder for its image inputs."""
    import torch

    torch.manual_seed(7)
    common = {
        "vocab_size": 256,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 256,
    }
    if family == "qwen3_5":
        from transformers import (
            Qwen3_5Config,
            Qwen3_5ForConditionalGeneration,
            Qwen3_5TextConfig,
            Qwen3_5VisionConfig,
        )

        text = Qwen3_5TextConfig(
            **common,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 2],
            },
        )
        # A 4x4 patch grid merged 2x2 gives four soft tokens per image.
        vision = Qwen3_5VisionConfig(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=4,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=64,
            deepstack_visual_indexes=[],
        )
        config = Qwen3_5Config(
            text_config=text,
            vision_config=vision,
            image_token_id=IMG,
            video_token_id=253,
            vision_start_token_id=BOI,
            vision_end_token_id=EOI,
        )
        model = Qwen3_5ForConditionalGeneration(config).eval()

        def media(images):
            return {
                "pixel_values": torch.randn(16 * len(images), 3 * 4 * 4),
                "image_grid_thw": torch.tensor([[1, 4, 4]] * len(images)),
            }

    else:
        from transformers import (
            Gemma4Config,
            Gemma4ForConditionalGeneration,
            Gemma4TextConfig,
            Gemma4VisionConfig,
        )

        text = Gemma4TextConfig(
            **common,
            sliding_window=4,
            hidden_size_per_layer_input=0,
            layer_types=["sliding_attention", "full_attention"],
        )
        # 16 patches pooled 2x2 give four soft tokens per image.
        vision = Gemma4VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            patch_size=4,
            pooling_kernel_size=2,
            position_embedding_size=16,
        )
        config = Gemma4Config(
            text_config=text,
            vision_config=vision,
            image_token_id=IMG,
            boi_token_id=BOI,
            eoi_token_id=EOI,
            video_token_id=253,
            audio_token_id=254,
            boa_token_id=248,
            eoa_token_index=249,
        )
        model = Gemma4ForConditionalGeneration(config).eval()

        def media(images):
            grid = [[r, c] for r in range(4) for c in range(4)]
            return {
                "pixel_values": torch.randn(len(images), 16, 3 * 4 * 4),
                "image_position_ids": torch.tensor([grid] * len(images)),
            }

    return model, media


def token_types(ids):
    import torch

    return torch.tensor([[1 if t == IMG else 0 for t in ids]])


@pytest.mark.parametrize("family", ["qwen3_5", "gemma4"])
async def test_cached_image_scoring_matches_full_forwards(family):
    """Prefix forward consumes the image; suffix batches score like full prompts."""
    import torch
    from hf_server import HFBackend

    torch.set_num_threads(2)
    model, media_for = tiny_vlm(family)
    torch.manual_seed(11)
    media = media_for([object()])
    shared = [1, 2, BOI, IMG, IMG, IMG, IMG, EOI, 3, 4]
    sequences = [
        shared + suffix for suffix in ([20, 21], [22, 23], [24, 25], [26], [27, 28, 29])
    ]
    media["mm_token_type_ids"] = token_types(sequences[0])
    branches = [Branch(str(i), ids, [30, 32, 34], [], "") for i, ids in enumerate(sequences)]
    compiled = CompiledRequest(None, branches, media, len(sequences[0]), len(shared))
    backend = HFBackend(model, max_batch_size=2)
    calls = []
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(
            (tuple(kwargs["input_ids"].shape), "pixel_values" in kwargs)
        ),
        with_kwargs=True,
    )
    result = await backend.score(compiled)
    hook.remove()
    # Only the prefix forward sees the image; the three suffix batches are text.
    assert calls == [((1, 10), True), ((2, 3), False), ((2, 2), False), ((1, 1), False)]
    with torch.inference_mode():
        for i, ids in enumerate(sequences):
            full = {k: v for k, v in media.items() if k != "mm_token_type_ids"}
            expected = model(
                input_ids=torch.tensor([ids]),
                mm_token_type_ids=token_types(ids),
                **full,
            ).logits[0, -1, [30, 32, 34]]
            np.testing.assert_allclose(
                result.logits[str(i)], expected.numpy(), atol=2e-5, rtol=2e-4
            )
    # An image-free request on the same backend must not inherit multimodal
    # state (M-RoPE deltas) from the previous request.
    text_sequences = [[5, 6, 7, 8] + suffix for suffix in ([20, 21], [22])]
    compiled = CompiledRequest(
        None, [Branch(str(i), ids, [30, 32, 34], [], "") for i, ids in enumerate(text_sequences)]
    )
    result = await backend.score(compiled)
    with torch.inference_mode():
        for i, ids in enumerate(text_sequences):
            expected = model(input_ids=torch.tensor([ids])).logits[0, -1, [30, 32, 34]]
            np.testing.assert_allclose(
                result.logits[str(i)], expected.numpy(), atol=2e-5, rtol=2e-4
            )


async def test_prefix_must_cover_the_images():
    import torch
    from hf_server import HFBackend

    model, media_for = tiny_vlm("gemma4")
    media = media_for([object()])
    ids = [1, BOI, IMG, IMG, IMG, IMG, EOI, 3]
    media["mm_token_type_ids"] = token_types(ids)
    # Branches diverge inside the image run: the prefix cannot hold the image.
    branches = [Branch("a", ids, [30, 32], [], ""), Branch("b", ids[:3] + [9, 9], [30, 32], [], "")]
    with pytest.raises(ValueError, match="within the shared prompt prefix"):
        await HFBackend(model).score(CompiledRequest(None, branches, media, len(ids), 7))


async def test_http_image_request_end_to_end():
    """Compile with fakes, score with a tiny real VLM, answer over HTTP."""
    import httpx
    import torch
    from hf_server import DecisionService, HFBackend, create_app

    torch.set_num_threads(2)
    model, media_for = tiny_vlm("qwen3_5")
    compiler = PromptCompiler(ImageTokenizer(), processor=FakeProcessor(media_for))
    service = DecisionService("test", compiler, HFBackend(model), advanced_metrics=True)
    questions = {
        **QUESTIONS,
        "level": {"type": "score", "instructions": "Red?", "criteria": ["no", "yes"]},
    }
    with_image = request([{"type": "text", "text": "Photo:"}, image_part(16)], questions=questions)
    text_only = request("Photo:", questions=questions)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test"
    ) as client:
        response = await client.post("/v1/classifier", json=with_image)
        assert response.status_code == 200, response.text
        body = response.json()
        assert set(body["answers"]) == {"color", "yes", "level"}
        assert body["metrics"]["scored_positions"] == 3
        assert body["usage"]["output_tokens"] == 0
        plain = (await client.post("/v1/classifier", json=text_only)).json()
        # Soft tokens count as input tokens: begin + 4 soft + end.
        assert body["usage"]["input_tokens"] == plain["usage"]["input_tokens"] + 6
        # Without a processor the same request is a validation error, not a crash.
        service.compiler = PromptCompiler(ImageTokenizer())
        assert (await client.post("/v1/classifier", json=with_image)).status_code == 422
