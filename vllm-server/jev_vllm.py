"""Simple Jev classifier API served by vLLM.

Same request/response contract as hf-server (common owns validation, the
prompt template and the answer math), but inference goes through a vLLM
OpenAI-compatible server instead of eager Transformers:

    ClassifierRequest -> prepare_prompt -> one chat completion per branch
    (assistant turn prefilled with the answer prefix, max_tokens=1,
    top_logprobs) -> label log-probabilities -> common.build_response

vLLM's automatic prefix caching makes the branches cheap: they share the
system briefing, the context (including images) and the reminder, so after
the first branch has run only each branch's short tail is prefilled. The first
branch is sent alone and the rest fan out concurrently, so the shared prefix
is computed exactly once rather than once per branch in the same step.

Probabilities are the softmax over the permitted labels of their next-token
log-probabilities, which equals the softmax over their logits (the log-partition
term cancels), so answers match the HF server's for the same model. A label
that falls outside the requested top_logprobs window counts as -inf.

Environment: MODEL (path or Hub id, default /repository), MODEL_ALIAS (extra
served name), TEMPLATE (v1 | v2c), PORT (this API, default 8000), VLLM_PORT
(default 8001), MAX_MODEL_LEN, MAX_IMAGES, GPU_MEMORY_UTILIZATION, TOP_LOGPROBS
(default 64), VLLM_EXTRA_ARGS (appended verbatim).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

_checkout_root = Path(__file__).resolve().parent.parent
if (_checkout_root / "common" / "prompt_builder.py").is_file():
    sys.path.insert(0, str(_checkout_root))

from common import ClassifierRequest, build_response, prepare_prompt  # noqa: E402
from common.prompt_builder import DEFAULT_TEMPLATE_VERSION, canonical  # noqa: E402

log = logging.getLogger("jev-vllm")

MODEL = os.environ.get("MODEL", "/repository")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "").strip()
TEMPLATE = os.environ.get("TEMPLATE", DEFAULT_TEMPLATE_VERSION)
PORT = int(os.environ.get("PORT", "8000"))
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8001"))
VLLM_URL = os.environ.get("VLLM_URL", f"http://127.0.0.1:{VLLM_PORT}")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "4"))
TOP_LOGPROBS = int(os.environ.get("TOP_LOGPROBS", "64"))
MAX_REQUEST_BRANCHES = int(os.environ.get("MAX_REQUEST_BRANCHES", "100"))
SERVED_NAMES = {MODEL} | ({MODEL_ALIAS} if MODEL_ALIAS else set())


# Prompt assembly


def branch_messages(plan, request: ClassifierRequest, question) -> list[dict]:
    """Chat messages for one branch: classifier briefing as the system turn,
    the caller's context (state, or their conversation with images), the
    reminder plus this branch's question as a final user turn, and the answer
    prefix as an unfinished assistant turn (continue_final_message)."""
    system = plan.system_prompt_prefix + plan.prefix_instruction
    tail = plan.suffix_instruction + question.instruction
    if request.messages is None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"State:\n{canonical(request.state)}\n\n" + tail},
        ]
    else:
        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        if messages[0]["role"] == "system":
            messages[0]["content"] = system + "\n" + messages[0]["content"]
        else:
            messages.insert(0, {"role": "system", "content": system})
        messages.append({"role": "user", "content": tail})
    messages.append({"role": "assistant", "content": question.answer_prefix})
    return messages


def label_logprobs(top: list[dict], labels) -> dict[str, float]:
    """Map each permitted label to its next-token log-probability from a vLLM
    top_logprobs list ([{"token": "A", "logprob": -0.1}, ...]); -inf when the
    label is not in the window. Tokens are compared after stripping the
    tokenizer's space marker, because the answer prefix ends in a quote or a
    space depending on the question type."""
    seen = {}
    for entry in top:
        tok = (entry.get("token") or "").replace("▁", " ").strip().strip('"')
        if tok and tok not in seen:
            seen[tok] = float(entry["logprob"])
    found = {label: seen[label] for label in labels if label in seen}
    if not found:
        raise ValueError("No permitted label among the top tokens: "
                         + ", ".join(repr(e.get("token")) for e in top[:12]))
    # A label outside the window is rarer than the rarest one inside it; a
    # finite floor keeps the scorer's softmax well defined (-inf is rejected).
    floor = min(found.values()) - 8.0
    return {label: found.get(label, floor) for label in labels}


# vLLM client


class VLLM:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def healthy(self) -> bool:
        try:
            r = await self.client.get(self.base_url + "/health")
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    async def score_branch(self, model: str, messages: list[dict], labels, top_k: int):
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": top_k,
            "continue_final_message": True,
            "add_generation_prompt": False,
        }
        r = await self.client.post(self.base_url + "/v1/chat/completions", json=body)
        r.raise_for_status()
        j = r.json()
        choice = j["choices"][0]
        content = ((choice.get("logprobs") or {}).get("content") or [])
        top = content[0].get("top_logprobs", []) if content else []
        usage = j.get("usage") or {}
        return label_logprobs(top, labels), int(usage.get("prompt_tokens", 0))

    async def close(self):
        await self.client.aclose()


# Service


class Service:
    def __init__(self, vllm: VLLM, model: str, template: str = TEMPLATE,
                 top_logprobs: int = TOP_LOGPROBS, max_branches: int = MAX_REQUEST_BRANCHES,
                 advanced: bool | None = None):
        self.vllm, self.model, self.template = vllm, model, template
        self.top_logprobs, self.max_branches = top_logprobs, max_branches
        self.advanced = (os.environ.get("ENABLE_OPEN_JEV_ADVANCED_METRICS", "").strip().lower()
                         in {"1", "true", "yes", "on"} if advanced is None else advanced)

    async def classify(self, request: ClassifierRequest) -> dict:
        if request.model not in SERVED_NAMES and request.model != self.model:
            raise ValueError(f"Loaded model is {self.model!r}")
        if request.tools or request.mm_processor_kwargs or request.media_io_kwargs:
            raise ValueError("tools and media options are not supported")
        if len(request.questions) > self.max_branches:
            raise ValueError(f"Request has {len(request.questions)} scoring branches; "
                             f"maximum is {self.max_branches}")
        if request.messages:
            images = sum(1 for m in request.messages if isinstance(m.content, list)
                         for p in m.content if isinstance(p, dict)
                         and p.get("type") in {"image", "image_url"})
            if images > MAX_IMAGES:
                raise ValueError(f"At most {MAX_IMAGES} images per request")
        plan = prepare_prompt(request, version=self.template)
        start = time.perf_counter()
        top_k = self.top_logprobs
        model = request.model if request.model in SERVED_NAMES else self.model
        questions = list(plan.questions)
        # Warm the shared prefix with the first branch, then fan out.
        first, tokens = await self.vllm.score_branch(
            model, branch_messages(plan, request, questions[0]), questions[0].output_labels, top_k)
        rest = await asyncio.gather(*[
            self.vllm.score_branch(model, branch_messages(plan, request, q), q.output_labels, top_k)
            for q in questions[1:]])
        logits = {questions[0].branch_id: first}
        for q, (lp, _) in zip(questions[1:], rest):
            logits[q.branch_id] = lp
        response = build_response(plan, logits, input_tokens=tokens, output_tokens=0,
                                  advanced=self.advanced)
        if self.advanced:
            response["metadata"] = {**response["metadata"], "backend": "vllm",
                                    "usage_accounting": "first_branch_prompt_tokens"}
            response["metrics"] = {"backend": "vllm", "backend_seconds": time.perf_counter() - start,
                                   "scored_positions": len(questions), "top_logprobs": top_k}
        return response


# HTTP


def create_app(service: Service, ready) -> FastAPI:
    """ready: () -> bool. /health answers 503 until vLLM serves, so a host's
    readiness probe waits for the weights."""
    app = FastAPI(title="Simple Jev (vLLM)")

    @app.get("/health")
    async def health():
        if not ready():
            return JSONResponse({"status": "starting"}, status_code=503)
        return {"status": "ready", "model": service.model}

    async def classify(request: Request):
        try:
            body = ClassifierRequest.model_validate(await request.json())
            return await service.classify(body)
        except (ValidationError, ValueError) as exc:
            return JSONResponse({"error": {"message": str(exc)[:500], "type": "invalid_request_error",
                                           "code": 422}}, status_code=422)
        except httpx.HTTPStatusError as exc:
            return JSONResponse({"error": {"message": f"vllm {exc.response.status_code}: "
                                           f"{exc.response.text[:300]}", "type": "backend_error",
                                           "code": 502}}, status_code=502)

    app.post("/v1/classifier")(classify)
    app.post("/v1/systemone")(classify)

    @app.post("/v1/chat/completions")
    async def passthrough(request: Request):
        """Raw access to the vLLM server behind this API (the host's auth on this
        port applies): generation on the same model, streaming when the body
        asks for it, so a client can converse on the box that classifies."""
        body = await request.body()
        try:
            wants_stream = bool(json.loads(body).get("stream"))
        except Exception:  # noqa: BLE001 — vLLM reports malformed JSON itself
            wants_stream = False
        url = service.vllm.base_url + "/v1/chat/completions"
        headers = {"content-type": "application/json"}
        if not wants_stream:
            r = await service.vllm.client.post(url, content=body, headers=headers)
            ct = r.headers.get("content-type", "")
            return JSONResponse(r.json() if ct.startswith("application/json") else {"raw": r.text[:2000]},
                                status_code=r.status_code)
        req = service.vllm.client.build_request("POST", url, content=body, headers=headers)
        r = await service.vllm.client.send(req, stream=True)
        if r.status_code != 200:
            text = (await r.aread()).decode(errors="replace")
            await r.aclose()
            return JSONResponse({"error": {"message": text[:1000], "code": r.status_code}},
                                status_code=r.status_code)

        async def relay():
            try:
                async for chunk in r.aiter_raw():
                    yield chunk
            except httpx.StreamConsumed:
                yield r.content          # body already in memory (small replies, test transports)
            finally:
                await r.aclose()

        return StreamingResponse(relay(), media_type="text/event-stream",
                                 headers={"cache-control": "no-cache", "x-accel-buffering": "no"})

    return app


def start_vllm(model: str) -> subprocess.Popen:
    args = ["vllm", "serve", model, "--host", "127.0.0.1", "--port", str(VLLM_PORT),
            "--max-model-len", str(MAX_MODEL_LEN), "--max-logprobs", str(TOP_LOGPROBS),
            "--limit-mm-per-prompt", f'{{"image": {MAX_IMAGES}}}',
            "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"),
            "--enable-prefix-caching", "--dtype", os.environ.get("DTYPE", "bfloat16")]
    if MODEL_ALIAS:
        args += ["--served-model-name", model, MODEL_ALIAS]
    args += shlex.split(os.environ.get("VLLM_EXTRA_ARGS", ""))
    log.info("starting: %s", " ".join(args))
    return subprocess.Popen(args)


def main():
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    proc = start_vllm(MODEL)
    vllm = VLLM(VLLM_URL)
    state = {"ready": False}
    service = Service(vllm, MODEL)

    app = create_app(service, lambda: state["ready"])

    @app.on_event("startup")
    async def wait_for_vllm():
        async def poll():
            while not state["ready"]:
                if proc.poll() is not None:
                    log.error("vllm exited with %s", proc.returncode)
                    os._exit(1)
                if await vllm.healthy():
                    state["ready"] = True
                    log.info("vllm ready on %s (template %s)", VLLM_URL, TEMPLATE)
                    break
                await asyncio.sleep(2)
        asyncio.ensure_future(poll())

    @app.on_event("shutdown")
    async def stop():
        with contextlib.suppress(Exception):
            proc.terminate()
        await vllm.close()

    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
