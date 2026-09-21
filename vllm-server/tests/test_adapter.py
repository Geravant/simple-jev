"""vllm-server adapter: prompt assembly, logprob → answer math, HTTP, with a
fake vLLM behind httpx.MockTransport. No GPU, no vLLM install needed."""
import asyncio
import json
import math
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jev_vllm as jv  # noqa: E402
from common import prepare_prompt  # noqa: E402

REQ = {"model": "/repository", "messages": [
    {"role": "user", "content": [{"type": "text", "text": "Photo:"},
                                 {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}],
    "questions": {
        "color": {"type": "choice", "instructions": "Color?", "criteria": {"red": None, "blue": None}},
        "level": {"type": "score", "instructions": "Level?", "criteria": ["low", "mid", "high"]},
        "yes": {"type": "noul", "instructions": "Red?"},
    }}


def test_branch_messages_keep_context_and_prefill_answer():
    from common import ClassifierRequest
    req = ClassifierRequest.model_validate(REQ)
    plan = prepare_prompt(req, version="v2c")
    msgs = jv.branch_messages(plan, req, plan.questions[0])
    assert [m["role"] for m in msgs] == ["system", "user", "user", "assistant"]
    assert msgs[0]["content"].startswith(plan.system_prompt_prefix)
    assert msgs[1]["content"][1]["type"] == "image_url"          # caller's image untouched
    assert msgs[2]["content"].startswith(plan.suffix_instruction)
    assert "Question: Color?" in msgs[2]["content"]
    assert msgs[3] == {"role": "assistant", "content": '{"answer": "'}
    # state form
    req2 = ClassifierRequest.model_validate({**REQ, "messages": None, "state": "The car is red."})
    plan2 = prepare_prompt(req2, version="v2c")
    msgs2 = jv.branch_messages(plan2, req2, plan2.questions[1])
    assert [m["role"] for m in msgs2] == ["system", "user", "assistant"]
    assert msgs2[1]["content"].startswith('State:\n"The car is red."')
    assert msgs2[2]["content"] == '{"answer": "'


def test_label_logprobs_matches_tokens_and_fills_missing():
    top = [{"token": "A", "logprob": -0.1}, {"token": "▁B", "logprob": -2.3}, {"token": "?", "logprob": -9}]
    lp = jv.label_logprobs(top, ("A", "B", "C"))
    assert lp == {"A": -0.1, "B": -2.3, "C": -2.3 - 8.0}      # missing → finite floor
    assert jv.label_logprobs([{"token": '"A', "logprob": -1}], ("A",)) == {"A": -1}
    with pytest.raises(ValueError, match="No permitted label"):
        jv.label_logprobs([{"token": "The", "logprob": -1}], ("A", "B"))


@pytest.fixture
def fake_vllm(monkeypatch):
    calls = []

    def handler(request: httpx.Request):
        if request.url.path == "/health":
            return httpx.Response(200)
        body = json.loads(request.content)
        calls.append(body)
        assert body["max_tokens"] == 1 and body["continue_final_message"] is True
        assert body["messages"][-1]["content"] == '{"answer": "'    # every v2c branch is quoted
        tail = body["messages"][-2]["content"]
        # Answer by branch kind: A/B for the choice, A/B/C for the score
        # levels, the nine rating digits for noul.
        if "Truth rubric" in tail:
            top = [{"token": str(i), "logprob": math.log(0.1 if i < 9 else 0.2)} for i in range(1, 10)]
        elif "Levels:" in tail:
            top = [{"token": s, "logprob": math.log(p)} for s, p in zip("ABC", [0.1, 0.2, 0.7])]
        else:
            top = [{"token": "A", "logprob": math.log(0.75)}, {"token": "B", "logprob": math.log(0.25)}]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "A"},
                         "logprobs": {"content": [{"token": "A", "top_logprobs": top}]}}],
            "usage": {"prompt_tokens": 1234}})

    vllm = jv.VLLM("http://vllm.test")
    vllm.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(jv, "SERVED_NAMES", {"/repository", "alias/model"})
    return vllm, calls


def test_service_scores_all_branches_with_one_warm_then_fanout(fake_vllm):
    vllm, calls = fake_vllm
    svc = jv.Service(vllm, "/repository", template="v2c", advanced=True)
    from common import ClassifierRequest
    out = asyncio.run(svc.classify(ClassifierRequest.model_validate(REQ)))
    assert set(out["answers"]) == {"color", "level", "yes"}
    assert out["answers"]["color"]["choice"] == "red"
    assert out["answers"]["color"]["probabilities"]["red"] == pytest.approx(0.75)
    assert out["answers"]["level"]["score"] == pytest.approx(0.1 * 0 + 0.2 * 1 + 0.7 * 2, abs=1e-6)
    assert 0.01 <= out["answers"]["yes"]["noul"] <= 0.99
    assert out["usage"] == {"input_tokens": 1234, "output_tokens": 0}
    assert out["metrics"]["backend"] == "vllm" and out["metrics"]["scored_positions"] == 3
    assert len(calls) == 3
    assert all(c["top_logprobs"] <= jv.TOP_LOGPROBS for c in calls)


def test_http_validation_and_health(fake_vllm):
    vllm, _ = fake_vllm
    svc = jv.Service(vllm, "/repository", template="v2c")
    ready = {"ok": False}
    app = jv.create_app(svc, lambda: ready["ok"])

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            assert (await c.get("/health")).status_code == 503
            ready["ok"] = True
            assert (await c.get("/health")).json()["status"] == "ready"
            r = await c.post("/v1/classifier", json=REQ)
            assert r.status_code == 200 and "answers" in r.json()
            r = await c.post("/v1/classifier", json={**REQ, "model": "other"})
            assert r.status_code == 422 and "Loaded model" in r.json()["error"]["message"]
            r = await c.post("/v1/systemone", json={**REQ, "model": "alias/model"})
            assert r.status_code == 200
    asyncio.run(run())
