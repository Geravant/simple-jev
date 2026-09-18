"""Check logical token-prefix accounting and its HTTP presentation.

The parameterized cases cover duplicate leaves, prefix-contained leaves, nested
shared paths, disjoint roots, empty input, and order independence. The ASGI test
uses deliberately different backend counters to ensure public usage is derived
from compiled prompts rather than backend warmup/padded-work statistics.
"""

import httpx
import pytest
from conftest import FakeCompiler
from hf_server import BackendResult, DecisionService, create_app, unique_prompt_tokens


@pytest.mark.parametrize(
    "sequences,expected",
    [
        ([], 0),
        ([[1, 2, 3], [1, 2, 4]], 4),
        ([[1, 2], [1, 2], [1, 2, 3]], 3),
        ([[1, 2, 3], [1, 2, 4], [1, 5, 3]], 6),
        ([[1, 2], [3, 2]], 4),
    ],
)
def test_exact_hierarchical_prefix_union(sequences, expected):
    """Count distinct token-tree edges regardless of input leaf ordering."""
    assert unique_prompt_tokens(sequences) == expected
    assert unique_prompt_tokens(list(reversed(sequences))) == expected


async def test_classifier_alias_nonstreaming_and_usage_excludes_warmup():
    """Both routes return JSON and report logical usage, not fake engine counters."""

    class Backend:
        async def score(self, compiled):
            return BackendResult(
                {
                    q.branch_id: {label: 0.0 for label in q.output_labels}
                    for q in compiled.plan.questions
                },
                {
                    "prompt_tokens": 8,
                    "branch_prompt_tokens": 6,
                    "branch_output_tokens": 2,
                    "engine_output_tokens": 3,
                },
            )

    service = DecisionService("m", FakeCompiler(), Backend(), advanced_metrics=True)
    body = {
        "model": "m",
        "state": "shared",
        "questions": {k: {"type": "noul", "instructions": k} for k in ["a", "b"]},
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test"
    ) as client:
        for route in ["/v1/classifier", "/v1/systemone"]:
            r = await client.post(route, json=body)
            assert r.status_code == 200
            assert r.headers["content-type"] == "application/json"
            assert r.json()["usage"] == {"input_tokens": 4, "output_tokens": 2}
            assert r.json()["metrics"]["prompt_tokens"] == 8
