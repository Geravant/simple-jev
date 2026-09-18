"""Service admission and cancellation tests, without numerical inference.

The historical filename is retained, but answer math belongs to common/tests.
A controllable fake backend holds an active slot until cancellation, allowing a
deterministic assertion that overload is detected and admission is released.
No sleeps, downloads, or accelerator availability are required.
"""

import asyncio

import pytest
from conftest import FakeCompiler
from hf_server import DecisionService, OverloadedError


async def test_cancellation_releases_admission_and_propagates():
    """A saturated zero-wait queue rejects work, then cancellation frees its slot."""
    entered, cancelled = asyncio.Event(), asyncio.Event()

    class Backend:
        async def score(self, compiled):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    service = DecisionService(
        "m", FakeCompiler(), Backend(), concurrency=1, queue_size=0
    )
    request = {
        "model": "m",
        "state": "",
        "questions": {"q": {"type": "noul", "instructions": "True?"}},
    }
    task = asyncio.create_task(service.classify(request))
    await entered.wait()
    with pytest.raises(OverloadedError):
        await service.classify(request)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert service._inflight == 0


async def test_question_limit_rejects_before_inference():
    """Reject too many questions before touching even a missing backend."""
    service = DecisionService("m", FakeCompiler(), None, max_request_branches=1)
    with pytest.raises(ValueError, match="maximum is 1"):
        await service.classify(
            {
                "model": "m",
                "state": "",
                "questions": {
                    key: {"type": "noul", "instructions": "True?"} for key in ["a", "b"]
                },
            }
        )
