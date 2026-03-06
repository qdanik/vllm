import asyncio

import pytest
import torch

from vllm.poc.engine.bridge import PoCWaiterEntry
from vllm.poc.engine.output import resolve_poc_outputs
from vllm.poc.engine.params import PoCSchedulerParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequestKind, FinishReason
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


def test_output_handler_drops_orphan_poc_outputs():
    # PoC output without a matching waiter must never go through normal
    # token streaming / OutputProcessor.
    out = EngineCoreOutput(
        request_id="poc-1",
        new_token_ids=[],
        finish_reason=FinishReason.STOP,
        poc_result={"nonces": [1], "vectors_b64": ["AAAA"]},
        kind=EngineCoreRequestKind.POC,
    )

    remaining, resolved, orphaned = resolve_poc_outputs([out], {})
    assert remaining == []
    assert resolved == 0
    assert orphaned == 1


def test_output_handler_duplicate_poc_outputs_resolve_once_then_drop():
    # If the engine accidentally emits multiple PoC outputs for the same
    # request_id, the frontend must resolve at most once and never forward
    # PoC outputs into token streaming.
    loop = asyncio.new_event_loop()
    try:
        fut: asyncio.Future[dict] = loop.create_future()
        waiters = {"poc-1": PoCWaiterEntry(future=fut)}

        outs = [
            EngineCoreOutput(
                request_id="poc-1",
                new_token_ids=[],
                finish_reason=FinishReason.STOP,
                poc_result={"nonces": [1], "vectors_b64": ["AAAA"]},
                kind=EngineCoreRequestKind.POC,
            ),
            EngineCoreOutput(
                request_id="poc-1",
                new_token_ids=[],
                finish_reason=FinishReason.STOP,
                poc_result={"nonces": [1], "vectors_b64": ["BBBB"]},
                kind=EngineCoreRequestKind.POC,
            ),
        ]

        remaining, resolved, orphaned = resolve_poc_outputs(outs, waiters)
        assert remaining == []
        assert resolved == 1
        assert orphaned == 1
        assert fut.done()
        assert fut.result()["vectors_b64"] == ["AAAA"]
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_output_handler_resolves_waiter_once():
    fut: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
    waiters = {"poc-1": PoCWaiterEntry(future=fut)}

    out = EngineCoreOutput(
        request_id="poc-1",
        new_token_ids=[],
        finish_reason=FinishReason.STOP,
        poc_result={"nonces": [1], "vectors_b64": ["AAAA"]},
        kind=EngineCoreRequestKind.POC,
    )

    remaining, resolved, orphaned = resolve_poc_outputs([out], waiters)
    assert remaining == []
    assert resolved == 1
    assert orphaned == 0
    assert fut.done()
    assert fut.result()["nonces"] == [1]


