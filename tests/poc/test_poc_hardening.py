import asyncio

import pytest
import torch

from vllm.poc.engine.bridge import PoCWaiterEntry
from vllm.poc.engine.dedup import PoCDedupRegistry
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


def test_poc_registry_dedup_single_execution_multiple_subscribers():
    reg = PoCDedupRegistry()
    r1 = reg.on_accept(
        identity_key="poc:abc",
        request_id="r1",
    )
    assert r1.accepted is True
    assert r1.canonical_request_id == "r1"

    r2 = reg.on_accept(
        identity_key="poc:abc",
        request_id="r2",
    )
    assert r2.accepted is False
    assert r2.canonical_request_id == "r1"  # still in-flight

    aliases = reg.on_executed_and_emitted(
        request_id="r1",
    )
    assert aliases == []

    # No caching: once canonical finishes, identity may be accepted again.
    r3 = reg.on_accept(identity_key="poc:abc", request_id="r3")
    assert r3.accepted is True
    assert r3.canonical_request_id == "r3"


def test_poc_registry_abort_prevents_alias_emission():
    reg = PoCDedupRegistry()

    reg.on_accept(identity_key="poc:abc", request_id="r1")
    reg.on_abort("r1")
    assert reg.is_aborted("r1") is True

    # Completion cleans up in-flight + aborted state.
    reg.on_executed_and_emitted(
        request_id="r1",
    )
    assert reg.is_aborted("r1") is False

    # Identity is no longer stuck in-flight.
    r2 = reg.on_accept(identity_key="poc:abc", request_id="r2")
    assert r2.accepted is True


def test_poc_registry_abort_count_is_idempotent():
    reg = PoCDedupRegistry()

    reg.on_accept(identity_key="poc:abc", request_id="r1")
    reg.on_abort("r1")
    reg.on_abort("r1")

    assert reg.is_aborted("r1") is True


def _make_poc_params() -> PoCSchedulerParams:
    return PoCSchedulerParams(
        block_hash="0x00",
        public_key="pk",
        block_height=1,
        nonce=1,
        seq_len=8,
        k_dim=16,
        r_target=1.0,
        return_vectors=False,
    )


def test_poc_input_batch_rejects_non_empty_block_ids():
    batch = InputBatch(
        max_num_reqs=1,
        max_model_len=16,
        max_num_batched_tokens=16,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=256,
        block_sizes=[16],
        kernel_block_sizes=[16],
        max_num_blocks_per_req=[2],
    )

    req = CachedRequestState(
        req_id="poc-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=0,
        output_token_ids=[],
        poc_params=_make_poc_params(),
    )

    with pytest.raises(AssertionError, match=r"PoC request must be KV-less"):
        batch.add_request(req)


def test_poc_input_batch_marks_kvless_row_as_pad():
    batch = InputBatch(
        max_num_reqs=1,
        max_model_len=16,
        max_num_batched_tokens=16,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=256,
        block_sizes=[16],
        kernel_block_sizes=[16],
        max_num_blocks_per_req=[2],
    )

    req = CachedRequestState(
        req_id="poc-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=([],),
        num_computed_tokens=0,
        output_token_ids=[],
        poc_params=_make_poc_params(),
    )

    req_index = batch.add_request(req)
    row = batch.block_table[0].get_numpy_array()[req_index]
    assert (row == -1).all()
