import time

from vllm.poc.v1.scheduler_params import PoCSchedulerParams
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import PriorityRequestQueue
from vllm.v1.engine import EngineCoreRequestKind
from vllm.v1.request import Request


def test_engine_core_request_kind_has_poc():
    assert EngineCoreRequestKind.POC.name == "POC"


def test_poc_params_is_stable_struct():
    params = PoCSchedulerParams(
        block_hash="deadbeef",
        public_key="pk",
        block_height=100,
        nonce=1,
        seq_len=8,
        k_dim=12,
    )
    assert params.block_hash == "deadbeef"
    assert params.nonce == 1


def test_poc_priority_queue_orders_by_priority_then_arrival():
    q = PriorityRequestQueue()

    t0 = time.time()
    r1 = Request(
        request_id="b",
        client_index=0,
        arrival_time=t0,
        priority=0,
        prompt_token_ids=[0] * 8,
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        pooling_params=None,
        eos_token_id=2,
        kind=EngineCoreRequestKind.POC,
        poc_params=PoCSchedulerParams(
            block_hash="h",
            public_key="pk",
            block_height=100,
            nonce=1,
            seq_len=8,
            k_dim=12,
        ),
    )
    r2 = Request(
        request_id="a",
        client_index=0,
        arrival_time=t0 + 1,
        priority=-1,
        prompt_token_ids=[0] * 8,
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        pooling_params=None,
        eos_token_id=2,
        kind=EngineCoreRequestKind.POC,
        poc_params=r1.poc_params,
    )

    q.add_request(r1)
    q.add_request(r2)

    first = q.pop_request()
    second = q.pop_request()

    assert first.request_id == "a"  # higher priority (-1)
    assert second.request_id == "b"
