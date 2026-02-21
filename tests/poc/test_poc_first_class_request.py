import time

from vllm.poc.v1.request import PoCRequest
from vllm.v1.core.sched.request_queue import PriorityRequestQueue
from vllm.v1.engine import EngineCoreRequestKind, PoCParams


def test_engine_core_request_kind_has_poc():
    assert EngineCoreRequestKind.POC.name == "POC"


def test_poc_params_is_stable_struct():
    params = PoCParams(
        block_hash="deadbeef",
        public_key="pk",
        nonces=[1, 2, 3],
        seq_len=8,
        k_dim=12,
    )
    assert params.block_hash == "deadbeef"
    assert params.nonces == [1, 2, 3]


def test_poc_priority_queue_orders_by_priority_then_arrival():
    q = PriorityRequestQueue()

    t0 = time.time()
    r1 = PoCRequest(
        request_id="b",
        client_index=0,
        arrival_time=t0,
        priority=0,
        poc_params=PoCParams(
            block_hash="h",
            public_key="pk",
            nonces=[1],
            seq_len=8,
            k_dim=12,
        ),
    )
    r2 = PoCRequest(
        request_id="a",
        client_index=0,
        arrival_time=t0 + 1,
        priority=-1,
        poc_params=r1.poc_params,
    )

    q.add_request(r1)
    q.add_request(r2)

    first = q.pop_request()
    second = q.pop_request()

    assert first.request_id == "a"  # higher priority (-1)
    assert second.request_id == "b"


def test_poc_request_is_hashable_identity():
    params = PoCParams(
        block_hash="h",
        public_key="pk",
        nonces=[1],
        seq_len=8,
        k_dim=12,
    )
    r = PoCRequest(
        request_id="x",
        client_index=0,
        arrival_time=time.time(),
        priority=0,
        poc_params=params,
    )
    s = {r}
    assert r in s
