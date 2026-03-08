from vllm.poc.engine.bridge import _make_request
from vllm.poc.engine.params import PoCSchedulerParams
from vllm.v1.engine import EngineCoreRequestKind


def test_make_request_builds_poc_request_with_current_engine_schema():
    req = _make_request(
        request_id="poc-compat-1",
        client_index=0,
        priority=-1,
        seq_len=16,
        poc_params=PoCSchedulerParams(
            block_hash="h",
            public_key="pk",
            block_height=1,
            nonce=42,
            seq_len=16,
            k_dim=8,
        ),
    )

    assert req.request_id == "poc-compat-1"
    assert req.kind == EngineCoreRequestKind.POC
    assert req.poc_params is not None
    assert req.poc_params.nonce == 42
    assert req.prompt_token_ids == [0] * 16
    # vLLM >=0.17 moved eos to sampling_params; this should remain valid.
    assert req.sampling_params is not None
