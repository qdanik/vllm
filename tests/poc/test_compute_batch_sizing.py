from vllm.poc.server.compute import _resolve_generation_batch_size, _resolve_pipelined_batch_size
from vllm.poc.server.models import PoCConfig
import vllm.poc.env as poc_env


def test_pipelined_batch_size_clamped_by_token_budget(monkeypatch):
    monkeypatch.setenv("POC_MAX_NUM_SEQS", "256")
    monkeypatch.setenv("POC_MAX_NUM_BATCHED_TOKENS", "32768")
    monkeypatch.setenv("POC_BATCH_SIZE_DEFAULT", "128")
    poc_env.disable_envs_cache()

    # 32768 / 1024 == 32 -> token budget is tighter than max_num_seqs.
    assert _resolve_pipelined_batch_size(1024, 128) == 32


def test_generation_batch_size_clamped_by_token_budget(monkeypatch):
    monkeypatch.setenv("POC_FORCE_BATCH_SIZE_DEFAULT_ON_INIT", "0")
    monkeypatch.setenv("POC_MAX_NUM_SEQS", "256")
    monkeypatch.setenv("POC_MAX_NUM_BATCHED_TOKENS", "32768")
    monkeypatch.setenv("POC_BATCH_SIZE_DEFAULT", "128")
    poc_env.disable_envs_cache()

    cfg = PoCConfig(
        block_hash="h",
        block_height=1,
        public_key="pk",
        seq_len=1024,
        k_dim=12,
        batch_size=128,
    )
    assert _resolve_generation_batch_size(cfg) == 32
