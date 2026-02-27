# ruff: noqa: E501
"""Tests for PoC+Chat coexistence (scheduler-native).

These tests validate that PoC is scheduled as a normal request kind (no
parallel scheduler, no background worker thread/stream), can share the same
token budget with chat, and yields to chat via priority scheduling.
"""

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, patch

import pytest
import torch

from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.poc.v1.params import PoCParams
from vllm.poc.protocol.config import PoCConfig
from vllm.poc.api.generation import generation_loop
from vllm.poc.runtime.state import PoCGenerationStats
import vllm.poc.utils.env as env
from vllm.poc.constants import POC_REQUEST_PRIORITY
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequestKind
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


@pytest.fixture
def create_test_scheduler():
    """Create a minimal scheduler for testing scheduling semantics."""

    def _create(*, max_num_batched_tokens: int = 2048):
        model_config = ModelConfig(
            model="facebook/opt-125m",
            task="generate",
            tokenizer="facebook/opt-125m",
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="float16",
            seed=0,
            skip_tokenizer_init=True,
        )
        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            swap_space_bytes=0,
            cache_dtype="auto",
        )
        scheduler_config = SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=1024,
            is_encoder_decoder=False,
            policy="priority",
        )
        parallel_config = ParallelConfig(tensor_parallel_size=1)

        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            parallel_config=parallel_config,
        )

        kv_cache_config = KVCacheConfig(
            block_coverage=1.0,
            swap_ratio=0.0,
            min_num_sliding_window_blocks=0,
        )
        kv_cache_config.num_gpu_blocks = 100

        structured_output_manager = StructuredOutputManager()

        return Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=16,
        )

    return _create


class TestPoCSchedulerNativeCoexistence:
    """Scheduler-level coexistence tests.

    Skip on CPU: vLLM v1 scheduler instantiation validates CUDA-dependent
    config paths.
    """

    pytestmark = pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Scheduler tests require CUDA",
    )

    def test_poc_and_chat_can_share_token_budget(self, create_test_scheduler):
        scheduler = create_test_scheduler(max_num_batched_tokens=4096)

        chat_req = Request(
            request_id="chat-1",
            client_index=0,
            prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            arrival_time=time.time(),
            priority=0,
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
            pooling_params=None,
            eos_token_id=2,
        )

        seq_len = 256
        poc_req = Request(
            request_id="poc-1",
            client_index=0,
            prompt_token_ids=[0] * seq_len,
            arrival_time=time.time(),
            priority=POC_REQUEST_PRIORITY,
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
            pooling_params=None,
            eos_token_id=2,
            kind=EngineCoreRequestKind.POC,
            poc_params=PoCParams(
                block_hash="hash1",
                public_key="key1",
                block_height=100,
                nonce=123,
                seq_len=seq_len,
                k_dim=12,
            ),
        )

        scheduler.add_request(chat_req)
        scheduler.add_request(poc_req)

        out = scheduler.schedule()
        req_ids = {r.req_id for r in out.scheduled_new_reqs}
        assert req_ids == {"chat-1", "poc-1"}

    def test_poc_yields_to_chat_when_budget_tight(self, create_test_scheduler):
        # Budget only fits the chat prefill; PoC must not be scheduled.
        scheduler = create_test_scheduler(max_num_batched_tokens=8)

        chat_req = Request(
            request_id="chat-1",
            client_index=0,
            prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            arrival_time=time.time(),
            priority=0,
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
            pooling_params=None,
            eos_token_id=2,
        )
        poc_req = Request(
            request_id="poc-1",
            client_index=0,
            prompt_token_ids=[0] * 256,
            arrival_time=time.time(),
            priority=POC_REQUEST_PRIORITY,
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
            pooling_params=None,
            eos_token_id=2,
            kind=EngineCoreRequestKind.POC,
            poc_params=PoCParams(
                block_hash="hash1",
                public_key="key1",
                block_height=100,
                nonce=123,
                seq_len=256,
                k_dim=12,
            ),
        )

        scheduler.add_request(chat_req)
        scheduler.add_request(poc_req)

        out = scheduler.schedule()
        req_ids = {r.req_id for r in out.scheduled_new_reqs}
        assert req_ids == {"chat-1"}

    def test_poc_does_not_allocate_kv_blocks(self, create_test_scheduler):
        scheduler = create_test_scheduler(max_num_batched_tokens=4096)

        poc_req = Request(
            request_id="poc-1",
            client_index=0,
            prompt_token_ids=[0] * 256,
            arrival_time=time.time(),
            priority=POC_REQUEST_PRIORITY,
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
            pooling_params=None,
            eos_token_id=2,
            kind=EngineCoreRequestKind.POC,
            poc_params=PoCParams(
                block_hash="hash1",
                public_key="key1",
                block_height=100,
                nonce=123,
                seq_len=256,
                k_dim=12,
            ),
        )

        scheduler.add_request(poc_req)
        out = scheduler.schedule()

        assert len(out.scheduled_new_reqs) == 1
        new_req = out.scheduled_new_reqs[0]
        assert new_req.req_id == "poc-1"
        # KV-less: every KV cache group has an empty block list.
        assert all(len(group_blocks) == 0 for group_blocks in new_req.block_ids)


class TestGenerationLoopBackoff:
    @pytest.mark.asyncio
    async def test_generation_loop_backs_off_on_timeout(self):
        engine_client = AsyncMock()
        stop_event = asyncio.Event()

        config = PoCConfig(
            block_hash="hash",
            block_height=100,
            public_key="key",
            node_id=0,
            node_count=1,
            group_id=0,
            n_groups=1,
            seq_len=256,
            k_dim=12,
        )
        stats = PoCGenerationStats()

        call_count = 0

        async def _mock_poc_compute(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise TimeoutError("engine busy")
            stop_event.set()
            nonce = kwargs["nonce"]
            return {"nonces": [nonce], "vectors_b64": ["AAAA"]}

        engine_client.poc_compute = _mock_poc_compute

        with patch("vllm.poc.api.generation.POC_CHAT_BUSY_BACKOFF_SEC", 0.001):
            task = asyncio.create_task(
                generation_loop(engine_client, stop_event, None, config, stats)
            )
            await asyncio.sleep(0.1)
            stop_event.set()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)

        assert call_count >= 2
