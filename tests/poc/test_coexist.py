# ruff: noqa: E501
"""Tests for PoC+Chat coexistence (parallel execution)."""

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, patch

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.poc.protocol.config import PoCConfig
from vllm.poc.runtime.routes import _generation_loop
from vllm.poc.runtime.state import PoCGenerationStats
from vllm.poc.v1.constants import POC_REQUEST_PRIORITY
from vllm.poc.v1.request import PoCRequest
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import PoCParams
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager


@pytest.fixture
def mock_engine_client():
    """Create a mock engine client for testing."""
    client = AsyncMock()
    client.poc_request = AsyncMock()
    client.poc_request.return_value = {
        "artifacts": [],
    }
    return client


@pytest.fixture
def create_test_scheduler():
    """Create a minimal scheduler for testing PoC scheduling logic."""
    def _create():
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
            max_num_batched_tokens=2048,
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


class TestPoCScheduling:
    """Tests for PoC scheduling logic in Scheduler."""
    
    # Skip scheduler tests on CPU (VllmConfig validation fails without CUDA)
    pytestmark = pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Scheduler tests require CUDA (VllmConfig validation fails on CPU)"
    )
    
    def test_poc_not_scheduled_when_another_running(self, create_test_scheduler):
        """Test that PoC is not scheduled if another PoC is already running."""
        scheduler = create_test_scheduler()
        
        # Add two PoC requests
        poc_req1 = PoCRequest(
            request_id="poc-1",
            client_index=0,
            arrival_time=time.time(),
            priority=POC_REQUEST_PRIORITY,
            poc_params=PoCParams(
                block_hash="hash1",
                public_key="key1",
                nonces=[0, 1],
                seq_len=256,
                k_dim=12,
            ),
        )
        poc_req2 = PoCRequest(
            request_id="poc-2",
            client_index=0,
            arrival_time=time.time() + 1,
            priority=POC_REQUEST_PRIORITY,
            poc_params=PoCParams(
                block_hash="hash2",
                public_key="key2",
                nonces=[2, 3],
                seq_len=256,
                k_dim=12,
            ),
        )
        
        scheduler.add_request(poc_req1)
        scheduler.add_request(poc_req2)
        
        # First schedule should pick poc-1
        output = scheduler.schedule()
        assert output.poc_request is not None
        assert output.poc_request.request_id == "poc-1"
        assert scheduler._poc_running == poc_req1
        assert len(scheduler.poc_waiting) == 1
        
        # Second schedule should NOT pick poc-2 (poc-1 still running)
        output2 = scheduler.schedule()
        assert output2.poc_request is None  # No new PoC scheduled
        assert scheduler._poc_running == poc_req1  # Still running
        assert len(scheduler.poc_waiting) == 1  # poc-2 still waiting
    
    def test_poc_scheduled_after_previous_completes(self, create_test_scheduler):
        """Test that next PoC is scheduled after previous completes."""
        scheduler = create_test_scheduler()
        
        # Add two PoC requests
        poc_req1 = PoCRequest(
            request_id="poc-1",
            client_index=0,
            arrival_time=time.time(),
            priority=POC_REQUEST_PRIORITY,
            poc_params=PoCParams(
                block_hash="hash1",
                public_key="key1",
                nonces=[0, 1],
                seq_len=256,
                k_dim=12,
            ),
        )
        poc_req2 = PoCRequest(
            request_id="poc-2",
            client_index=0,
            arrival_time=time.time() + 1,
            priority=POC_REQUEST_PRIORITY,
            poc_params=PoCParams(
                block_hash="hash2",
                public_key="key2",
                nonces=[2, 3],
                seq_len=256,
                k_dim=12,
            ),
        )
        
        scheduler.add_request(poc_req1)
        scheduler.add_request(poc_req2)
        
        # First schedule picks poc-1
        output = scheduler.schedule()
        assert output.poc_request is not None
        assert output.poc_request.request_id == "poc-1"
        
        # Mark poc-1 as completed
        scheduler._poc_running.status = RequestStatus.FINISHED_STOPPED
        scheduler.poc_requests.pop(scheduler._poc_running.request_id)
        scheduler._poc_running = None
        
        # Now schedule should pick poc-2
        output2 = scheduler.schedule()
        assert output2.poc_request is not None
        assert output2.poc_request.request_id == "poc-2"
    
    def test_poc_scheduled_in_parallel_with_chat(self, create_test_scheduler):
        """Test that PoC is scheduled in parallel with normal chat requests."""
        scheduler = create_test_scheduler()
        
        # Add a normal chat request
        chat_req = Request(
            request_id="chat-1",
            client_index=0,
            prompt_token_ids=[1, 2, 3, 4],
            arrival_time=time.time(),
            priority=0,
        )
        
        # Add a PoC request
        poc_req = PoCRequest(
            request_id="poc-1",
            client_index=0,
            arrival_time=time.time(),
            priority=POC_REQUEST_PRIORITY,
            poc_params=PoCParams(
                block_hash="hash1",
                public_key="key1",
                nonces=[0, 1],
                seq_len=256,
                k_dim=12,
            ),
        )
        
        scheduler.add_request(chat_req)
        scheduler.add_request(poc_req)
        
        # Schedule should handle both
        output = scheduler.schedule()
        
        # Chat request should be scheduled
        assert len(output.scheduled_new_reqs) == 1
        assert output.scheduled_new_reqs[0].req_id == "chat-1"
        
        # PoC request should also be scheduled (in parallel)
        assert output.poc_request is not None
        assert output.poc_request.request_id == "poc-1"
    
    def test_poc_priority_queue_ordering(self, create_test_scheduler):
        """Test that PoC requests are scheduled by priority."""
        scheduler = create_test_scheduler()
        
        # Add PoC requests with different priorities (lower = higher priority)
        poc_low_priority = PoCRequest(
            request_id="poc-low",
            client_index=0,
            arrival_time=time.time(),
            priority=5,  # Lower priority
            poc_params=PoCParams(
                block_hash="hash1",
                public_key="key1",
                nonces=[0, 1],
                seq_len=256,
                k_dim=12,
            ),
        )
        poc_high_priority = PoCRequest(
            request_id="poc-high",
            client_index=0,
            arrival_time=time.time() + 1,
            priority=1,  # Higher priority (smaller number)
            poc_params=PoCParams(
                block_hash="hash2",
                public_key="key2",
                nonces=[2, 3],
                seq_len=256,
                k_dim=12,
            ),
        )
        
        # Add in wrong order (low priority first)
        scheduler.add_request(poc_low_priority)
        scheduler.add_request(poc_high_priority)
        
        # Schedule should pick high priority first
        output = scheduler.schedule()
        assert output.poc_request is not None
        assert output.poc_request.request_id == "poc-high"


# Note: AsyncLLMEngine tests are skipped because the full engine stack 
# is difficult to mock. The scheduler logic is tested above.


class TestGenerationLoopBackoff:
    """Tests for generation loop backoff behavior."""
    
    @pytest.mark.asyncio
    async def test_generation_loop_backs_off_on_skip(self, mock_engine_client):
        """Test that generation loop backs off when engine times out (engine busy)."""
        stop_event = asyncio.Event()
        config = PoCConfig(
            block_hash="hash",
            block_height=100,
            public_key="key",
            node_id=0,
            node_count=1,
            group_id=0,
            n_groups=1,
            batch_size=4,
            seq_len=256,
            k_dim=12,
        )
        stats = PoCGenerationStats()
        
        call_count = 0
        async def mock_poc_request(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise TimeoutError("engine busy")
            stop_event.set()
            return {"nonces": [], "vectors_b64": []}
        
        mock_engine_client.poc_request = mock_poc_request
        
        with patch("vllm.poc.runtime.routes.env.POC_CHAT_BUSY_BACKOFF_SEC", 0.001):
            task = asyncio.create_task(
                _generation_loop(mock_engine_client, stop_event, None, config, stats)
            )
            await asyncio.sleep(0.1)
            stop_event.set()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)
        
        assert call_count >= 2
