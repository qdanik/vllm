#!/usr/bin/env python3
"""Test PoC scheduler integration with realistic parameters.

This test verifies PoC scheduler integration using the same parameters
as poc.py profiler (model, seq_len, k_dim, test data).
"""
import sys
import time
from typing import Any

# Add parent directory to path
sys.path.insert(0, '.')

import torch
from vllm import SamplingParams
from vllm.config import VllmConfig, CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, FullAttentionSpec
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.engine import EngineCoreRequestKind, PoCParams
from vllm.v1.request import Request, RequestStatus
from vllm.poc.v1.request import PoCRequest
from vllm.poc.v1.constants import POC_REQUEST_PRIORITY
from vllm.utils.hashing import sha256

# Real test data from poc.py
PUBLIC_KEY = "02e0f3b6b7f832ead7af2a235b9b27715a4d586b0fa108e735f0676a5086479225"
BLOCK_HASH = "8d148df1530d06a3412acd3deda4db16bae780eefdd160e081e6f878417de92a"
MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
SEQ_LEN = 1024
K_DIM = 12
EOS_TOKEN_ID = 50256

# Initialize hash function
_none_hash_initialized = False


def create_normal_request(request_id: str, prompt_token_ids: list[int], block_size: int = 16) -> Request:
    """Helper to create normal Request with proper parameters."""
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True
    
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        mm_features=None,
        eos_token_id=EOS_TOKEN_ID,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )



def create_scheduler():
    """Create a scheduler instance with realistic config matching poc.py."""
    block_size = 16
    
    try:
        model_config = ModelConfig(
            model=MODEL,
            task="generate",
            tokenizer=MODEL,
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="float16",
            seed=0,
            skip_tokenizer_init=True,
        )
        
        cache_config = CacheConfig(
            block_size=block_size,
            gpu_memory_utilization=0.9,
            swap_space_bytes=0,
            cache_dtype="auto",
        )
        
        scheduler_config = SchedulerConfig(
            max_num_seqs=32,
            max_num_batched_tokens=8192,
            max_model_len=240000,  # Same as poc.py
            is_encoder_decoder=False,
            policy="priority",  # Use priority queue for PoC priority testing
        )
        
        parallel_config = ParallelConfig(
            tensor_parallel_size=4,  # Same as poc.py
        )
        
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            parallel_config=parallel_config,
        )
        
        # Create KV cache config (simplified for testing)
        num_blocks = 10000
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        )
        
        # CRITICAL: Set num_gpu_blocks in cache_config
        vllm_config.cache_config.num_gpu_blocks = num_blocks
        
        structured_output_manager = StructuredOutputManager(vllm_config)
        
        scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            log_stats=False,
        )
        
        return scheduler
    except Exception as e:
        print(f"\nError creating scheduler: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_poc_request_added_to_separate_queue():
    """Test that PoC requests are added to the poc_waiting queue, not waiting queue."""
    print("Test 1: PoC request added to separate queue...", end=" ")
    
    try:
        scheduler = create_scheduler()
        poc_req = PoCRequest(
            request_id="poc-1",
            client_index=0,
            arrival_time=time.time(),
            priority=POC_REQUEST_PRIORITY,
            poc_params=PoCParams(
                block_hash=BLOCK_HASH,
                public_key=PUBLIC_KEY,
                nonces=[1, 3, 5],  # Real nonces from VALIDATION_SAMPLE
                seq_len=SEQ_LEN,
                k_dim=K_DIM,
            ),
        )
        
        scheduler.add_request(poc_req)
        
        # Check that PoC request is in poc_waiting queue
        assert len(scheduler.poc_waiting) == 1, f"Expected 1 PoC in waiting queue, got {len(scheduler.poc_waiting)}"
        assert len(scheduler.waiting) == 0, f"Expected 0 normal requests, got {len(scheduler.waiting)}"
        assert "poc-1" in scheduler.poc_requests, "PoC request not in poc_requests dict"
        assert poc_req.status == RequestStatus.WAITING, f"Expected WAITING status, got {poc_req.status}"
        
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_poc_request_scheduled_with_high_priority():
    """Test that PoC requests are scheduled before normal requests due to priority."""
    print("Test 2: PoC request scheduled with high priority...", end=" ")
    
    scheduler = create_scheduler()
    
    # Add a normal generate request with default priority (0)
    normal_req = create_normal_request(
        request_id="normal-1",
        prompt_token_ids=[1, 2, 3],
    )
    
    # Add a PoC request with higher priority (-100)
    poc_req = PoCRequest(
        request_id="poc-1",
        client_index=0,
        arrival_time=time.time() + 1,  # Arrives later but has higher priority
        priority=POC_REQUEST_PRIORITY,
        poc_params=PoCParams(
            block_hash=BLOCK_HASH,
            public_key=PUBLIC_KEY,
            nonces=[1, 3, 5, 7, 9],  # Real nonces from VALIDATION_SAMPLE
            seq_len=SEQ_LEN,
            k_dim=K_DIM,
        ),
    )
    
    scheduler.add_request(normal_req)
    scheduler.add_request(poc_req)
    
    # Schedule - PoC should be scheduled first despite arriving later
    scheduler_output = scheduler.schedule()
    
    # PoC request should be scheduled (poc_request field populated)
    assert scheduler_output.poc_request is not None, "PoC request was not scheduled"
    assert scheduler_output.poc_request.request_id == "poc-1", "Wrong request scheduled"
    
    # No token scheduling for PoC (no KV cache)
    assert len(scheduler_output.num_scheduled_tokens) == 0, "PoC should not schedule tokens"
    assert scheduler_output.total_num_scheduled_tokens == 0, "PoC should have 0 total tokens"
    
    # Normal request should still be waiting
    assert len(scheduler.waiting) == 1, "Normal request should still be waiting"
    
    print("✓ PASSED")


def test_poc_scheduler_output_structure():
    """Test that PoC scheduler output has correct structure with real parameters."""
    print("Test 3: PoC scheduler output structure...", end=" ")
    
    scheduler = create_scheduler()
    poc_req = PoCRequest(
        request_id="poc-1",
        client_index=0,
        arrival_time=time.time(),
        priority=POC_REQUEST_PRIORITY,
        poc_params=PoCParams(
            block_hash=BLOCK_HASH,
            public_key=PUBLIC_KEY,
            nonces=[1, 3, 5, 7, 9],  # Real nonces from VALIDATION_SAMPLE
            seq_len=SEQ_LEN,
            k_dim=K_DIM,
        ),
    )
    
    scheduler.add_request(poc_req)
    scheduler_output = scheduler.schedule()
    
    # Verify PoC request data
    assert scheduler_output.poc_request is not None, "No PoC request in output"
    poc_data = scheduler_output.poc_request
    
    assert poc_data.request_id == "poc-1", f"Wrong request_id: {poc_data.request_id}"
    assert poc_data.block_hash == BLOCK_HASH, f"Wrong block_hash: {poc_data.block_hash}"
    assert poc_data.public_key == PUBLIC_KEY, f"Wrong public_key: {poc_data.public_key}"
    assert poc_data.nonces == [1, 3, 5, 7, 9], f"Wrong nonces: {poc_data.nonces}"
    assert poc_data.seq_len == SEQ_LEN, f"Wrong seq_len: {poc_data.seq_len}"
    assert poc_data.k_dim == K_DIM, f"Wrong k_dim: {poc_data.k_dim}"
    assert poc_data.priority == POC_REQUEST_PRIORITY, f"Wrong priority: {poc_data.priority}"
    
    print("✓ PASSED")


def test_poc_request_no_kv_cache():
    """Test that PoC requests don't allocate or use KV cache."""
    print("Test 4: PoC request doesn't use KV cache...", end=" ")
    
    scheduler = create_scheduler()
    poc_req = PoCRequest(
        request_id="poc-1",
        client_index=0,
        arrival_time=time.time(),
        priority=POC_REQUEST_PRIORITY,
        poc_params=PoCParams(
            block_hash=BLOCK_HASH,
            public_key=PUBLIC_KEY,
            nonces=[11, 13, 15],  # Real nonces from VALIDATION_SAMPLE
            seq_len=SEQ_LEN,
            k_dim=K_DIM,
        ),
    )
    
    scheduler.add_request(poc_req)
    scheduler_output = scheduler.schedule()
    
    # Verify no KV cache blocks allocated
    assert scheduler_output.total_num_scheduled_tokens == 0, "PoC allocated tokens"
    assert len(scheduler_output.num_scheduled_tokens) == 0, "PoC has scheduled tokens"
    
    # Verify no cached requests
    assert len(scheduler_output.scheduled_cached_reqs.req_ids) == 0, "PoC has cached requests"
    
    # Verify no new requests scheduled (in generate sense)
    assert len(scheduler_output.scheduled_new_reqs) == 0, "PoC scheduled new requests"
    
    print("✓ PASSED")


def test_multiple_poc_requests_queued():
    """Test that multiple PoC requests queue correctly and execute in priority order."""
    print("Test 5: Multiple PoC requests prioritized correctly...", end=" ")
    
    scheduler = create_scheduler()
    
    # Add PoC requests with different priorities
    poc_req_low = PoCRequest(
        request_id="poc-low",
        client_index=0,
        arrival_time=time.time(),
        priority=0,  # Lower priority
        poc_params=PoCParams(
            block_hash=BLOCK_HASH,
            public_key=PUBLIC_KEY,
            nonces=[1, 3],  # Real nonces
            seq_len=SEQ_LEN,
            k_dim=K_DIM,
        ),
    )
    
    poc_req_high = PoCRequest(
        request_id="poc-high",
        client_index=0,
        arrival_time=time.time() + 1,
        priority=POC_REQUEST_PRIORITY,  # Higher priority
        poc_params=PoCParams(
            block_hash=BLOCK_HASH,
            public_key=PUBLIC_KEY,
            nonces=[5, 7],  # Real nonces
            seq_len=SEQ_LEN,
            k_dim=K_DIM,
        ),
    )
    
    scheduler.add_request(poc_req_low)
    scheduler.add_request(poc_req_high)
    
    # First schedule should pick high priority
    scheduler_output = scheduler.schedule()
    assert scheduler_output.poc_request is not None, "No PoC scheduled"
    assert scheduler_output.poc_request.request_id == "poc-high", "High priority not scheduled first"
    
    # Simulate completion
    scheduler._poc_running.status = RequestStatus.FINISHED_STOPPED
    scheduler.poc_requests.pop(scheduler._poc_running.request_id, None)
    scheduler._poc_running = None
    
    # Second schedule should pick low priority
    scheduler_output = scheduler.schedule()
    assert scheduler_output.poc_request is not None, "Second PoC not scheduled"
    assert scheduler_output.poc_request.request_id == "poc-low", "Low priority not scheduled second"
    
    print("✓ PASSED")


def test_poc_consecutive_steps_counter():
    """Test that _poc_consecutive_steps counter increments and resets correctly."""
    print("Test 6: PoC consecutive steps counter...", end=" ")
    
    scheduler = create_scheduler()
    
    # Add PoC request
    poc_req = PoCRequest(
        request_id="poc-1",
        client_index=0,
        arrival_time=time.time(),
        priority=POC_REQUEST_PRIORITY,
        poc_params=PoCParams(
            block_hash=BLOCK_HASH,
            public_key=PUBLIC_KEY,
            nonces=[17, 19],  # Real nonces from VALIDATION_SAMPLE
            seq_len=SEQ_LEN,
            k_dim=K_DIM,
        ),
    )
    scheduler.add_request(poc_req)
    
    initial_steps = scheduler._poc_consecutive_steps
    
    # Schedule PoC request
    scheduler_output = scheduler.schedule()
    assert scheduler_output.poc_request is not None, "PoC not scheduled"
    
    # Counter should increment
    assert scheduler._poc_consecutive_steps == initial_steps + 1, \
        f"Counter not incremented: {scheduler._poc_consecutive_steps} vs {initial_steps + 1}"
    
    # Simulate completion
    scheduler._poc_running.status = RequestStatus.FINISHED_STOPPED
    scheduler.poc_requests.pop(scheduler._poc_running.request_id, None)
    scheduler._poc_running = None
    
    # Add normal request
    normal_req = create_normal_request(
        request_id="normal-1",
        prompt_token_ids=[1, 2, 3],
    )
    scheduler.add_request(normal_req)
    
    # Schedule normal request (no PoC waiting)
    scheduler_output = scheduler.schedule()
    
    # Counter should reset after scheduling non-PoC
    assert scheduler._poc_consecutive_steps == 0, \
        f"Counter not reset: {scheduler._poc_consecutive_steps}"
    
    print("✓ PASSED")


def test_poc_determinism():
    """Test that same PoC parameters produce consistent scheduling behavior."""
    print("Test 7: PoC determinism (same params)...", end=" ")
    
    # Create two schedulers with identical configs
    scheduler1 = create_scheduler()
    scheduler2 = create_scheduler()
    
    # Add identical PoC requests
    nonces = [1, 3, 5, 7, 9]
    
    poc_req1 = PoCRequest(
        request_id="poc-determinism-1",
        client_index=0,
        arrival_time=123456.789,  # Fixed timestamp for determinism
        priority=POC_REQUEST_PRIORITY,
        poc_params=PoCParams(
            block_hash=BLOCK_HASH,
            public_key=PUBLIC_KEY,
            nonces=nonces,
            seq_len=SEQ_LEN,
            k_dim=K_DIM,
        ),
    )
    
    poc_req2 = PoCRequest(
        request_id="poc-determinism-2",
        client_index=0,
        arrival_time=123456.789,  # Same timestamp
        priority=POC_REQUEST_PRIORITY,
        poc_params=PoCParams(
            block_hash=BLOCK_HASH,
            public_key=PUBLIC_KEY,
            nonces=nonces,  # Same nonces
            seq_len=SEQ_LEN,
            k_dim=K_DIM,
        ),
    )
    
    scheduler1.add_request(poc_req1)
    scheduler2.add_request(poc_req2)
    
    # Schedule both
    output1 = scheduler1.schedule()
    output2 = scheduler2.schedule()
    
    # Both should schedule PoC requests
    assert output1.poc_request is not None, "First scheduler didn't schedule PoC"
    assert output2.poc_request is not None, "Second scheduler didn't schedule PoC"
    
    # Verify identical parameters in output
    assert output1.poc_request.block_hash == output2.poc_request.block_hash, "Block hash mismatch"
    assert output1.poc_request.public_key == output2.poc_request.public_key, "Public key mismatch"
    assert output1.poc_request.nonces == output2.poc_request.nonces, "Nonces mismatch"
    assert output1.poc_request.seq_len == output2.poc_request.seq_len, "Seq len mismatch"
    assert output1.poc_request.k_dim == output2.poc_request.k_dim, "K-dim mismatch"
    
    print("✓ PASSED")


def test_poc_starvation_guard_with_realistic_params():
    """Test starvation guard with realistic PoC parameters."""
    print("Test 8: Starvation guard with realistic params...", end=" ")
    
    scheduler = create_scheduler()
    
    # Add normal request first
    normal_req = create_normal_request(
        request_id="normal-1",
        prompt_token_ids=[1, 2, 3, 4, 5],
    )
    scheduler.add_request(normal_req)
    
    # Add multiple PoC requests with real parameters
    for i in range(10):
        poc_req = PoCRequest(
            request_id=f"poc-{i}",
            client_index=0,
            arrival_time=time.time() + i * 0.001,
            priority=POC_REQUEST_PRIORITY,
            poc_params=PoCParams(
                block_hash=BLOCK_HASH,
                public_key=PUBLIC_KEY,
                nonces=[1 + i*2, 3 + i*2],  # Different nonces per request
                seq_len=SEQ_LEN,
                k_dim=K_DIM,
            ),
        )
        scheduler.add_request(poc_req)
    
    poc_scheduled_count = 0
    normal_scheduled = False
    
    # Run scheduler multiple times
    for iteration in range(15):
        scheduler_output = scheduler.schedule()
        
        if scheduler_output.poc_request is not None:
            poc_scheduled_count += 1
            # Simulate completion
            scheduler._poc_running.status = RequestStatus.FINISHED_STOPPED
            scheduler.poc_requests.pop(scheduler._poc_running.request_id, None)
            scheduler._poc_running = None
        else:
            # Normal request scheduled
            if len(scheduler_output.scheduled_new_reqs) > 0:
                normal_scheduled = True
                break
    
    # Verify that starvation guard kicked in
    # (_poc_starvation_limit is 8, so after 8 PoC iterations, normal should get scheduled)
    assert poc_scheduled_count <= scheduler._poc_starvation_limit + 1, \
        f"Too many PoC iterations: {poc_scheduled_count} > {scheduler._poc_starvation_limit + 1}"
    assert normal_scheduled or len(scheduler.waiting) == 0, \
        "Normal request not scheduled despite starvation guard"
    
    print("✓ PASSED")


def main():
    """Run all tests."""
    print("=" * 70)
    print("PoC Scheduler Integration Tests")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Seq length: {SEQ_LEN}, K-dim: {K_DIM}")
    print(f"Block hash: {BLOCK_HASH[:16]}...")
    print(f"Public key: {PUBLIC_KEY[:16]}...")
    print("=" * 70)
    print()
    
    try:
        test_poc_request_added_to_separate_queue()
        test_poc_request_scheduled_with_high_priority()
        test_poc_scheduler_output_structure()
        test_poc_request_no_kv_cache()
        test_multiple_poc_requests_queued()
        test_poc_consecutive_steps_counter()
        test_poc_determinism()
        test_poc_starvation_guard_with_realistic_params()
        
        print()
        print("=" * 70)
        print("✓ ALL TESTS PASSED (8/8)")
        print("=" * 70)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
