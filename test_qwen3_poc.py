#!/usr/bin/env python3
"""Simple PoC forward test for Qwen3 models.

Tests execute_poc_forward directly without starting API server.

Usage:
    # Small model (fast testing, default):
    VLLM_USE_V1=1 python3 test_qwen3_poc.py
    
    # Large production model:
    VLLM_USE_V1=1 TEST_LARGE_MODEL=1 python3 test_qwen3_poc.py
    
    # With specific backends:
    VLLM_USE_V1=1 TEST_LARGE_MODEL=1 \\
      VLLM_ATTENTION_BACKEND=FLASHINFER \\
      VLLM_FP8_MOE_BACKEND=AITER \\
      python3 test_qwen3_poc.py
    
    # Docker (small model):
    docker run --rm --gpus all \\
      -v $(pwd):/workspace \\
      -e VLLM_USE_V1=1 \\
      vllm/vllm-openai:latest \\
      python3 /workspace/test_qwen3_poc.py
      
    # Docker (large model with TP=4):
    docker run --rm --gpus all \\
      -v $(pwd):/workspace \\
      -e VLLM_USE_V1=1 \\
      -e TEST_LARGE_MODEL=1 \\
      --shm-size=8g \\
      vllm/vllm-openai:latest \\
      python3 /workspace/test_qwen3_poc.py
      
Environment Variables:
    TEST_LARGE_MODEL: Set to "1" to use large model (default: "0")
    VLLM_ATTENTION_BACKEND: Attention backend to use
        Options: FLASH_ATTN (default), FLASHINFER, TRITON_ATTN, FLEX_ATTENTION
    VLLM_FP8_MOE_BACKEND: FP8 MoE backend to use (for FP8 models)
        Options: TRITON (default), AITER, MARLIN, BATCHED_TRITON
"""
import os
import sys
import time
import torch

# Force v1 engine
os.environ["VLLM_USE_V1"] = "1"

# Set backend defaults if not specified
if "VLLM_ATTENTION_BACKEND" not in os.environ:
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
if "VLLM_FP8_MOE_BACKEND" not in os.environ:
    os.environ["VLLM_FP8_MOE_BACKEND"] = "TRITON"

from vllm import LLM
from vllm.distributed import get_tp_group


# Model configurations
MODEL_CONFIGS = {
    "small": {
        "model": "Qwen/Qwen3-0.6B",
        "tensor_parallel_size": 1,
        "max_num_seqs": 32,
        "max_model_len": None,
        "dtype": "auto",
    },
    "large": {
        "model": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "tensor_parallel_size": 4,
        "max_num_seqs": 32,
        "max_model_len": 240000,
        "dtype": "float16",
    },
}


def test_poc_forward():
    """Test PoC forward pass with production parameters."""
    
    # Select model based on environment variable
    use_large_model = os.environ.get("TEST_LARGE_MODEL", "1") == "1"
    config_key = "large" if use_large_model else "small"
    config = MODEL_CONFIGS[config_key]
    
    # Get backend configuration
    attention_backend = os.environ.get("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    moe_backend = os.environ.get("VLLM_FP8_MOE_BACKEND", "TRITON")
    
    print("=" * 70)
    print(f"PoC Forward Test - {config['model']}")
    print(f"Configuration: {config_key}")
    print(f"Attention Backend: {attention_backend}")
    print(f"FP8 MoE Backend: {moe_backend}")
    print("=" * 70)
    
    # Initialize vLLM
    print("\n[1/4] Initializing vLLM...")
    llm_kwargs = {
        "model": config["model"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "max_num_seqs": config["max_num_seqs"],
        "enforce_eager": False,  # Use CUDA graphs
        "dtype": config["dtype"],
    }
    
    if config["max_model_len"] is not None:
        llm_kwargs["max_model_len"] = config["max_model_len"]
    
    llm = LLM(**llm_kwargs)
    
    # Get engine_core client (works in both multiprocess and single-process mode)
    engine_core = llm.llm_engine.engine_core
    
    # Test parameters (matching production request)
    block_hash = "69B2F6FC38D2BE8181983AF17D7AFAC5B616EF95674F8762A4AB0EAA7F8032A5"
    public_key = "02704a4bc225f08a2ef8c19439109bb73ff0833d9d87c78a8d072b85262ecaf074"
    
    # Use production batch size (32) to see real optimization impact
    nonces = list(range(32))
    seq_len = 128  # Shorter for faster testing
    
    # Get hidden_size from model config
    hidden_size = llm.llm_engine.model_config.get_hidden_size()
    k_dim = 12
    
    print(f"\n[2/4] Test parameters:")
    print(f"  model: {config['model']}")
    print(f"  tensor_parallel_size: {config['tensor_parallel_size']}")
    print(f"  dtype: {config['dtype']}")
    print(f"  block_hash: {block_hash[:16]}...")
    print(f"  public_key: {public_key[:16]}...")
    print(f"  batch_size: {len(nonces)} nonces")
    print(f"  seq_len: {seq_len}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  k_dim: {k_dim}")
    
    # Warmup run
    print(f"\n[3/4] Warmup run...")
    engine_core.collective_rpc(
        "execute_poc_forward",
        timeout=60.0,
        args=(block_hash, public_key, [0], seq_len, hidden_size, k_dim),
    )
    
    # Execute multiple runs for accurate benchmarking
    print(f"\n[4/4] Benchmarking ({len(nonces)} nonces per batch)...")
    times = []
    all_hashes = []
    
    for run in range(5):
        t0 = time.time()
        
        # Call collective_rpc - dispatches to all workers, returns results from all ranks
        results = engine_core.collective_rpc(
            "execute_poc_forward",
            timeout=60.0,
            args=(block_hash, public_key, nonces, seq_len, hidden_size, k_dim),
        )
        
        # Only the last PP rank returns a result (non-None)
        result = next((r for r in results if r is not None), None)
        
        elapsed = time.time() - t0
        times.append(elapsed)
        
        if result and "vectors_b64" in result:
            all_hashes.append(result["vectors_b64"][0])  # First nonce hash
        
        per_nonce_ms = elapsed * 1000 / len(nonces)
        throughput = len(nonces) / elapsed
        print(f"  Run {run+1}: {elapsed*1000:.1f}ms total ({per_nonce_ms:.1f}ms/nonce, {throughput:.1f} nonces/sec)")
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    avg_per_nonce = avg_time * 1000 / len(nonces)
    avg_throughput = len(nonces) / avg_time
    
    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    if result is None:
        print("ERROR: No result returned (are you on PP rank 0?)")
        return False
    
    if "nonces" not in result or "vectors_b64" not in result:
        print(f"ERROR: Invalid result format: {result.keys()}")
        return False
    
    print(f"\nTiming Statistics ({len(nonces)} nonces per batch):")
    print(f"  Average: {avg_time*1000:.1f}ms total ({avg_per_nonce:.2f}ms per nonce)")
    print(f"  Min:     {min_time*1000:.1f}ms")
    print(f"  Max:     {max_time*1000:.1f}ms")
    print(f"  Throughput: {avg_throughput:.1f} nonces/sec")
    
    # Check determinism (all hashes should be identical)
    if len(set(all_hashes)) == 1:
        print(f"\n✓ Deterministic: all runs produced same hash")
        print(f"  First nonce hash: {all_hashes[0]}")
    else:
        print(f"\n✗ NON-DETERMINISTIC: {len(set(all_hashes))} different hashes!")
        for i, h in enumerate(set(all_hashes)):
            print(f"    Hash variant {i+1}: {h}")
        return False
    
    # Expected baseline hashes for validation
    EXPECTED_HASHES = {
        "small": "XDC0t1IwMzXRs8g0WK3uNto3QqhvMeGu",  # Qwen3-0.6B
        "large": "ZbWht/cxNDSFMHo0a7CvtDOztLVpNp4v",  # Qwen3-235B-FP8
    }
    
    expected_hash = EXPECTED_HASHES.get(config_key)
    if expected_hash and all_hashes[0] == expected_hash:
        print(f"  ✓ Hash matches expected baseline for {config_key} model")
    elif expected_hash:
        print(f"  ✗ WARNING: Hash mismatch!")
        print(f"    Expected: {expected_hash}")
        print(f"    Got:      {all_hashes[0]}")
        print(f"    This indicates algorithm change or numerical issue")
    
    # Display sample vectors (first 3)
    print(f"\nSample Vectors (first 3 of {len(result['vectors_b64'])}):")
    success = True
    for i in range(min(3, len(result["vectors_b64"]))):
        nonce = result["nonces"][i]
        vector_b64 = result["vectors_b64"][i]
        print(f"  nonce={nonce:5d}  vector_b64={vector_b64}")
        
        # Check vector length (12 * 2 bytes FP16 = 24 bytes → 32 chars base64)
        if len(vector_b64) != 32:  # For k_dim=12
            print(f"    WARNING: Unexpected length {len(vector_b64)}, expected 32")
            success = False
    
    # Performance comparison
    print(f"\n{'=' * 70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 70}")
    
    baseline_per_nonce = 4.8  # Old performance from profiling (153ms / 32 nonces)
    speedup = baseline_per_nonce / avg_per_nonce
    
    print(f"  Current:  {avg_per_nonce:.2f}ms per nonce ({avg_throughput:.0f} nonces/sec)")
    print(f"  Baseline: {baseline_per_nonce:.2f}ms per nonce (~205 nonces/sec)")
    print(f"  Speedup:  {speedup:.2f}x faster")
    
    if speedup >= 2.5:
        print(f"  Status:   🚀 EXCELLENT - Major optimization achieved!")
    elif speedup >= 1.5:
        print(f"  Status:   ✓ GOOD - Significant improvement")
    elif speedup >= 1.1:
        print(f"  Status:   ✓ OK - Moderate improvement")
    else:
        print(f"  Status:   ⚠ WARNING - Minimal improvement")
    
    print("=" * 70)
    
    return success


if __name__ == "__main__":
    try:
        success = test_poc_forward()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"✗ Test FAILED with exception:")
        print(f"{'=' * 70}")
        import traceback
        traceback.print_exc()
        exit(1)
