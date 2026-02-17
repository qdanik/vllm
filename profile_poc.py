#!/usr/bin/env python3
"""Profile PoC forward pass to find bottlenecks."""
import os
import time
os.environ["VLLM_USE_V1"] = "1"
os.environ["POC_SKIP_COMPILED"] = "1"  # Slow mode
os.environ["POC_PROFILE"] = "1"  # Slow mode

from vllm import LLM

def profile_poc():
    print("Initializing vLLM...")
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        max_num_seqs=32,
        enforce_eager=False,
    )
    
    engine_core = llm.llm_engine.engine_core
    hidden_size = llm.llm_engine.model_config.get_hidden_size()
    
    block_hash = "69B2F6FC38D2BE8181983AF17D7AFAC5B616EF95674F8762A4AB0EAA7F8032A5"
    public_key = "02704a4bc225f08a2ef8c19439109bb73ff0833d9d87c78a8d072b85262ecaf074"
    nonces = list(range(32))  # Full batch
    seq_len = 128
    k_dim = 12
    
    print(f"\nProfiling with {len(nonces)} nonces...")
    print("=" * 70)
    
    # Warmup
    print("Warmup run...")
    engine_core.collective_rpc(
        "execute_poc_forward",
        timeout=60.0,
        args=(block_hash, public_key, [0], seq_len, hidden_size, k_dim),
    )
    
    # Profile multiple runs
    times = []
    for run in range(5):
        t0 = time.time()
        results = engine_core.collective_rpc(
            "execute_poc_forward",
            timeout=60.0,
            args=(block_hash, public_key, nonces, seq_len, hidden_size, k_dim),
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        
        result = next((r for r in results if r is not None), None)
        print(f"Run {run+1}: {elapsed*1000:.1f}ms ({len(nonces)/elapsed:.1f} nonces/sec)")
    
    avg_time = sum(times) / len(times)
    print("=" * 70)
    print(f"Average: {avg_time*1000:.1f}ms ({len(nonces)/avg_time:.1f} nonces/sec)")
    print(f"Per-nonce: {avg_time*1000/len(nonces):.1f}ms")
    
    # Show estimated throughput
    print(f"\nEstimated throughput for 1000 nonces: {1000/len(nonces)*avg_time:.1f}s")
    print(f"                     for 10000 nonces: {10000/len(nonces)*avg_time:.1f}s")

if __name__ == "__main__":
    profile_poc()
