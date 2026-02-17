#!/usr/bin/env python3
"""Test different backend combinations for optimal performance.

Tests various attention and MoE backends to find the best speed/quality tradeoff.

Usage:
    # Test all combinations:
    VLLM_USE_V1=1 TEST_LARGE_MODEL=1 python3 test_backend_matrix.py
    
    # Test specific backend:
    VLLM_USE_V1=1 TEST_LARGE_MODEL=1 ATTN_BACKEND=FLASHINFER python3 test_backend_matrix.py
"""
import os
import sys
import time
import subprocess
from typing import Dict, List, Tuple

# Backend configurations to test
ATTENTION_BACKENDS = [
    "FLASH_ATTN",      # Current default
    "FLASHINFER",      # Alternative, may be faster
    # "TRITON_ATTN",   # Slower, skip for now
    # "FLEX_ATTENTION", # Experimental
]

MOE_BACKENDS = [
    "TRITON",          # Current default
    "AITER",           # May be faster
    "MARLIN",          # FP8 optimized
    # "BATCHED_TRITON", # Testing
]

# Expected baseline hash for large model (nonce=0)
EXPECTED_HASH = "ZbWht/cxNDSFMHo0a7CvtDOztLVpNp4v"

# Number of test runs per configuration
NUM_RUNS = 3


def run_test(attn_backend: str, moe_backend: str) -> Dict:
    """Run PoC test with specific backend configuration.
    
    Returns:
        dict with keys: success, throughput, latency, hash, error
    """
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "1"
    env["TEST_LARGE_MODEL"] = "1"
    
    # Set backend preferences
    env["VLLM_ATTENTION_BACKEND"] = attn_backend
    env["VLLM_FP8_MOE_BACKEND"] = moe_backend
    
    print(f"\n{'='*70}")
    print(f"Testing: ATTN={attn_backend}, MoE={moe_backend}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            ["python3", "test_qwen3_poc.py"],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
        )
        
        output = result.stdout + result.stderr
        
        # Parse output
        throughput = None
        latency = None
        hash_val = None
        
        for line in output.split('\n'):
            if "Throughput:" in line and "nonces/sec" in line:
                try:
                    throughput = float(line.split("Throughput:")[1].split("nonces/sec")[0].strip())
                except:
                    pass
            
            if "Average:" in line and "ms per nonce" in line:
                try:
                    # Extract "X.XXms per nonce"
                    parts = line.split("(")[1].split("ms per nonce")[0].strip()
                    latency = float(parts)
                except:
                    pass
            
            if "First nonce hash:" in line:
                try:
                    hash_val = line.split("First nonce hash:")[1].strip()
                except:
                    pass
        
        # Check for errors
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Exit code {result.returncode}",
                "output": output[-500:]  # Last 500 chars
            }
        
        # Verify hash
        hash_match = hash_val == EXPECTED_HASH if hash_val else False
        
        return {
            "success": True,
            "throughput": throughput,
            "latency": latency,
            "hash": hash_val,
            "hash_match": hash_match,
            "output": None,
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Timeout (>5min)",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def main():
    """Run backend matrix test."""
    
    print("=" * 70)
    print("Backend Matrix Test for PoC Performance")
    print("=" * 70)
    print(f"\nTesting {len(ATTENTION_BACKENDS)} attention backends × {len(MOE_BACKENDS)} MoE backends")
    print(f"Runs per config: {NUM_RUNS}")
    print(f"Expected hash: {EXPECTED_HASH}")
    print()
    
    results: List[Tuple[str, str, Dict]] = []
    
    # Test each combination
    for attn_backend in ATTENTION_BACKENDS:
        for moe_backend in MOE_BACKENDS:
            runs = []
            
            for run in range(NUM_RUNS):
                print(f"\n  Run {run+1}/{NUM_RUNS}...")
                result = run_test(attn_backend, moe_backend)
                runs.append(result)
                
                if not result["success"]:
                    print(f"  ✗ FAILED: {result.get('error', 'Unknown error')}")
                    break
                else:
                    lat = result.get("latency", "?")
                    tp = result.get("throughput", "?")
                    match = "✓" if result.get("hash_match") else "✗"
                    print(f"  {match} {lat}ms/nonce, {tp} nonces/sec")
            
            # Average successful runs
            successful = [r for r in runs if r["success"]]
            
            if successful:
                avg_throughput = sum(r["throughput"] for r in successful if r["throughput"]) / len(successful)
                avg_latency = sum(r["latency"] for r in successful if r["latency"]) / len(successful)
                hash_match = all(r.get("hash_match", False) for r in successful)
                
                results.append((attn_backend, moe_backend, {
                    "success": True,
                    "avg_throughput": avg_throughput,
                    "avg_latency": avg_latency,
                    "hash_match": hash_match,
                    "runs": len(successful),
                }))
            else:
                results.append((attn_backend, moe_backend, {
                    "success": False,
                    "error": runs[0].get("error", "Unknown"),
                }))
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Attention':<15} {'MoE':<15} {'Latency':<12} {'Throughput':<15} {'Hash':<8} {'Status'}")
    print("-" * 70)
    
    best_throughput = 0
    best_config = None
    
    for attn, moe, result in results:
        if result["success"]:
            lat = f"{result['avg_latency']:.2f}ms"
            tp = f"{result['avg_throughput']:.1f}/s"
            hash_status = "✓ MATCH" if result["hash_match"] else "✗ WRONG"
            status = "OK" if result["hash_match"] else "SKIP"
            
            print(f"{attn:<15} {moe:<15} {lat:<12} {tp:<15} {hash_status:<8} {status}")
            
            if result["hash_match"] and result["avg_throughput"] > best_throughput:
                best_throughput = result["avg_throughput"]
                best_config = (attn, moe, result)
        else:
            error = result.get("error", "Failed")[:20]
            print(f"{attn:<15} {moe:<15} {'ERROR':<12} {'-':<15} {'-':<8} {error}")
    
    print("-" * 70)
    
    if best_config:
        attn, moe, result = best_config
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Attention: {attn}")
        print(f"   MoE:       {moe}")
        print(f"   Latency:   {result['avg_latency']:.2f}ms per nonce")
        print(f"   Throughput: {result['avg_throughput']:.1f} nonces/sec")
        print(f"   Hash:      ✓ Correct")
        print()
        print(f"To use this configuration:")
        print(f"  export VLLM_ATTENTION_BACKEND={attn}")
        print(f"  export VLLM_FP8_MOE_BACKEND={moe}")
    else:
        print("\n⚠ No valid configuration found with correct hashes!")
    
    print("=" * 70)


if __name__ == "__main__":
    if not os.path.exists("test_qwen3_poc.py"):
        print("ERROR: test_qwen3_poc.py not found in current directory")
        sys.exit(1)
    
    main()
