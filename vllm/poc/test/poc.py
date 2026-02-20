#!/usr/bin/env python3
"""Profile PoC through OpenAI API servers.

This script directly simulates what happens when /init/generate is called:
    1. Initialize LLM (like server startup)
    2. Calculate optimal batch_size (like calculate_optimal_batch_size in routes.py)
    3. Run multiple RPC calls (like _generation_loop)

Environment Variables:
    POC_PROFILE_RUNS: Number of batch runs to profile (default: 10)

NOTE: DeepGEMM warmup takes 15-30 minutes on first run to compile all kernel variants.
      This is NORMAL and required for optimal performance. Subsequent runs will be fast.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import vllm
import torch

from vllm import LLM
from vllm.config import CompilationConfig, PassConfig
from vllm.poc.runtime.validation_utils import validate_artifacts
from vllm.poc.utils.env import (
    POC_AUTO_BATCH_SIZE_DEFAULT,
    POC_BATCH_SIZE_DEFAULT,
    POC_PROFILE_DIST_THRESHOLD,
    POC_PROFILE_FRAUD_THRESHOLD,
    POC_PROFILE_P_MISMATCH,
    POC_PROFILE_RUNS,
    POC_PROFILE_VALIDATION_JSON,
)

stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(line_buffering=True, write_through=True)
stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
if callable(stderr_reconfigure):
    stderr_reconfigure(line_buffering=True, write_through=True)

PUBLIC_KEY = "02e0f3b6b7f832ead7af2a235b9b27715a4d586b0fa108e735f0676a5086479225"
BLOCK_HASH = "8d148df1530d06a3412acd3deda4db16bae780eefdd160e081e6f878417de92a"
VALIDATION_SAMPLE = {
    "public_key": PUBLIC_KEY,
    "block_hash": BLOCK_HASH,
    "block_height": 2732723,
    "node_id": 1,
    "artifacts": [
        {"nonce": 1, "vector_b64": "ta47Lzc2LrhUrvomZ7BqOIs1Qy7fs2Mx"},
        {"nonce": 3, "vector_b64": "FrfIMNsx2jXDNHap7rL2rZipbTWjOAWw"},
        {"nonce": 5, "vector_b64": "Tq2CrB8sVSz0sGO0PTeCNzq3VLfXpX6y"},
        {"nonce": 7, "vector_b64": "4jHoLYc1ATTlNQMwAivsuASr1TDhNKO1"},
        {"nonce": 9, "vector_b64": "4K7PtGmwBTgMON4197UhoFenkS5hs8+x"},
        {"nonce": 11, "vector_b64": "ELgkrW6vZrR7Mt00PbUJtoMqcDVKNRiy"},
        {"nonce": 13, "vector_b64": "CzIRp/i0uSg3JL0zVCw2L+a0MbdYufsy"},
        {"nonce": 15, "vector_b64": "uTQMta41FrjDKnKvrTS9LyWvEzg4Lm8x"},
        {"nonce": 17, "vector_b64": "B7DPrw82dDRbNJW5+i2JINiumyh8NQez"},
        {"nonce": 19, "vector_b64": "ZzG0MS+zUC+6tAU4jbNGN4Y3uSzKLiMy"},
    ],
    "encoding": {"dtype": "f16", "k_dim": 12, "endian": "le"},
}

SERVER_STARTUP_TIMEOUT_SEC = int(os.environ.get("POC_PROFILE_SERVER_STARTUP_TIMEOUT_SEC", "900"))
SERVER_STARTUP_PROGRESS_SEC = int(os.environ.get("POC_PROFILE_SERVER_PROGRESS_SEC", "5"))
BASE_PORT = 8766
PROJECT_ROOT = Path(__file__).resolve().parent

_SERVER_LOG_TAILS: Dict[int, deque[str]] = {}
_BASE_URL_TO_SERVER_IDX: Dict[str, int] = {}


def _stream_server_logs(
    server_idx: int,
    stream: Any,
    log_file: Any,
) -> None:
    for line in iter(stream.readline, ""):
        text = line.rstrip("\n")
        if not text:
            continue
        print(f"[server-{server_idx + 1}] {text}")
        try:
            log_file.write(line)
        except Exception:
            pass
        _SERVER_LOG_TAILS.setdefault(server_idx, deque(maxlen=120)).append(text)
    try:
        stream.close()
    except Exception:
        pass


def _load_validation_payload() -> Optional[Dict[str, Any]]:
    validation_path = POC_PROFILE_VALIDATION_JSON
    if validation_path:
        with open(validation_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return VALIDATION_SAMPLE


def _build_validation_map(payload: Dict[str, Any]) -> Dict[int, str]:
    artifacts = payload.get("artifacts") or []
    return {int(a["nonce"]): a["vector_b64"] for a in artifacts}

    return validation_map


def calculate_optimal_batch_size_local(
    llm, seq_len: int, safety_factor: float = 0.7
) -> int:
    """Local version of calculate_optimal_batch_size from routes.py."""
    try:
        vllm_config = llm.llm_engine.vllm_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        hidden_size = model_config.get_hidden_size()
        num_layers = model_config.get_num_layers(parallel_config)

        bytes_per_token = 2  # fp16
        mem_per_sample = (
            seq_len * hidden_size * bytes_per_token
            + 2 * num_layers * seq_len * hidden_size * bytes_per_token
            + seq_len * hidden_size * bytes_per_token
        )

        import torch

        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        reserved_memory = torch.cuda.memory_reserved(0)

        truly_free = (total_memory - reserved_memory) + (
            reserved_memory - allocated_memory
        )
        free_memory = truly_free * safety_factor

        free_memory = max(free_memory, 512 * 1024**2)

        batch_size = int(free_memory / mem_per_sample)
        batch_size = max(32, min(batch_size, 512))

        print(f"  Calculated batch_size: {batch_size}")
        print(f"    seq_len={seq_len}, hidden_size={hidden_size}, num_layers={num_layers}")
        print(f"    total_memory={total_memory/1024**3:.1f}GB")
        print(
            f"    reserved_memory={reserved_memory/1024**3:.1f}GB (includes KV cache)"
        )
        print(f"    allocated_memory={allocated_memory/1024**3:.1f}GB")
        print(f"    free_memory={free_memory/1024**3:.1f}GB (usable)")
        print(f"    mem_per_sample={mem_per_sample/1024**2:.1f}MB")

        return batch_size
    except Exception as e:
        print(f"  Warning: Could not calculate optimal batch_size: {e}")
        import traceback

        traceback.print_exc()
        return 32


def profile_poc():
    profile_runs = POC_PROFILE_RUNS
    dist_threshold = POC_PROFILE_DIST_THRESHOLD
    p_mismatch = POC_PROFILE_P_MISMATCH
    fraud_threshold = POC_PROFILE_FRAUD_THRESHOLD

    model = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
    seq_len = 1024
    k_dim = 12

    public_key = PUBLIC_KEY
    block_hash = BLOCK_HASH

    print("=" * 70)
    print("Simulating: /init/generate code path")
    print(f"Model: {model}")
    print(f"Profile runs: {profile_runs}")
    print(f"TP size: {tp_size}")
    print(f"API servers: {api_server_count}")
    print(f"Visible CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"Device slices: {device_slices}")
    print(f"max_model_len: {max_model_len}")
    print("model_args:")
    print("  --max-model-len 240000")
    print("  --enable-auto-tool-choice")
    print("  --tool-call-parser hermes")
    print("=" * 70)
    print(
        f"  dist_threshold={dist_threshold}, p_mismatch={p_mismatch}, fraud_threshold={fraud_threshold}"
    )

    pass_config = PassConfig(
        eliminate_noops=True,
        fuse_norm_quant=True,
        fuse_act_quant=True,
        fuse_attn_quant=False,
        enable_sp=False,
        fuse_gemm_comms=False,
        fuse_allreduce_rms=False,
    )

    print("\nInitializing vLLM...")
    llm = LLM(
        model=model,
        tensor_parallel_size=4,
        max_num_seqs=32,
        max_model_len=240000,
        max_num_batched_tokens=8192,
        enforce_eager=False,
        dtype="float16",
        disable_custom_all_reduce=False,
        kv_cache_dtype="auto",
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        compilation_config=CompilationConfig(pass_config=pass_config),
    )

    engine_core = llm.llm_engine.engine_core
    hidden_size = llm.llm_engine.model_config.get_hidden_size()

    print("\nCalculating optimal batch_size...")
    batch_size = POC_BATCH_SIZE_DEFAULT
    if POC_AUTO_BATCH_SIZE_DEFAULT:
        batch_size = calculate_optimal_batch_size_local(llm, seq_len)
    else:
        print(
            "Using default batch_size: "
            f"{batch_size} (set POC_AUTO_BATCH_SIZE_DEFAULT=1 to auto-calculate)"
        )

    print(f"\nRunning {profile_runs} batches with batch_size={batch_size}...")

    nonces = list(range(batch_size))
    print("\nWarmup run...")
    engine_core.collective_rpc(
        "execute_poc_forward",
        timeout=60.0,
        args=(block_hash, public_key, nonces, seq_len, hidden_size, k_dim),
    )

    times = []
    all_hashes = []
    first_hash = None
    total_nonces = 0

    print("\nProfiling...")
    validation_payload = _load_validation_payload()
    validation_map = _build_validation_map(validation_payload) if validation_payload else {}
    validated_once = False
    for run in range(profile_runs):
        batch_nonces = list(range(run * batch_size, (run + 1) * batch_size))

        t0 = time.time()
        results = engine_core.collective_rpc(
            "execute_poc_forward",
            timeout=60.0,
            args=(block_hash, public_key, batch_nonces, seq_len, hidden_size, k_dim),
        )
        elapsed = time.time() - t0
        times.append(elapsed)

        result = next((r for r in results if r is not None), None)
        if result and "vectors_b64" in result:
            hash_value = result["vectors_b64"][0]
            all_hashes.append(hash_value)
            if first_hash is None:
                first_hash = hash_value

            len_nonces = len(result["vectors_b64"])
            total_nonces += len_nonces

            nonces_per_sec = len_nonces / elapsed if elapsed > 0 else 0
            ms_per_nonce = elapsed * 1000 / len_nonces if len_nonces > 0 else 0
            print(
                f"  Run {run+1:2d}: {elapsed*1000:.1f}ms, {len_nonces} nonces "
                f"({nonces_per_sec:.1f}/sec, {ms_per_nonce:.2f}ms/nonce)"
            )

            if validation_map and not validated_once:
                computed_artifacts = [
                    {"nonce": nonce, "vector_b64": vector_b64}
                    for nonce, vector_b64 in zip(batch_nonces, result["vectors_b64"])
                ]

                print("\n[DEBUG] Validation Info:")
                print(
                    "  validation_map has "
                    f"{len(validation_map)} entries: {sorted(validation_map.keys())}"
                )
                print(
                    f"  batch_nonces: {batch_nonces[:5]}... (first 5 of {len(batch_nonces)})"
                )
                print(f"  computed_artifacts has {len(computed_artifacts)} entries")

                matching_nonces = [n for n in batch_nonces if n in validation_map]
                print(
                    f"  matching nonces: {matching_nonces} ({len(matching_nonces)} found)"
                )

                if matching_nonces:
                    for nonce in matching_nonces:
                        expected = validation_map[nonce]
                        computed = next(
                            a["vector_b64"]
                            for a in computed_artifacts
                            if a["nonce"] == nonce
                        )
                        match = "OK" if expected == computed else "MISMATCH"
                        print(f"    nonce {nonce}: {match}")
                        if expected != computed:
                            print(f"      expected: {expected}")
                            print(f"      got:      {computed}")

                try:
                    validation_result = validate_artifacts(
                        computed_artifacts,
                        validation_map,
                        dist_threshold=dist_threshold,
                        p_mismatch=p_mismatch,
                        fraud_threshold=fraud_threshold,
                    )
                except Exception as exc:
                    print(f"\n[WARN] Validation failed ({type(exc).__name__}): {exc}")
                    validation_result = {
                        "n_total": len(computed_artifacts),
                        "n_mismatch": len(computed_artifacts),
                        "p_value": 0.0,
                        "fraud_detected": True,
                        "mismatch_nonces": [],
                    }
                validated_once = True
                print("\nValidation result:")
                print(
                    f"  n_total={validation_result['n_total']}, "
                    f"n_mismatch={validation_result['n_mismatch']}, "
                    f"p_value={validation_result['p_value']:.6f}, "
                    f"fraud_detected={validation_result['fraud_detected']}"
                )
        else:
            print(f"  Run {run+1:2d}: FAILED - no result")

    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)

    if times:
        avg_time = sum(times) / len(times)
        avg_nonces = total_nonces / len(times)
        avg_rate = avg_nonces / avg_time if avg_time > 0 else 0
        time_per_nonce = avg_time / avg_nonces if avg_nonces > 0 else 0

        print(f"Batch size used: {batch_size}")
        print(f"Total batches: {len(times)}")
        print(f"Total nonces: {total_nonces}")
        print(f"Average batch time: {avg_time*1000:.1f}ms")
        print(f"Average rate: {avg_rate:.2f} nonces/sec")
        print(f"Average rate: {avg_rate * 60:.0f} nonces/min")
        print(f"Time per nonce: {time_per_nonce*1000:.2f}ms")

        if len(times) > 1:
            min_time = min(times) * 1000
            max_time = max(times) * 1000
            variance = ((max_time - min_time) / (avg_time * 1000) * 100) if avg_time > 0 else 0
            print(
                f"\nBatch time variance: {min_time:.1f} - {max_time:.1f}ms "
                f"(±{variance:.1f}%)"
            )

        for i, h in enumerate(set(all_hashes)):
            print(f"  Variant {i+1}: {h}")

        target_ms_per_nonce = 57
        current_ms = time_per_nonce * 1000
        gap = (current_ms - target_ms_per_nonce) / target_ms_per_nonce * 100

        print(f"\n{'='*70}")
        print("Performance vs target:")
        print(f"  Current: {current_ms:.2f}ms/nonce")
        print(f"  Target:  {target_ms_per_nonce:.2f}ms/nonce")
        if gap > 0:
            print(f"  Gap: {gap:.1f}% slower (need {gap:.0f}% improvement)")
        else:
            print(f"  OK Exceeded target by {-gap:.1f}%!")

        print(
            "\nEstimated time for 1000 nonces: "
            f"{1000/avg_rate:.1f}s ({1000/avg_rate/60:.1f}min)"
        )
        print(
            "Estimated time for 10000 nonces: "
            f"{10000/avg_rate:.1f}s ({10000/avg_rate/60:.1f}min)"
        )
    else:
        print("No timing data collected!")


if __name__ == "__main__":
    profile_poc()
