#!/usr/bin/env python3
"""Profile PoC by simulating /init/generate code path locally.

This script directly simulates what happens when /init/generate is called:
    1. Initialize LLM (like server startup)
    2. Run multiple RPC calls (like _generation_loop)

NOTE: DeepGEMM warmup takes 6-10 minutes on first run to compile all kernel variants.
      This is NORMAL and required for optimal performance. Subsequent runs will be fast.
"""

# ruff: noqa: E501

import os
import time
from typing import Any

os.environ["VLLM_USE_V1"] = "1"

from vllm import LLM
from vllm.config import CompilationConfig, PassConfig
from vllm.poc.runtime.validation_utils import validate_artifacts
from vllm.poc.utils.env import (
    POC_BATCH_SIZE_DEFAULT,
    POC_PROFILE_DIST_THRESHOLD,
    POC_PROFILE_FRAUD_THRESHOLD,
    POC_PROFILE_P_MISMATCH,
)
from vllm.poc.constants import POC_REQUEST_PRIORITY
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestKind, PoCParams

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


def _build_validation_map(payload: dict[str, Any]) -> dict[int, str]:
    artifacts = payload.get("artifacts") or []
    validation_map = {int(a["nonce"]): a["vector_b64"] for a in artifacts}

    return validation_map


def _run_poc_once_via_scheduler(
    engine_core,
    *,
    request_id: str,
    block_hash: str,
    public_key: str,
    block_height: int,
    nonce: int,
    seq_len: int,
    k_dim: int,
    timeout_s: float = 60.0,
    priority: int = POC_REQUEST_PRIORITY,
) -> dict[str, Any]:
    """Submit one PoC nonce request into the v1 scheduler and block for its result."""

    req = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=None,
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        client_index=0,
        priority=priority,
        kind=EngineCoreRequestKind.POC,
        poc_params=PoCParams(
            block_hash=block_hash,
            public_key=public_key,
            block_height=block_height,
            nonce=nonce,
            seq_len=seq_len,
            k_dim=k_dim,
        ),
    )

    engine_core.add_request(req)

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        outputs = engine_core.get_output()
        for out in outputs.outputs:
            if out.request_id != request_id:
                continue
            if out.poc_result is not None:
                return out.poc_result
            raise RuntimeError("PoC request finished without result")

    raise TimeoutError(f"Timeout waiting for PoC result (request_id={request_id})")


def profile_poc():
    profile_runs = 10  # Number of profiling iterations
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

    # Use configured batch size
    batch_size = POC_BATCH_SIZE_DEFAULT
    print(f"Using batch_size: {batch_size}")

    print(f"\nRunning {profile_runs} batches with batch_size={batch_size}...")

    block_height = VALIDATION_SAMPLE["block_height"]
    nonce_base = 0
    print("\nWarmup run...")
    _run_poc_once_via_scheduler(
        engine_core,
        request_id="warmup",
        block_hash=block_hash,
        public_key=public_key,
        block_height=block_height,
        nonce=nonce_base,
        seq_len=seq_len,
        k_dim=k_dim,
        timeout_s=60.0,
    )

    times = []
    all_hashes = []
    first_hash = None
    total_nonces = 0

    print("\nProfiling...")
    validation_map = _build_validation_map(VALIDATION_SAMPLE)
    validated_once = False
    for run in range(profile_runs):
        nonce = nonce_base + run

        t0 = time.time()
        result = _run_poc_once_via_scheduler(
            engine_core,
            request_id=f"profile-{run}",
            block_hash=block_hash,
            public_key=public_key,
            block_height=block_height,
            nonce=nonce,
            seq_len=seq_len,
            k_dim=k_dim,
            timeout_s=60.0,
        )
        elapsed = time.time() - t0
        times.append(elapsed)

        if result and "vector_b64" in result:
            hash_value = result["vector_b64"]
            all_hashes.append(hash_value)
            if first_hash is None:
                first_hash = hash_value

            nonces_per_sec = 1.0 / elapsed if elapsed > 0 else 0
            ms_per_nonce = elapsed * 1000
            print(
                f"  Run {run + 1:2d}: {elapsed * 1000:.1f}ms, nonce {nonce} "
                f"({nonces_per_sec:.1f}/sec, {ms_per_nonce:.2f}ms/nonce)"
            )

            if validation_map and not validated_once:
                computed_nonce = nonce
                computed_vector = result["vector_b64"]

                print("\n[DEBUG] Validation Info:")
                print(
                    "  validation_map has "
                    f"{len(validation_map)} entries: {sorted(validation_map.keys())}"
                )
                print(f"  computed_nonce: {computed_nonce}")
                print(f"  computed_vector: {computed_vector}")

                if computed_nonce in validation_map:
                    expected = validation_map[computed_nonce]
                    match = "OK" if expected == computed_vector else "MISMATCH"
                    print(f"  nonce {computed_nonce}: {match}")
                    if expected != computed_vector:
                        print(f"    expected: {expected}")
                        print(f"    got:      {computed_vector}")

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
                if validation_result["mismatch_nonces"]:
                    print(f"  mismatch_nonces={validation_result['mismatch_nonces']}")
                print()
        else:
            print(f"  Run {run + 1:2d}: FAILED - no result")

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
        print(f"Average batch time: {avg_time * 1000:.1f}ms")
        print(f"Average rate: {avg_rate:.2f} nonces/sec")
        print(f"Average rate: {avg_rate * 60:.0f} nonces/min")
        print(f"Time per nonce: {time_per_nonce * 1000:.2f}ms")

        if len(times) > 1:
            min_time = min(times) * 1000
            max_time = max(times) * 1000
            variance = ((max_time - min_time) / (avg_time * 1000) * 100) if avg_time > 0 else 0
            print(f"\nBatch time variance: {min_time:.1f} - {max_time:.1f}ms (±{variance:.1f}%)")

        for i, h in enumerate(set(all_hashes)):
            print(f"  Variant {i + 1}: {h}")

        target_ms_per_nonce = 57
        current_ms = time_per_nonce * 1000
        gap = (current_ms - target_ms_per_nonce) / target_ms_per_nonce * 100

        print(f"\n{'=' * 70}")
        print("Performance vs target:")
        print(f"  Current: {current_ms:.2f}ms/nonce")
        print(f"  Target:  {target_ms_per_nonce:.2f}ms/nonce")
        if gap > 0:
            print(f"  Gap: {gap:.1f}% slower (need {gap:.0f}% improvement)")
        else:
            print(f"  OK Exceeded target by {-gap:.1f}%!")

        print(
            "\nEstimated time for 1000 nonces: "
            f"{1000 / avg_rate:.1f}s ({1000 / avg_rate / 60:.1f}min)"
        )
        print(
            "Estimated time for 10000 nonces: "
            f"{10000 / avg_rate:.1f}s ({10000 / avg_rate / 60:.1f}min)"
        )
    else:
        print("No timing data collected!")


if __name__ == "__main__":
    profile_poc()
