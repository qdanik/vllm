#!/usr/bin/env python3
"""Benchmark PoC execution with and without concurrent inference requests.

This script demonstrates how PoC and inference requests execute together in
the V1 scheduler with priority-based scheduling - using 4 scenarios:
    1. Baseline: PoC only
    2. PoC + inference (max_tokens=2000)
    3. PoC + inference (max_tokens=3000)
    4. PoC + inference (max_tokens=4000)

Compares execution speed and analyzes performance overhead.

NOTE: DeepGEMM warmup takes 6-10 minutes on first run to compile kernels.
      This is NORMAL and required for performance. Subsequent runs are fast.
"""

import os
import time
from typing import Optional

os.environ["VLLM_USE_V1"] = "1"

from vllm import LLM, SamplingParams
from vllm.poc.constants import POC_REQUEST_PRIORITY
from vllm.poc.env import POC_BATCH_SIZE_DEFAULT
from vllm.poc.v1.scheduler_params import PoCSchedulerParams
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestKind

PUBLIC_KEY = "02e0f3b6b7f832ead7af2a235b9b27715a4d586b0fa108e735f0676a5086479225"
BLOCK_HASH = "8d148df1530d06a3412acd3deda4db16bae780eefdd160e081e6f878417de92a"


def run_scenario(
    engine_core,
    scenario_name: str,
    num_poc_nonces: int,
    num_inference_requests: int,
    poc_seq_len: int = 16,  # Reduced from 32
    poc_k_dim: int = 8,  # Reduced from 12
    nonce_start: int = 1,
    inference_max_tokens: int = 32,
    timeout_seconds: float = 900.0,
) -> dict:
    """Run a single scenario and return results."""
    scenario_tag = (
        scenario_name.split(":", 1)[0].replace("Scenario", "").strip().lower()
        or "scenario"
    )

    print("=" * 80)
    print(f"{scenario_name}")
    print("=" * 80)
    print(
        "PoC nonces: "
        f"{num_poc_nonces}, Inference requests: {num_inference_requests}, "
        f"Nonce start: {nonce_start}, "
        f"Inference max_tokens: {inference_max_tokens if num_inference_requests else 'N/A'}"
    )
    print()
    print(
        f"Submitting {num_poc_nonces} PoC nonces in parallel with "
        f"{num_inference_requests} inference requests..."
    )
    print()

    # Track request timings
    results = {
        "poc_requests": [],
        "inference_requests": [],
    }

    # Map request IDs to their nonces
    poc_nonce_map = {}
    scenario_start_time = time.time()

    # Submit PoC and inference requests in parallel (interleaved)
    poc_request_ids = []
    inference_request_ids = []
    scenario_errors = []

    # Submit all PoC requests
    for i in range(num_poc_nonces):
        request_id = f"{scenario_tag}-poc-{i}"
        poc_request_ids.append(request_id)
        nonce = nonce_start + i * 2
        poc_nonce_map[request_id] = nonce

        req = EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=[1],  # Minimal dummy token for PoC
            mm_features=None,
            sampling_params=SamplingParams(),  # Required for Request validation
            pooling_params=None,
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            client_index=0,
            priority=POC_REQUEST_PRIORITY,
            kind=EngineCoreRequestKind.POC,
            poc_params=PoCSchedulerParams(
                block_hash=BLOCK_HASH,
                public_key=PUBLIC_KEY,
                block_height=2732723,
                nonce=nonce,
                seq_len=poc_seq_len,
                k_dim=poc_k_dim,
            ),
        )

        submit_time = time.time()
        try:
            engine_core.add_request(req)
            results["poc_requests"].append(
                {
                    "id": request_id,
                    "submit_time": submit_time,
                    "nonce": nonce,
                }
            )
            print(f"  [{submit_time:.2f}] Submitted PoC {request_id} (nonce={nonce})")
        except Exception as e:
            scenario_errors.append(f"Failed to submit {request_id}: {e}")
            print(f"  ✗ Failed to submit PoC {request_id}: {e}")

    # Submit inference requests immediately after PoC (parallel execution)
    for i in range(num_inference_requests):
        request_id = f"{scenario_tag}-infer-{i}"
        inference_request_ids.append(request_id)

        req = EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=[1, 2, 3, 4, 5],  # Dummy prompt
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=inference_max_tokens),
            pooling_params=None,
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            client_index=0,
            priority=0,  # Chat priority (normally 0)
            kind=EngineCoreRequestKind.GENERATE,
        )

        submit_time = time.time()
        try:
            engine_core.add_request(req)
            results["inference_requests"].append(
                {
                    "id": request_id,
                    "submit_time": submit_time,
                }
            )
            print(
                f"  [{submit_time:.2f}] Submitted Inference {request_id} "
                f"(priority=0, max_tokens={inference_max_tokens})"
            )
        except Exception as e:
            scenario_errors.append(f"Failed to submit {request_id}: {e}")
            print(f"  ✗ Failed to submit inference {request_id}: {e}")

    print()
    print("Collecting results...")
    print()

    # Collect results - track which complete first
    completion_order = []
    poc_completion_times: dict[str, Optional[float]] = {
        rid: None for rid in poc_request_ids
    }
    inference_completion_times: dict[str, Optional[float]] = {
        rid: None for rid in inference_request_ids
    }

    deadline = time.time() + timeout_seconds
    expected_total = len(poc_request_ids) + len(inference_request_ids)
    while time.time() < deadline:
        if len(completion_order) == expected_total:
            break

        try:
            outputs = engine_core.get_output()
        except Exception as e:
            scenario_errors.append(f"Engine error in {scenario_name}: {e}")
            print(f"  ✗ Engine error: {e}")
            break

        if outputs is None or outputs.outputs is None:
            time.sleep(0.1)
            continue

        for out in outputs.outputs:
            if (
                out.request_id in poc_request_ids
                and poc_completion_times[out.request_id] is None
            ):
                poc_completion_times[out.request_id] = time.time()
                completion_order.append((out.request_id, "poc"))
                nonce = poc_nonce_map[out.request_id]
                finish_reason_name = getattr(getattr(out, "finish_reason", None), "name", "")
                stop_reason = getattr(out, "stop_reason", None)
                artifact = "N/A"
                poc_result = getattr(out, "poc_result", None)
                if isinstance(poc_result, dict) and "vectors_b64" in poc_result:
                    vectors = poc_result["vectors_b64"]
                    artifact = vectors[0][:32] if vectors else "empty"

                if finish_reason_name == "ERROR":
                    error_msg = (
                        f"PoC request failed: id={out.request_id}, nonce={nonce}, "
                        f"stop_reason={stop_reason}"
                    )
                    scenario_errors.append(error_msg)
                    print(f"  ✗ {out.request_id} (nonce={nonce:2d}) error={stop_reason}")
                else:
                    print(f"  ✓ {out.request_id} (nonce={nonce:2d}) [{artifact}]")

            elif (
                out.request_id in inference_request_ids
                and inference_completion_times[out.request_id] is None
            ):
                inference_completion_times[out.request_id] = time.time()
                completion_order.append((out.request_id, "inference"))
                finish_reason_name = getattr(getattr(out, "finish_reason", None), "name", "")
                stop_reason = getattr(out, "stop_reason", None)
                if finish_reason_name == "ERROR":
                    error_msg = (
                        f"Inference request failed: id={out.request_id}, "
                        f"stop_reason={stop_reason}"
                    )
                    scenario_errors.append(error_msg)
                    print(f"  ✗ {out.request_id} error={stop_reason}")
                else:
                    print(f"  ✓ {out.request_id} completed")

        time.sleep(0.1)

    pending_requests = [rid for rid, t in poc_completion_times.items() if t is None] + [
        rid for rid, t in inference_completion_times.items() if t is None
    ]
    if pending_requests:
        try:
            engine_core.abort_requests(pending_requests)
        except Exception as e:
            scenario_errors.append(f"Abort failed for {scenario_name}: {e}")
            print(f"  ⚠ Failed to abort pending requests: {e}")

    if len(completion_order) < expected_total and time.time() >= deadline:
        scenario_errors.append(
            f"Timeout in {scenario_name}: completed {len(completion_order)}/{expected_total}"
        )

    scenario_end_time = time.time()
    scenario_duration = scenario_end_time - scenario_start_time

    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)

    print()
    print("Execution Timeline:")
    for i, (rid, kind) in enumerate(completion_order):
        if kind == "poc":
            submit_time = next(
                r["submit_time"] for r in results["poc_requests"] if r["id"] == rid
            )
            complete_time = poc_completion_times[rid]
            nonce = poc_nonce_map[rid]
            duration = complete_time - submit_time
            print(f"  {i + 1}. {rid:10s} (nonce={nonce:5d}) - {duration:.2f}s")
        else:
            submit_time = next(
                r["submit_time"]
                for r in results["inference_requests"]
                if r["id"] == rid
            )
            complete_time = inference_completion_times[rid]
            duration = complete_time - submit_time
            print(f"  {i + 1}. {rid:10s} ({kind:4s}) - {duration:.2f}s")

    print()
    print("-" * 80)
    print(f"Scenario Duration: {scenario_duration:.2f}s")
    completed_count = len(completion_order)
    total_count = expected_total
    print(f"Completed: {completed_count}/{total_count}")
    if scenario_errors:
        print("Errors:")
        for error in scenario_errors:
            print(f"  - {error}")
    print()

    return {
        "name": scenario_name,
        "duration": scenario_duration,
        "poc_count": num_poc_nonces,
        "inference_count": num_inference_requests,
        "inference_max_tokens": inference_max_tokens,
        "completed_count": completed_count,
        "total_count": total_count,
        "completion_order": completion_order,
        "errors": scenario_errors,
        "success": completed_count == total_count and len(scenario_errors) == 0,
    }


def profile_poc_and_chat():
    """Profile PoC with different scenarios."""

    print()
    print("=" * 80)
    print("PoC + Inference Coexistence Benchmark - 4 Scenarios")
    print("=" * 80)
    print("Model: Qwen/Qwen3-0.6B")
    print(f"PoC batch size: {POC_BATCH_SIZE_DEFAULT}")
    print(f"PoC priority: {POC_REQUEST_PRIORITY} (higher = yields to lower)")
    print()

    model_kwargs = dict(
        model="Qwen/Qwen3-0.6B",
        trust_remote_code=True,
        enforce_eager=True,
        skip_tokenizer_init=False,
    )

    llm = LLM(**model_kwargs)
    engine_core = llm.llm_engine.engine_core

    scenarios = []
    poc_nonces = 10

    scenario_configs = [
        {
            "name": "Scenario A: PoC only (baseline)",
            "num_poc_nonces": poc_nonces,
            "num_inference_requests": 0,
            "inference_max_tokens": 32,
            "timeout_seconds": 900.0,
        },
        {
            "name": "Scenario B: PoC + Inference (2k tokens)",
            "num_poc_nonces": poc_nonces,
            "num_inference_requests": 1,
            "inference_max_tokens": 2000,
            "timeout_seconds": 900.0,
        },
        {
            "name": "Scenario C: PoC + Inference (3k tokens)",
            "num_poc_nonces": poc_nonces,
            "num_inference_requests": 1,
            "inference_max_tokens": 3000,
            "timeout_seconds": 900.0,
        },
        {
            "name": "Scenario D: PoC + Inference (4k tokens)",
            "num_poc_nonces": poc_nonces,
            "num_inference_requests": 1,
            "inference_max_tokens": 4000,
            "timeout_seconds": 900.0,
        },
    ]

    for index, config in enumerate(scenario_configs):
        print()
        print(f"Running {config['name']} ({index + 1}/{len(scenario_configs)})...")
        print()
        try:
            scenarios.append(
                run_scenario(
                    engine_core=engine_core,
                    scenario_name=config["name"],
                    num_poc_nonces=config["num_poc_nonces"],
                    num_inference_requests=config["num_inference_requests"],
                    nonce_start=1 + index * 100000,
                    inference_max_tokens=config["inference_max_tokens"],
                    timeout_seconds=config["timeout_seconds"],
                )
            )
        except Exception as e:
            total_count = config["num_poc_nonces"] + config["num_inference_requests"]
            print(f"✗ {config['name']} crashed: {e}")
            scenarios.append(
                {
                    "name": config["name"],
                    "duration": 0.0,
                    "poc_count": config["num_poc_nonces"],
                    "inference_count": config["num_inference_requests"],
                    "inference_max_tokens": config["inference_max_tokens"],
                    "completed_count": 0,
                    "total_count": total_count,
                    "completion_order": [],
                    "errors": [f"Scenario crash: {e}"],
                    "success": False,
                }
            )

    # Print final comparison
    print()
    print()
    print("=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)
    print()
    print(
        f"{'Scenario':<42} {'Duration':<12} {'PoC':<6} {'Infer':<6} "
        f"{'Tokens':<8} {'Completed':<12} {'Delta vs A':<12} {'Status':<8}"
    )
    print("-" * 120)

    baseline_duration = None
    if scenarios and scenarios[0].get("success"):
        baseline_duration = scenarios[0]["duration"]

    for scenario in scenarios:
        total = scenario["total_count"]
        completed = f"{scenario['completed_count']}/{total}"
        delta = "N/A"
        if baseline_duration and scenario.get("success"):
            delta_pct = (scenario["duration"] - baseline_duration) / baseline_duration * 100
            delta = f"{delta_pct:+.1f}%"
        status = "OK" if scenario.get("success") else "FAILED"

        print(
            f"{scenario['name']:<42} "
            f"{scenario['duration']:>6.2f}s        "
            f"{scenario['poc_count']:>4d}   "
            f"{scenario['inference_count']:>4d}   "
            f"{scenario['inference_max_tokens']:>6d}   "
            f"{completed:>10s}   "
            f"{delta:>10s}   "
            f"{status:>6s}"
        )

        if scenario.get("errors"):
            for error in scenario["errors"]:
                print(f"    error: {error}")

    print()

    successful = [s for s in scenarios if s.get("success")]
    if baseline_duration and len(successful) > 1:
        print("Performance Analysis (vs Scenario A baseline):")
        for scenario in scenarios[1:]:
            if not scenario.get("success"):
                print(f"  - {scenario['name']}: skipped (failed)")
                continue
            overhead = (scenario["duration"] - baseline_duration) / baseline_duration * 100
            print(f"  - {scenario['name']}: {overhead:+.1f}%")

        max_success = max(scenarios[1:], key=lambda s: s["duration"] if s.get("success") else -1)
        if max_success.get("success"):
            print(
                "  - Worst-case successful overhead: "
                f"{((max_success['duration'] - baseline_duration) / baseline_duration * 100):+.1f}% "
                f"({max_success['name']})"
            )
    else:
        print("⚠ Performance analysis limited - baseline or mixed scenarios failed")

    print()
    print("=" * 80)
    print("✓ Benchmark complete!")
    print("=" * 80)

    try:
        engine_core.shutdown()
    finally:
        del llm


if __name__ == "__main__":
    profile_poc_and_chat()
