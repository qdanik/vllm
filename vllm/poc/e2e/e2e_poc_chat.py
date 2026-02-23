#!/usr/bin/env python3
"""Benchmark PoC execution with and without concurrent chat requests.

This script demonstrates how PoC and chat requests execute together in the
V1 scheduler with priority-based scheduling - using 3 scenarios:
    1. Scenario A: 10 PoC nonces + 4 chat requests (parallel)
    2. Scenario B: 14 PoC nonces only (baseline for comparison)
    3. Scenario C: 10 PoC nonces + 4 chat requests (verify consistency)

Compares execution speed and analyzes performance overhead.

NOTE: DeepGEMM warmup takes 6-10 minutes on first run to compile kernels.
      This is NORMAL and required for performance. Subsequent runs are fast.
"""

import os
import time

os.environ["VLLM_USE_V1"] = "1"

from vllm import LLM, SamplingParams
from vllm.poc.utils.env import POC_BATCH_SIZE_DEFAULT
from vllm.poc.constants import POC_REQUEST_PRIORITY
from vllm.poc.v1.params import PoCParams
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestKind

PUBLIC_KEY = "02e0f3b6b7f832ead7af2a235b9b27715a4d586b0fa108e735f0676a5086479225"
BLOCK_HASH = "8d148df1530d06a3412acd3deda4db16bae780eefdd160e081e6f878417de92a"


def run_scenario(
    engine_core,
    scenario_name: str,
    num_poc_nonces: int,
    num_chat_requests: int,
    poc_seq_len: int = 16,  # Reduced from 32
    poc_k_dim: int = 8,      # Reduced from 12
    chat_max_tokens: int = 32,  # Reduced from 50
) -> dict:
    """Run a single scenario and return results."""
    scenario_tag = (
        scenario_name.split(":", 1)[0]
        .replace("Scenario", "")
        .strip()
        .lower()
        or "scenario"
    )

    print("=" * 80)
    print(f"{scenario_name}")
    print("=" * 80)
    print(f"PoC nonces: {num_poc_nonces}, Chat requests: {num_chat_requests}")
    print()
    print(f"Submitting {num_poc_nonces} PoC nonces in parallel with {num_chat_requests} chat requests...")
    print()

    # Track request timings
    results = {
        "poc_requests": [],
        "chat_requests": [],
    }

    # Map request IDs to their nonces
    poc_nonce_map = {}
    scenario_start_time = time.time()

    # Submit PoC and chat requests in parallel (interleaved)
    poc_request_ids = []
    chat_request_ids = []
    
    # Submit all PoC requests
    for i in range(num_poc_nonces):
        request_id = f"{scenario_tag}-poc-{i}"
        poc_request_ids.append(request_id)
        nonce = i * 2 + 1
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
            poc_params=PoCParams(
                block_hash=BLOCK_HASH,
                public_key=PUBLIC_KEY,
                block_height=2732723,
                nonce=nonce,
                seq_len=poc_seq_len,
                k_dim=poc_k_dim,
            ),
        )

        submit_time = time.time()
        engine_core.add_request(req)
        results["poc_requests"].append({
            "id": request_id,
            "submit_time": submit_time,
            "nonce": nonce,
        })
        print(f"  [{submit_time:.2f}] Submitted PoC {request_id} (nonce={nonce})")
                                                                                  
    # Submit chat requests immediately after PoC (parallel execution)
    for i in range(num_chat_requests):
        request_id = f"{scenario_tag}-chat-{i}"
        chat_request_ids.append(request_id)

        req = EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=[1, 2, 3, 4, 5],  # Dummy prompt
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=chat_max_tokens),
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
        engine_core.add_request(req)
        results["chat_requests"].append({
            "id": request_id,
            "submit_time": submit_time,
        })
        print(f"  [{submit_time:.2f}] Submitted Chat {request_id} (priority=0)")
                                                                              
    print()
    print("Collecting results...")
    print()

    # Collect results - track which complete first
    completion_order = []
    poc_completion_times = {rid: None for rid in poc_request_ids}
    chat_completion_times = {rid: None for rid in chat_request_ids}

    deadline = time.time() + 600.0  # 10-minute timeout
    while time.time() < deadline:
        if len(completion_order) == (num_poc_nonces + num_chat_requests):
            break

        try:
            outputs = engine_core.get_output()
        except Exception as e:
            print(f"  ✗ Engine error: {e}")
            break

        if outputs is None or outputs.outputs is None:
            time.sleep(0.1)
            continue

        for out in outputs.outputs:
            if out.request_id in poc_request_ids and poc_completion_times[out.request_id] is None:
                poc_completion_times[out.request_id] = time.time()
                completion_order.append((out.request_id, "poc"))
                nonce = poc_nonce_map[out.request_id]
                print(f"    DEBUG: out={out}")
                print(f"    DEBUG: poc_result={getattr(out, 'poc_result', None)}")
                artifact = "N/A"
                poc_result = getattr(out, "poc_result", None)
                if isinstance(poc_result, dict) and "vectors_b64" in poc_result:
                    vectors = poc_result["vectors_b64"]
                    if vectors:
                        artifact = vectors[0][:32]
                    else:
                        artifact = "empty"
                print(f"  ✓ {out.request_id} (nonce={nonce:2d}) [{artifact}]")

            elif (
                out.request_id in chat_request_ids
                and chat_completion_times[out.request_id] is None
            ):
                chat_completion_times[out.request_id] = time.time()
                completion_order.append((out.request_id, "chat"))
                print(f"  ✓ {out.request_id} completed")

        time.sleep(0.1)

    pending_requests = [
        rid for rid, t in poc_completion_times.items() if t is None
    ] + [rid for rid, t in chat_completion_times.items() if t is None]
    if pending_requests:
        engine_core.abort_requests(pending_requests)

    scenario_end_time = time.time()
    scenario_duration = scenario_end_time - scenario_start_time

    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)

    # Analyze completion order
    poc_count_before_first_chat = 0
    first_chat_idx = None

    for idx, (rid, kind) in enumerate(completion_order):
        if kind == "chat" and first_chat_idx is None:
            first_chat_idx = idx
            poc_count_before_first_chat = idx

    print()
    print("Execution Timeline:")
    for i, (rid, kind) in enumerate(completion_order):
        if kind == "poc":
            submit_time = next(r["submit_time"] for r in results["poc_requests"] if r["id"] == rid)
            complete_time = poc_completion_times[rid]
            nonce = poc_nonce_map[rid]
            duration = complete_time - submit_time
            print(f"  {i + 1}. {rid:10s} (nonce={nonce:5d}) - {duration:.2f}s")
        else:
            submit_time = next(r["submit_time"] for r in results["chat_requests"] if r["id"] == rid)
            complete_time = chat_completion_times[rid]
            duration = complete_time - submit_time
            print(f"  {i + 1}. {rid:10s} ({kind:4s}) - {duration:.2f}s")

    print()
    print("-" * 80)
    print(f"Scenario Duration: {scenario_duration:.2f}s")
    completed_count = len(completion_order)
    total_count = num_poc_nonces + num_chat_requests
    print(f"Completed: {completed_count}/{total_count}")
    print()

    return {
        "name": scenario_name,
        "duration": scenario_duration,
        "poc_count": num_poc_nonces,
        "chat_count": num_chat_requests,
        "completed_count": completed_count,
        "completion_order": completion_order,
    }


def profile_poc_and_chat():
    """Profile PoC with different scenarios."""

    print()
    print("=" * 80)
    print("PoC + Chat Coexistence Benchmark - 3 Scenarios")
    print("=" * 80)
    print(f"Model: Qwen/Qwen3-0.6B")
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

    # Run scenarios
    scenarios = []
    
    # Scenario A: Chat only (test baseline)
    scenarios.append(run_scenario(
        engine_core,
        "Scenario A: 1 PoC + 2 Chat (chat baseline test)",
        num_poc_nonces=1,
        num_chat_requests=2,
    ))
    
    if scenarios[0]['completed_count'] == 0:
        print()
        print("✗ Scenario A failed completely. Aborting remaining scenarios.")
        return
    
    print()
    print("Continuing to Scenario B...")
    print()
    
    # Scenario B: 14 PoC only
    scenarios.append(run_scenario(
        engine_core,
        "Scenario B: 7 PoC only (baseline)",
        num_poc_nonces=7,
        num_chat_requests=0,
    ))
    
    if scenarios[1]['completed_count'] == 0:
        print()
        print("✗ Scenario B failed completely. Aborting remaining scenarios.")
        return
    
    print()
    print("Continuing to Scenario C...")
    print()
    
    # Scenario C: 10 PoC + 4 Chat again
    scenarios.append(run_scenario(
        engine_core,
        "Scenario C: 5 PoC + 2 Chat (verify consistency)",
        num_poc_nonces=5,
        num_chat_requests=2,
    ))

    # Print final comparison
    print()
    print()
    print("=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Scenario':<40} {'Duration':<12} {'PoC':<8} {'Chat':<8} {'Completed':<12}")
    print("-" * 80)
    for scenario in scenarios:
        completed = f"{scenario['completed_count']}/{scenario['poc_count'] + scenario['chat_count']}"
        print(
            f"{scenario['name']:<40} "
            f"{scenario['duration']:>6.2f}s        "
            f"{scenario['poc_count']:>6d}  "
            f"{scenario['chat_count']:>6d}  "
            f"{completed:>10s}"
        )
    
    print()
    
    # Analysis
    all_completed = all(
        s['completed_count'] == (s['poc_count'] + s['chat_count'])
        for s in scenarios
    )
    
    if all_completed and len(scenarios) == 3:
        print("Performance Analysis:")
        a_duration = scenarios[0]['duration']
        b_duration = scenarios[1]['duration']
        c_duration = scenarios[2]['duration']
        
        overhead_vs_b = ((a_duration - b_duration) / b_duration * 100)
        c_vs_b = ((c_duration - b_duration) / b_duration * 100)
        a_vs_c = abs(a_duration - c_duration) / max(a_duration, c_duration) * 100
        
        print(f"  A vs B (10+4 vs 14 PoC): {overhead_vs_b:+.1f}% overhead for 4 chat")
        print(f"  C vs B (10+4 vs 14 PoC): {c_vs_b:+.1f}% overhead for 4 chat")
        print(f"  A vs C (consistency):    {a_vs_c:.1f}% variation")
        print()
        
        if overhead_vs_b < 20:
            print("  ✓ Chat requests have minimal execution overhead")
        else:
            print("  ⚠ Significant overhead detected with mixed workloads")
    else:
        print("⚠ Performance analysis skipped - not all scenarios completed successfully")
    
    print()
    print("=" * 80)
    print("✓ Benchmark complete!")
    print("=" * 80)

    engine_core.shutdown()
    del llm


if __name__ == "__main__":
    profile_poc_and_chat()

