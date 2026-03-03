#!/usr/bin/env python3
"""Benchmark how heavy inference load impacts PoC throughput.

Methodology (fixed-time window):
    - Run each scenario for N seconds (default: 10)
    - Continuously keep PoC requests in-flight
    - Continuously keep inference requests in-flight (for mixed scenarios)
    - Measure how many PoC nonces complete within the same N-second window

Primary metric:
    - PoC throughput = successful PoC completions / window_seconds

Secondary metrics:
    - PoC p50 / p95 latency for completions within the window
    - Throughput degradation vs baseline (PoC-only)

NOTE: DeepGEMM warmup takes 6-10 minutes on first run to compile kernels.
      This is NORMAL and required for performance. Subsequent runs are fast.
"""

import os
import time

os.environ["VLLM_USE_V1"] = "1"

from vllm import LLM, SamplingParams
from vllm.poc.constants import POC_REQUEST_PRIORITY
from vllm.poc.engine.params import PoCSchedulerParams
from vllm.poc.env import POC_BATCH_SIZE_DEFAULT
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestKind

PUBLIC_KEY = "02e0f3b6b7f832ead7af2a235b9b27715a4d586b0fa108e735f0676a5086479225"
BLOCK_HASH = "8d148df1530d06a3412acd3deda4db16bae780eefdd160e081e6f878417de92a"
MODEL_NAME = "Qwen/Qwen3-0.6B"
WINDOW_SECONDS = float(os.getenv("POC_WINDOW_SECONDS", "10"))
# Keep default conservative to avoid overloading mixed PoC+inference path.
POC_INFLIGHT_TARGET = int(
    os.getenv("POC_INFLIGHT_TARGET", str(max(2, min(8, POC_BATCH_SIZE_DEFAULT))))
)
INFERENCE_STREAMS_DEFAULT = 1
MODEL_ENFORCE_EAGER = os.getenv("POC_MODEL_ENFORCE_EAGER", "0") == "1"
MAX_SUBMITS_PER_TICK = int(os.getenv("POC_MAX_SUBMITS_PER_TICK", "2"))
DISABLE_PREFIX_CACHING = os.getenv("POC_DISABLE_PREFIX_CACHING", "1") == "1"
POC_DUMMY_TOKEN_ID = int(os.getenv("POC_DUMMY_TOKEN_ID", "11"))
INFER_DUMMY_PROMPT = [
    int(x)
    for x in os.getenv("POC_INFER_PROMPT_IDS", "101,102,103,104,105").split(",")
]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    position = int(round(q * (len(sorted_values) - 1)))
    position = max(0, min(len(sorted_values) - 1, position))
    return sorted_values[position]


def run_scenario(
    engine_core,
    scenario_name: str,
    window_seconds: float,
    poc_inflight_target: int,
    num_inference_streams: int,
    poc_seq_len: int = 16,
    poc_k_dim: int = 8,
    nonce_start: int = 1,
    inference_max_tokens: int = 32,
) -> dict:
    """Run fixed-duration load test and return throughput metrics."""
    scenario_tag = (
        scenario_name.split(":", 1)[0].replace("Scenario", "").strip().lower()
        or "scenario"
    )

    print("=" * 80)
    print(f"{scenario_name}")
    print("=" * 80)
    print(
        "Window: "
        f"{window_seconds:.1f}s, PoC inflight target: {poc_inflight_target}, "
        f"Inference streams: {num_inference_streams}, "
        f"Nonce start: {nonce_start}, "
        f"Inference max_tokens: {inference_max_tokens if num_inference_streams else 'N/A'}"
    )
    print()
    print("Starting continuous load...")
    print()

    window_start = time.time()
    window_end = window_start + window_seconds

    submitted_times: dict[str, float] = {}
    request_kind: dict[str, str] = {}
    poc_nonce_map: dict[str, int] = {}
    in_flight_poc: set[str] = set()
    in_flight_infer: set[str] = set()

    poc_submitted = 0
    poc_success = 0
    poc_failed = 0
    infer_submitted = 0
    infer_success = 0
    infer_failed = 0
    poc_latencies: list[float] = []
    completion_order: list[tuple[str, str, bool]] = []
    scenario_errors: list[str] = []

    next_nonce = nonce_start
    next_poc_idx = 0
    next_infer_idx = 0

    def submit_poc_request() -> bool:
        nonlocal poc_submitted, next_nonce, next_poc_idx

        request_id = f"{scenario_tag}-poc-{next_poc_idx}"
        next_poc_idx += 1
        nonce = next_nonce
        next_nonce += 2
        poc_nonce_map[request_id] = nonce

        req = EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=[POC_DUMMY_TOKEN_ID],
            mm_features=None,
            sampling_params=SamplingParams(),
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

        try:
            engine_core.add_request(req)
            submitted_times[request_id] = time.time()
            request_kind[request_id] = "poc"
            in_flight_poc.add(request_id)
            poc_submitted += 1
            return True
        except Exception as e:
            scenario_errors.append(f"Failed to submit {request_id}: {e}")
            return False

    def submit_inference_request() -> bool:
        nonlocal infer_submitted, next_infer_idx

        request_id = f"{scenario_tag}-infer-{next_infer_idx}"
        next_infer_idx += 1

        req = EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=INFER_DUMMY_PROMPT,
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=inference_max_tokens),
            pooling_params=None,
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            client_index=0,
            priority=0,
            kind=EngineCoreRequestKind.GENERATE,
        )

        try:
            engine_core.add_request(req)
            submitted_times[request_id] = time.time()
            request_kind[request_id] = "inference"
            in_flight_infer.add(request_id)
            infer_submitted += 1
            return True
        except Exception as e:
            scenario_errors.append(f"Failed to submit {request_id}: {e}")
            return False

    initial_poc_submits = 0
    while len(in_flight_poc) < poc_inflight_target and initial_poc_submits < MAX_SUBMITS_PER_TICK:
        if not submit_poc_request():
            break
        initial_poc_submits += 1

    initial_infer_submits = 0
    while (
        len(in_flight_infer) < num_inference_streams
        and initial_infer_submits < MAX_SUBMITS_PER_TICK
    ):
        if not submit_inference_request():
            break
        initial_infer_submits += 1

    print(
        f"Initial load: PoC in-flight={len(in_flight_poc)}, "
        f"Inference in-flight={len(in_flight_infer)}"
    )

    while time.time() < window_end:
        poc_submits_this_tick = 0
        while (
            len(in_flight_poc) < poc_inflight_target
            and time.time() < window_end
            and poc_submits_this_tick < MAX_SUBMITS_PER_TICK
        ):
            if not submit_poc_request():
                break
            poc_submits_this_tick += 1

        infer_submits_this_tick = 0
        while (
            len(in_flight_infer) < num_inference_streams
            and num_inference_streams > 0
            and time.time() < window_end
            and infer_submits_this_tick < MAX_SUBMITS_PER_TICK
        ):
            if not submit_inference_request():
                break
            infer_submits_this_tick += 1

        try:
            outputs = engine_core.get_output()
        except Exception as e:
            scenario_errors.append(f"Engine error in {scenario_name}: {e}")
            print(f"  ✗ Engine error: {e}")
            break

        if outputs is None or outputs.outputs is None:
            time.sleep(0.05)
            continue

        for out in outputs.outputs:
            rid = out.request_id
            kind = request_kind.get(rid)
            if kind is None:
                continue

            finish_reason = getattr(out, "finish_reason", None)
            # For GENERATE requests, outputs can be streamed incrementally.
            # Only treat as completion when finish_reason is present.
            if finish_reason is None:
                continue

            finished_at = time.time()
            in_window = finished_at <= window_end
            finish_reason_name = getattr(
                finish_reason, "name", ""
            )
            stop_reason = getattr(out, "stop_reason", None)
            is_error = finish_reason_name == "ERROR"

            if kind == "poc":
                in_flight_poc.discard(rid)
                nonce = poc_nonce_map.get(rid, -1)
                completion_order.append((rid, "poc", not is_error))

                if is_error:
                    poc_failed += 1
                    error_msg = (
                        f"PoC request failed: id={rid}, nonce={nonce}, "
                        f"stop_reason={stop_reason}"
                    )
                    scenario_errors.append(error_msg)
                    print(f"  ✗ {rid} (nonce={nonce:2d}) error={stop_reason}")
                else:
                    if in_window:
                        poc_success += 1
                        submit_time = submitted_times.get(rid)
                        if submit_time is not None:
                            poc_latencies.append(finished_at - submit_time)
                    print(f"  ✓ {rid} (nonce={nonce:2d})")

            elif kind == "inference":
                in_flight_infer.discard(rid)
                completion_order.append((rid, "inference", not is_error))

                if is_error:
                    infer_failed += 1
                    error_msg = (
                        f"Inference request failed: id={rid}, "
                        f"stop_reason={stop_reason}"
                    )
                    scenario_errors.append(error_msg)
                    print(f"  ✗ {rid} error={stop_reason}")
                else:
                    if in_window:
                        infer_success += 1
                    print(f"  ✓ {rid} completed")

            submitted_times.pop(rid, None)
            request_kind.pop(rid, None)

        time.sleep(0.05)

    pending_requests = list(in_flight_poc) + list(in_flight_infer)
    if pending_requests:
        try:
            engine_core.abort_requests(pending_requests)
        except Exception as e:
            scenario_errors.append(f"Abort failed for {scenario_name}: {e}")
            print(f"  ⚠ Failed to abort pending requests: {e}")

    scenario_duration = window_end - window_start
    poc_throughput = poc_success / scenario_duration if scenario_duration > 0 else 0.0
    infer_throughput = (
        infer_success / scenario_duration if scenario_duration > 0 else 0.0
    )
    poc_p50 = _percentile(poc_latencies, 0.50)
    poc_p95 = _percentile(poc_latencies, 0.95)

    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print()
    print(f"Window: {scenario_duration:.2f}s")
    print(f"PoC submitted/completed/errors: {poc_submitted}/{poc_success}/{poc_failed}")
    print(
        f"Inference submitted/completed/errors: "
        f"{infer_submitted}/{infer_success}/{infer_failed}"
    )
    print(f"PoC throughput: {poc_throughput:.2f} nonces/s")
    print(f"Inference throughput: {infer_throughput:.2f} req/s")
    print(f"PoC latency p50/p95: {poc_p50:.3f}s / {poc_p95:.3f}s")
    if scenario_errors:
        print("Errors:")
        for error in scenario_errors:
            print(f"  - {error}")
    print()

    return {
        "name": scenario_name,
        "duration": scenario_duration,
        "poc_inflight_target": poc_inflight_target,
        "inference_streams": num_inference_streams,
        "inference_max_tokens": inference_max_tokens,
        "poc_submitted": poc_submitted,
        "poc_completed": poc_success,
        "poc_failed": poc_failed,
        "poc_throughput": poc_throughput,
        "poc_latency_p50": poc_p50,
        "poc_latency_p95": poc_p95,
        "infer_submitted": infer_submitted,
        "infer_completed": infer_success,
        "infer_failed": infer_failed,
        "infer_throughput": infer_throughput,
        "completion_order": completion_order,
        "errors": scenario_errors,
        "success": len(scenario_errors) == 0,
    }


def profile_poc_and_chat():
    """Profile PoC throughput impact from heavy inference load."""

    print()
    print("=" * 80)
    print("PoC Throughput Impact Benchmark - Fixed 10s Window")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Window duration: {WINDOW_SECONDS:.1f}s")
    print(f"PoC in-flight target: {POC_INFLIGHT_TARGET}")
    print(f"Max submits per tick: {MAX_SUBMITS_PER_TICK}")
    print(f"enforce_eager: {MODEL_ENFORCE_EAGER}")
    print(f"disable_prefix_caching: {DISABLE_PREFIX_CACHING}")
    print(f"poc_dummy_token_id: {POC_DUMMY_TOKEN_ID}")
    print(f"infer_dummy_prompt_ids: {INFER_DUMMY_PROMPT}")
    print(f"PoC batch size: {POC_BATCH_SIZE_DEFAULT}")
    print(f"PoC priority: {POC_REQUEST_PRIORITY} (higher = yields to lower)")
    print()

    model_kwargs = dict(
        model=MODEL_NAME,
        trust_remote_code=True,
        enforce_eager=MODEL_ENFORCE_EAGER,
        enable_prefix_caching=not DISABLE_PREFIX_CACHING,
        skip_tokenizer_init=False,
    )

    llm = LLM(**model_kwargs)
    engine_core = llm.llm_engine.engine_core

    scenarios: list[dict] = []

    scenario_configs = [
        {
            "name": "Scenario A: PoC only (baseline)",
            "num_inference_streams": 0,
            "inference_max_tokens": 32,
        },
        {
            "name": "Scenario B: PoC + Inference (2k tokens)",
            "num_inference_streams": INFERENCE_STREAMS_DEFAULT,
            "inference_max_tokens": 2000,
        },
        {
            "name": "Scenario C: PoC + Inference (4k tokens)",
            "num_inference_streams": INFERENCE_STREAMS_DEFAULT,
            "inference_max_tokens": 4000,
        },
        {
            "name": "Scenario D: PoC + Inference (6k tokens)",
            "num_inference_streams": INFERENCE_STREAMS_DEFAULT,
            "inference_max_tokens": 6000,
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
                    window_seconds=WINDOW_SECONDS,
                    poc_inflight_target=POC_INFLIGHT_TARGET,
                    num_inference_streams=config["num_inference_streams"],
                    nonce_start=1 + index * 1000,
                    inference_max_tokens=config["inference_max_tokens"],
                )
            )
        except Exception as e:
            print(f"✗ {config['name']} crashed: {e}")
            scenarios.append(
                {
                    "name": config["name"],
                    "duration": 0.0,
                    "poc_inflight_target": POC_INFLIGHT_TARGET,
                    "inference_streams": config["num_inference_streams"],
                    "inference_max_tokens": config["inference_max_tokens"],
                    "poc_submitted": 0,
                    "poc_completed": 0,
                    "poc_failed": 0,
                    "poc_throughput": 0.0,
                    "poc_latency_p50": 0.0,
                    "poc_latency_p95": 0.0,
                    "infer_submitted": 0,
                    "infer_completed": 0,
                    "infer_failed": 0,
                    "infer_throughput": 0.0,
                    "completion_order": [],
                    "errors": [f"Scenario crash: {e}"],
                    "success": False,
                }
            )

    print()
    print()
    print("=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)
    print()
    print(
        f"{'Scenario':<42} {'PoC n/s':<10} {'PoC done':<10} {'PoC p95':<10} "
        f"{'Infer':<6} {'Tokens':<8} {'Delta vs A':<12} {'Status':<8}"
    )
    print("-" * 120)

    baseline_throughput = None
    if scenarios and scenarios[0].get("success"):
        baseline_throughput = scenarios[0]["poc_throughput"]

    for scenario in scenarios:
        delta = "N/A"
        if baseline_throughput and scenario.get("success"):
            delta_pct = (
                (scenario["poc_throughput"] - baseline_throughput)
                / baseline_throughput
                * 100
            )
            delta = f"{delta_pct:+.1f}%"
        status = "OK" if scenario.get("success") else "FAILED"

        print(
            f"{scenario['name']:<42} "
            f"{scenario['poc_throughput']:>7.2f}   "
            f"{scenario['poc_completed']:>8d}   "
            f"{scenario['poc_latency_p95']:>7.3f}s   "
            f"{scenario['inference_streams']:>4d}   "
            f"{scenario['inference_max_tokens']:>6d}   "
            f"{delta:>10s}   "
            f"{status:>6s}"
        )

        if scenario.get("errors"):
            for error in scenario["errors"]:
                print(f"    error: {error}")

    print()

    successful = [s for s in scenarios if s.get("success")]
    if baseline_throughput and len(successful) > 1:
        print("Performance Analysis (PoC throughput vs Scenario A baseline):")
        for scenario in scenarios[1:]:
            if not scenario.get("success"):
                print(f"  - {scenario['name']}: skipped (failed)")
                continue
            degradation = (
                (baseline_throughput - scenario["poc_throughput"])
                / baseline_throughput
                * 100
            )
            print(f"  - {scenario['name']}: {degradation:+.1f}% degradation")

        worst = min(
            (s for s in scenarios[1:] if s.get("success")),
            key=lambda s: s["poc_throughput"],
            default=None,
        )
        if worst is not None:
            print(
                "  - Worst-case PoC degradation: "
                f"{((baseline_throughput - worst['poc_throughput']) / baseline_throughput * 100):+.1f}% "
                f"({worst['name']})"
            )
    else:
        print("⚠ Performance analysis limited - baseline or mixed scenarios failed")

    print()
    print("Recommended primary metric: PoC nonces/s in fixed 10s windows.")
    print(
        "Optional stability metric: run each scenario 3-5 times and compare median "
        "PoC throughput to reduce warmup/noise effects."
    )

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
