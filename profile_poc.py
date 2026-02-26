#!/usr/bin/env python3
"""Profile PoC through OpenAI API servers.

This script starts multiple vLLM OpenAI API server processes and profiles
PoC /api/v1/pow/generate calls in parallel.
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

from vllm.poc.env import (
    POC_BATCH_SIZE_DEFAULT,
    POC_PROFILE_DIST_THRESHOLD,
    POC_PROFILE_FRAUD_THRESHOLD,
    POC_PROFILE_P_MISMATCH,
    POC_PROFILE_RUNS,
    POC_PROFILE_VALIDATION_JSON,
)
from vllm.poc.validation import run_validation

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


def _build_device_slice(server_idx: int, tp_size: int) -> str:
    start = server_idx * tp_size
    end = start + tp_size
    return ",".join(str(i) for i in range(start, end))


def _build_device_slices(tp_size: int, api_server_count: int) -> List[str]:
    visible_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    required_devices = tp_size * api_server_count
    if visible_device_count < required_devices:
        raise RuntimeError(
            "Not enough visible GPUs in this environment for requested topology: "
            f"need {required_devices} GPUs for api_server_count={api_server_count}, "
            f"tensor_parallel_size={tp_size}, but only {visible_device_count} visible. "
            "If running in Docker, check --gpus and container visibility. "
            "Or lower api_server_count/tp_size, or set POC_PROFILE_DEVICE_SLICES explicitly."
        )

    return [_build_device_slice(i, tp_size) for i in range(api_server_count)]


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _read_log_tail(log_path: Path, max_lines: int = 80) -> str:
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return "<unable to read log file>"
    if not lines:
        return "<log is empty>"
    return "".join(lines[-max_lines:]).rstrip()


def _extract_root_cause(log_path: Path, window: int = 120) -> str:
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return "<unable to read log file>"

    if not lines:
        return "<log is empty>"

    keywords = (
        "ValueError",
        "RuntimeError",
        "AssertionError",
        "CUDA out of memory",
        "out of memory",
        "invalid device",
        "WorkerProc failed",
    )
    for idx, line in enumerate(lines):
        if any(keyword in line for keyword in keywords):
            start = max(0, idx - 40)
            end = min(len(lines), idx + window)
            return "".join(lines[start:end]).rstrip()

    for idx, line in enumerate(lines):
        if "Traceback (most recent call last):" in line:
            start = max(0, idx)
            end = min(len(lines), idx + window)
            return "".join(lines[start:end]).rstrip()

    return _read_log_tail(log_path, max_lines=120)


def _wait_for_health(
    server_idx: int,
    proc: subprocess.Popen,
    port: int,
    timeout_sec: int,
    log_path: Path,
) -> None:
    started = time.time()
    next_progress_ts = started + SERVER_STARTUP_PROGRESS_SEC
    while time.time() - started < timeout_sec:
        if proc.poll() is not None:
            tail = _read_log_tail(log_path)
            raise RuntimeError(
                f"API server {server_idx + 1} exited before becoming healthy "
                f"(port={port}, exit_code={proc.returncode}).\n"
                f"Last log lines ({log_path}):\n{tail}"
            )
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass

        now = time.time()
        if now >= next_progress_ts:
            elapsed = int(now - started)
            print(
                f"  waiting server {server_idx + 1} on :{port} "
                f"({elapsed}s/{timeout_sec}s)..."
            )
            next_progress_ts = now + SERVER_STARTUP_PROGRESS_SEC
        time.sleep(1)

    tail = _read_log_tail(log_path)
    raise TimeoutError(
        f"API server {server_idx + 1} on port {port} did not become healthy "
        f"in {timeout_sec}s.\nLast log lines ({log_path}):\n{tail}"
    )


def _start_server(
    server_idx: int,
    model: str,
    tp_size: int,
    device_slice: str,
    port: int,
    max_model_len: int,
    start_delay_sec: int = 0,
) -> tuple[int, subprocess.Popen, Any, str]:
    logs_dir = Path("logs/profile_poc")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"server_{server_idx + 1}.log"
    log_file = open(log_path, "w", buffering=1)

    env = os.environ.copy()
    env["VLLM_USE_V1"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = device_slice
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else f"{PROJECT_ROOT}:{existing_pythonpath}"
    )

    if start_delay_sec > 0:
        print(f"  delaying API server {server_idx + 1} start by {start_delay_sec}s...")
        time.sleep(start_delay_sec)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
        "--host",
        "0.0.0.0",
        "--tensor-parallel-size",
        str(tp_size),
        "--max-num-seqs",
        "32",
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        "float16",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=PROJECT_ROOT,
        start_new_session=True,
        text=True,
        bufsize=1,
    )

    if proc.stdout is not None:
        threading.Thread(
            target=_stream_server_logs,
            args=(server_idx, proc.stdout, log_file),
            daemon=True,
        ).start()

    print(
        f"  starting API server {server_idx + 1} on :{port} "
        f"(CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})"
    )

    try:
        _wait_for_health(
            server_idx=server_idx,
            proc=proc,
            port=port,
            timeout_sec=SERVER_STARTUP_TIMEOUT_SEC,
            log_path=log_path,
        )
    except Exception:
        _stop_process(proc)
        log_file.close()
        raise

    return server_idx, proc, log_file, str(log_path)


def _run_forward_api(
    base_url: str,
    model: str,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    k_dim: int,
    batch_size: int,
) -> tuple[float, List[Dict[str, Any]]]:
    payload = {
        "block_hash": block_hash,
        "block_height": 2732723,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {
            "model": model,
            "seq_len": seq_len,
            "k_dim": k_dim,
        },
        "batch_size": batch_size,
        "wait": True,
    }

    attempts = 4
    last_error: Optional[str] = None
    for attempt in range(1, attempts + 1):
        t0 = time.time()
        response = requests.post(
            f"{base_url}/api/v1/pow/generate",
            json=payload,
            timeout=300,
        )

        if 200 <= response.status_code < 300:
            elapsed = time.time() - t0
            body = response.json()
            artifacts = body.get("artifacts")
            if artifacts is None:
                raise RuntimeError(f"No artifacts in response: {body}")
            return elapsed, artifacts

        response_text = response.text.strip()
        detail = response_text
        try:
            response_json = response.json()
            detail = str(response_json.get("detail", response_json))
        except ValueError:
            pass

        last_error = (
            f"HTTP {response.status_code} from {base_url}/api/v1/pow/generate: {detail}"
        )

        server_idx = _BASE_URL_TO_SERVER_IDX.get(base_url)
        if server_idx is not None:
            tail_lines = list(_SERVER_LOG_TAILS.get(server_idx, deque()))[-40:]
            if tail_lines:
                tail_text = "\n".join(tail_lines)
                last_error = (
                    f"{last_error}\nRecent server-{server_idx + 1} logs:\n{tail_text}"
                )

        raise RuntimeError(last_error)

    raise RuntimeError(last_error or "PoC request failed with unknown error")


def profile_poc() -> None:
    profile_runs = POC_PROFILE_RUNS
    dist_threshold = POC_PROFILE_DIST_THRESHOLD
    p_mismatch = POC_PROFILE_P_MISMATCH
    fraud_threshold = POC_PROFILE_FRAUD_THRESHOLD

    model = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
    seq_len = 1024
    k_dim = 12
    tp_size = 4
    api_server_count = 2
    max_model_len = 240000

    public_key = PUBLIC_KEY
    block_hash = BLOCK_HASH

    ports = [BASE_PORT + i for i in range(api_server_count)]
    base_urls = [f"http://127.0.0.1:{port}" for port in ports]
    _BASE_URL_TO_SERVER_IDX.clear()
    _BASE_URL_TO_SERVER_IDX.update({base_urls[i]: i for i in range(api_server_count)})
    device_slices = _build_device_slices(tp_size, api_server_count)

    print("=" * 70)
    print("Profiling PoC via OpenAI API /api/v1/pow/generate")
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
    print(f"Using vllm from: {vllm.__file__}")
    print(f"Project root: {PROJECT_ROOT}")
    print(
        f"  dist_threshold={dist_threshold}, p_mismatch={p_mismatch}, fraud_threshold={fraud_threshold}"
    )

    server_procs: List[subprocess.Popen] = []
    log_files: List[Any] = []
    start_delays = [0, 5]

    try:
        print("\nStarting OpenAI API servers in parallel...")
        start_t0 = time.time()
        with ThreadPoolExecutor(max_workers=api_server_count) as executor:
            futures = [
                executor.submit(
                    _start_server,
                    server_idx,
                    model,
                    tp_size,
                    device_slices[server_idx],
                    ports[server_idx],
                    max_model_len,
                    start_delays[server_idx] if server_idx < len(start_delays) else 0,
                )
                for server_idx in range(api_server_count)
            ]
            ready: List[Optional[tuple[subprocess.Popen, Any]]] = [None] * api_server_count
            for future in as_completed(futures):
                try:
                    server_idx, proc, log_file, log_path = future.result()
                except Exception as exc:
                    print(f"\n[ERROR] Server startup failed: {exc}")
                    for idx in range(api_server_count):
                        log_path = Path("logs/profile_poc") / f"server_{idx + 1}.log"
                        if log_path.exists():
                            print(
                                f"\n[LOG DIAG] server {idx + 1} root-cause excerpt ({log_path}):\n"
                                f"{_extract_root_cause(log_path)}"
                            )
                    raise
                ready[server_idx] = (proc, log_file)
                print(
                    f"  API server ready: {server_idx + 1}/{api_server_count} "
                    f"(port={ports[server_idx]}, log={log_path})"
                )

        for item in ready:
            if item is None:
                raise RuntimeError("Server startup failed")
            proc, log_file = item
            server_procs.append(proc)
            log_files.append(log_file)

        print(f"Parallel server startup finished in {time.time() - start_t0:.1f}s")

        batch_size = POC_BATCH_SIZE_DEFAULT
        print(f"\nRunning {profile_runs} batches with batch_size={batch_size}...")

        warmup_nonces = list(range(batch_size))
        print("\nWarmup run (parallel for all API servers)...")
        with ThreadPoolExecutor(max_workers=api_server_count) as executor:
            futures = [
                executor.submit(
                    _run_forward_api,
                    base_urls[i],
                    model,
                    block_hash,
                    public_key,
                    warmup_nonces,
                    seq_len,
                    k_dim,
                    batch_size,
                )
                for i in range(api_server_count)
            ]
            for future in as_completed(futures):
                future.result()

        times = []
        combined_times = []
        all_hashes = []
        total_nonces = 0
        per_engine_stats: Dict[int, Dict[str, float]] = {
            i: {"runs": 0, "nonces": 0, "time": 0.0} for i in range(api_server_count)
        }

        print("\nProfiling...")
        validation_payload = _load_validation_payload()
        validation_map = _build_validation_map(validation_payload) if validation_payload else {}
        validated_engines = set()

        for run in range(profile_runs):
            run_start_nonce = run * batch_size * api_server_count
            per_engine_nonces = {
                engine_idx: list(
                    range(
                        run_start_nonce + engine_idx * batch_size,
                        run_start_nonce + (engine_idx + 1) * batch_size,
                    )
                )
                for engine_idx in range(api_server_count)
            }

            wall_t0 = time.time()
            with ThreadPoolExecutor(max_workers=api_server_count) as executor:
                futures = {
                    executor.submit(
                        _run_forward_api,
                        base_urls[engine_idx],
                        model,
                        block_hash,
                        public_key,
                        per_engine_nonces[engine_idx],
                        seq_len,
                        k_dim,
                        batch_size,
                    ): engine_idx
                    for engine_idx in range(api_server_count)
                }

                run_results: Dict[int, tuple[float, List[Dict[str, Any]], List[int]]] = {}
                for future in as_completed(futures):
                    engine_idx = futures[future]
                    elapsed, artifacts = future.result()
                    run_results[engine_idx] = (
                        elapsed,
                        artifacts,
                        per_engine_nonces[engine_idx],
                    )

            wall_elapsed = time.time() - wall_t0
            combined_times.append(wall_elapsed)

            run_total_nonces = 0
            print(f"  Parallel run {run + 1:2d}: wall={wall_elapsed * 1000:.1f}ms")
            for engine_idx in sorted(run_results.keys()):
                elapsed, artifacts, batch_nonces = run_results[engine_idx]
                times.append(elapsed)

                vectors_b64 = [artifact["vector_b64"] for artifact in artifacts]
                if vectors_b64:
                    all_hashes.append(vectors_b64[0])

                len_nonces = len(vectors_b64)
                run_total_nonces += len_nonces
                total_nonces += len_nonces

                per_engine_stats[engine_idx]["runs"] += 1
                per_engine_stats[engine_idx]["nonces"] += len_nonces
                per_engine_stats[engine_idx]["time"] += elapsed

                nonces_per_sec = len_nonces / elapsed if elapsed > 0 else 0
                ms_per_nonce = elapsed * 1000 / len_nonces if len_nonces > 0 else 0
                print(
                    f"    Engine {engine_idx + 1}: {elapsed * 1000:.1f}ms, {len_nonces} nonces "
                    f"({nonces_per_sec:.1f}/sec, {ms_per_nonce:.2f}ms/nonce)"
                )

                if validation_map and engine_idx not in validated_engines:
                    computed_artifacts = artifacts

                    print(f"\n[DEBUG] Validation Info (engine {engine_idx + 1}):")
                    print(
                        f"  validation_map has {len(validation_map)} entries: {sorted(validation_map.keys())}"
                    )
                    print(
                        f"  batch_nonces: {batch_nonces[:5]}... (first 5 of {len(batch_nonces)})"
                    )
                    print(f"  computed_artifacts has {len(computed_artifacts)} entries")

                    matching_nonces = [n for n in batch_nonces if n in validation_map]
                    print(f"  matching nonces: {matching_nonces} ({len(matching_nonces)} found)")

                    if matching_nonces:
                        for nonce in matching_nonces:
                            expected = validation_map[nonce]
                            computed = next(
                                a["vector_b64"] for a in computed_artifacts if a["nonce"] == nonce
                            )
                            match = "✓" if expected == computed else "✗"
                            print(f"    nonce {nonce}: {match}")
                            if expected != computed:
                                print(f"      expected: {expected}")
                                print(f"      got:      {computed}")

                    try:
                        validation_result = run_validation(
                            computed_artifacts,
                            validation_map,
                            len(computed_artifacts),
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

                    validated_engines.add(engine_idx)
                    print(f"\nValidation result (engine {engine_idx + 1}):")
                    print(
                        f"  n_total={validation_result['n_total']}, "
                        f"n_mismatch={validation_result['n_mismatch']}, "
                        f"p_value={validation_result['p_value']:.6f}, "
                        f"fraud_detected={validation_result['fraud_detected']}"
                    )
                    if validation_result.get("mismatch_nonces"):
                        print(f"  mismatch_nonces={validation_result['mismatch_nonces']}")
                    print()

            combined_rate = run_total_nonces / wall_elapsed if wall_elapsed > 0 else 0
            print(
                f"    Combined: {run_total_nonces} nonces in {wall_elapsed * 1000:.1f}ms "
                f"({combined_rate:.1f}/sec)"
            )

        print("\n" + "=" * 70)
        print("RESULTS:")
        print("=" * 70)

        if times:
            avg_time = sum(times) / len(times)
            avg_parallel_time = (
                sum(combined_times) / len(combined_times) if combined_times else avg_time
            )
            avg_nonces = total_nonces / len(times)
            avg_rate = avg_nonces / avg_time if avg_time > 0 else 0
            avg_combined_nonces_per_run = (
                total_nonces / len(combined_times) if combined_times else avg_nonces
            )
            avg_combined_rate = (
                avg_combined_nonces_per_run / avg_parallel_time
                if avg_parallel_time > 0
                else 0
            )
            time_per_nonce = avg_time / avg_nonces if avg_nonces > 0 else 0

            print(f"Batch size used: {batch_size}")
            print(f"Total batches: {len(times)}")
            print(f"Parallel rounds: {len(combined_times)}")
            print(f"Total nonces: {total_nonces}")
            print(f"Average batch time: {avg_time * 1000:.1f}ms")
            print(f"Average parallel round time: {avg_parallel_time * 1000:.1f}ms")
            print(f"Average rate: {avg_rate:.2f} nonces/sec")
            print(f"Average combined parallel rate: {avg_combined_rate:.2f} nonces/sec")
            print(f"Average rate: {avg_rate * 60:.0f} nonces/min")
            print(f"Time per nonce: {time_per_nonce * 1000:.2f}ms")

            print("\nPer-engine summary:")
            for engine_idx in range(api_server_count):
                e_runs = per_engine_stats[engine_idx]["runs"]
                e_nonces = per_engine_stats[engine_idx]["nonces"]
                e_time = per_engine_stats[engine_idx]["time"]
                e_rate = e_nonces / e_time if e_time > 0 else 0
                e_ms_per_nonce = (e_time * 1000 / e_nonces) if e_nonces > 0 else 0
                print(
                    f"  Engine {engine_idx + 1}: runs={int(e_runs)}, nonces={int(e_nonces)}, "
                    f"rate={e_rate:.2f}/sec, ms/nonce={e_ms_per_nonce:.2f}"
                )

            if len(times) > 1:
                min_time = min(times) * 1000
                max_time = max(times) * 1000
                variance = (
                    ((max_time - min_time) / (avg_time * 1000) * 100)
                    if avg_time > 0
                    else 0
                )
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
                print(f"  ✓ Exceeded target by {-gap:.1f}%!")

            if avg_rate > 0:
                print(
                    f"\nEstimated time for 1000 nonces: {1000 / avg_rate:.1f}s "
                    f"({1000 / avg_rate / 60:.1f}min)"
                )
                print(
                    f"Estimated time for 10000 nonces: {10000 / avg_rate:.1f}s "
                    f"({10000 / avg_rate / 60:.1f}min)"
                )
        else:
            print("No timing data collected!")

    finally:
        for proc in server_procs:
            _stop_process(proc)
        for log_file in log_files:
            try:
                log_file.flush()
                log_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    profile_poc()
