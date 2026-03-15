#!/usr/bin/env python3
"""Profile PoC through OpenAI API servers.

This script starts multiple vLLM OpenAI API server processes and profiles
PoC /api/v1/pow/generate calls in parallel.
"""

import argparse
import contextlib
import http.server
import json as _json
import os
import queue as stdlib_queue
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import torch

import vllm
from vllm.poc.env import (
    POC_PROFILE_DIST_THRESHOLD,
    POC_PROFILE_FRAUD_THRESHOLD,
    POC_PROFILE_P_MISMATCH,
)
from vllm.poc.server.models import Artifact
from vllm.poc.server.validation import validate_artifacts

stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(line_buffering=True, write_through=True)
stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
if callable(stderr_reconfigure):
    stderr_reconfigure(line_buffering=True, write_through=True)

PUBLIC_KEY = "test_pub_key"
BLOCK_HASH = "TEST_BLOCK"
VALIDATION_SAMPLE = {
    "public_key": PUBLIC_KEY,
    "block_hash": BLOCK_HASH,
    "block_height": 2732723,
    "node_id": 1,
    "artifacts": [
        {"nonce": 0, "vector_b64": "6bIBLJ84d7S7KgK2kDOOs201Va3xMwq1"},
        {"nonce": 1, "vector_b64": "8rGdrQi3Vq58LTC09TEwMH42dDM5Khm5"},
        {"nonce": 2, "vector_b64": "VKw5Nh40yTcgNE0gdyxptzYlrLe4r4qu"},
        {"nonce": 3, "vector_b64": "NTVxKwYmsjdDMzMwAyybMdStQjfiNlM2"},
        {"nonce": 4, "vector_b64": "ByggMOa0WDQzMtOt0bXoOfwogjSrrawr"},
        {"nonce": 5, "vector_b64": "xJgTOZy3u7AQLFCsB7RRtG4zrh6bMMY1"},
        {"nonce": 6, "vector_b64": "S7jVr7A0o7SItjA3D6xuKhKnvLQVnQ20"},
        {"nonce": 7, "vector_b64": "5racsxKxXbZHtZKpoitTssSwbDR7OOOt"},
        {"nonce": 8, "vector_b64": "/jRANrQxADWfMNg2OzTnMvs0WSRAti4z"},
        {"nonce": 9, "vector_b64": "6qyEMBy2gi9KNyk4nDI1sZqybDY/pwc0"},
        {"nonce": 10, "vector_b64": "6zZJtciqzSkHNAW5Z7DJKgW0B7VzMzMs"},
        {"nonce": 11, "vector_b64": "uTIQNU058LHtJGUyiCQquNYwr60IMjgs"},
        {"nonce": 12, "vector_b64": "xrY7rWG0PbKBNH8kxLa9pKgxNDRqODgy"},
        {"nonce": 13, "vector_b64": "2jdKtCWw37BWNjkuLjbTrzc3Py8CNSWr"},
        {"nonce": 14, "vector_b64": "4LRvKLG0fC0CtsI2KbCdsaipFznLMR2t"},
        {"nonce": 15, "vector_b64": "7DCxL4U2vrDkMzM5BjDgNl00hyR6rJ2u"},
        {"nonce": 16, "vector_b64": "p7AKrwKzbLTGLNk4dbGsNseviSkvtNG2"},
        {"nonce": 17, "vector_b64": "y7UHMssuRaZ0MK61ljGHObwtJbIJMr+0"},
        {"nonce": 18, "vector_b64": "sDJ1NnqvjbPBtfE4H7ThqeQwU7AGtVcs"},
        {"nonce": 19, "vector_b64": "/a18Lz64QyGFtdO0MLPBMXwpN7TFNOW3"},
        {"nonce": 20, "vector_b64": "XTaBsNc2KjijJ7KslyywrVSZ/jK5tqU1"},
        {"nonce": 21, "vector_b64": "47J/Niwwl606s0O1QLUYsTQ29jJpNCE3"},
        {"nonce": 22, "vector_b64": "TzXhtD4jXThQtKezsza0sPomBauSNpSY"},
        {"nonce": 23, "vector_b64": "TDHDrAS0eioHohc5HbPxNK2pYrBNOJay"},
        {"nonce": 24, "vector_b64": "B7Jfp74v+q8nuPy2nzWcs0WxJjF3t9Qw"},
        {"nonce": 25, "vector_b64": "3DDoNPMmXLfbMIw29zFvrF8w2DQKuGY0"},
        {"nonce": 26, "vector_b64": "RTSzMzw5uzGXs38yNhpCOAgtrq2GqNkv"},
        {"nonce": 27, "vector_b64": "qLYwLoy0vrUWLGgscbWNraY4bbXhMMub"},
        {"nonce": 28, "vector_b64": "m7BHt442VzbNNQi1/qAhMco2AKh7LssX"},
        {"nonce": 29, "vector_b64": "LzF2tj4UWjX8MVU0Uzh0tPirurVtNDmv"},
        {"nonce": 30, "vector_b64": "Fa/xN/E1Oq3bLqww5C7isHI4ujWAtFYv"},
        {"nonce": 31, "vector_b64": "aDW5NYM25CwOOdau2DBVNAKYoDIFqusx"},
    ],
    "encoding": {"dtype": "f16", "k_dim": 12, "endian": "le"},
}

SERVER_STARTUP_TIMEOUT_SEC = int(
    os.environ.get("POC_PROFILE_SERVER_STARTUP_TIMEOUT_SEC", "900")
)
SERVER_STARTUP_PROGRESS_SEC = int(
    os.environ.get("POC_PROFILE_SERVER_PROGRESS_SEC", "5")
)
BASE_PORT = 8766
DEFAULT_TP_SIZE = int(os.environ.get("POC_TP_SIZE", "4"))
DEFAULT_DTYPE = os.environ.get("POC_DTYPE", "float16")
DEFAULT_KV_CACHE_DTYPE = os.environ.get("POC_KV_CACHE_DTYPE", "auto")


def _resolve_project_root() -> Path:
    """Best-effort repo root resolution.

    When this script is executed from a shallow path (e.g. copied to
    `/e2e_poc_tiny.py` in a container), `Path(__file__).parents[3]` is invalid.
    We instead search upwards for a folder that looks like the vLLM repo root.
    """
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "vllm").is_dir():
            return candidate

    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists() and (cwd / "vllm").is_dir():
        return cwd

    return script_path.parent


PROJECT_ROOT = _resolve_project_root()

_SERVER_LOG_TAILS: dict[int, deque[str]] = {}
_BASE_URL_TO_SERVER_IDX: dict[str, int] = {}


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
        with contextlib.suppress(Exception):
            log_file.write(line)
        _SERVER_LOG_TAILS.setdefault(server_idx, deque(maxlen=120)).append(text)
    with contextlib.suppress(Exception):
        stream.close()


def _load_validation_payload() -> dict[str, Any]:
    return VALIDATION_SAMPLE


def _build_validation_map(payload: dict[str, Any]) -> dict[int, str]:
    artifacts = payload.get("artifacts") or []
    return {int(a["nonce"]): a["vector_b64"] for a in artifacts}


def _run_validation(
    computed_artifacts: list[dict[str, Any]],
    validation_map: dict[int, str],
    *,
    dist_threshold: float,
    p_mismatch: float,
    fraud_threshold: float,
    k_dim: int,
) -> dict[str, Any]:
    artifacts = [
        Artifact(nonce=int(a["nonce"]), vector_b64=str(a["vector_b64"]))
        for a in computed_artifacts
    ]
    stats = validate_artifacts(
        computed_artifacts=artifacts,
        expected_map=validation_map,
        dist_threshold=dist_threshold,
        p_mismatch=p_mismatch,
        fraud_threshold=fraud_threshold,
        k_dim=k_dim,
    )
    return {
        "n_total": stats.n_total,
        "n_mismatch": stats.n_mismatch,
        "p_value": stats.p_value,
        "fraud_detected": stats.fraud_detected,
        "mismatch_nonces": stats.mismatch_nonces,
    }


def _build_device_slice(server_idx: int, tp_size: int) -> str:
    start = server_idx * tp_size
    end = start + tp_size
    return ",".join(str(i) for i in range(start, end))


def _build_device_slices(tp_size: int, api_server_count: int) -> list[str]:
    visible_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    required_devices = tp_size * api_server_count
    if visible_device_count < required_devices:
        raise RuntimeError(
            "Not enough visible GPUs in this environment for requested topology: "
            f"need {required_devices} GPUs for api_server_count={api_server_count}, "
            f"tensor_parallel_size={tp_size}, but only {visible_device_count} visible. "
            "If running in Docker, check --gpus and container visibility. "
            "Or lower api_server_count/tp_size, or set POC_PROFILE_DEVICE_SLICES "
            "explicitly."
        )

    return [_build_device_slice(i, tp_size) for i in range(api_server_count)]


def _normalize_device_slice(value: str) -> str:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty device slice")
    for part in parts:
        if not part.isdigit():
            raise ValueError(f"invalid CUDA device id: {part!r}")
    return ",".join(parts)


def _parse_device_slices_env(value: str) -> list[str]:
    normalized = value.replace("|", ";")
    slices = [s.strip() for s in normalized.split(";") if s.strip()]
    if not slices:
        return []
    return [_normalize_device_slice(s) for s in slices]


def _resolve_topology(
    *,
    tp_size: int,
    api_server_count: int,
) -> tuple[int, int, list[str]]:
    """Resolve (tp_size, api_server_count, device_slices) for this environment.

    - If `POC_PROFILE_DEVICE_SLICES` is set, use it (semicolon-separated slices).
      Example: "0;1" for 2 single-GPU servers, or "0,1;2,3" for 2 servers w/ TP=2.
    - Otherwise, auto-downshift the requested topology to fit visible GPUs.
    """
    device_slices_env = os.environ.get("POC_PROFILE_DEVICE_SLICES", "").strip()
    if device_slices_env:
        slices = _parse_device_slices_env(device_slices_env)
        if not slices:
            raise RuntimeError("POC_PROFILE_DEVICE_SLICES is set but empty")

        inferred_tp_size = len(slices[0].split(","))
        if any(len(s.split(",")) != inferred_tp_size for s in slices):
            raise RuntimeError(
                "POC_PROFILE_DEVICE_SLICES must use the same number of devices "
                "per slice"
            )

        if tp_size != inferred_tp_size:
            print(
                "[INFO] overriding tp_size="
                f"{tp_size} -> {inferred_tp_size} from POC_PROFILE_DEVICE_SLICES"
            )
            tp_size = inferred_tp_size
        if api_server_count != len(slices):
            print(
                "[INFO] overriding api_server_count="
                f"{api_server_count} -> {len(slices)} from POC_PROFILE_DEVICE_SLICES"
            )
            api_server_count = len(slices)

        return tp_size, api_server_count, slices

    visible_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_device_count <= 0:
        raise RuntimeError(
            "No CUDA GPUs visible. Set CUDA_VISIBLE_DEVICES or run on a GPU machine."
        )

    if tp_size > visible_device_count:
        print(
            "[INFO] lowering tp_size="
            f"{tp_size} -> {visible_device_count} to fit visible GPUs"
        )
        tp_size = visible_device_count

    max_servers = max(1, visible_device_count // max(1, tp_size))
    if api_server_count > max_servers:
        print(
            "[INFO] lowering api_server_count="
            f"{api_server_count} -> {max_servers} to fit visible GPUs"
        )
        api_server_count = max_servers

    return tp_size, api_server_count, _build_device_slices(tp_size, api_server_count)


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
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _read_log_tail(log_path: Path, max_lines: int = 80) -> str:
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return "<unable to read log file>"
    if not lines:
        return "<log is empty>"
    return "".join(lines[-max_lines:]).rstrip()


def _extract_root_cause(log_path: Path, window: int = 120) -> str:
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
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
    dtype: str,
    kv_cache_dtype: str,
    log_path: Path,
    log_file: Any,
    start_delay_sec: int = 0,
) -> tuple[int, subprocess.Popen, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
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
        "1024",
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        dtype,
        "--kv-cache-dtype",
        kv_cache_dtype,
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
        raise

    return server_idx, proc, str(log_path)


def _run_forward_api(
    base_url: str,
    model: str,
    block_hash: str,
    public_key: str,
    nonces: list[int],
    seq_len: int,
    k_dim: int,
    batch_size: int,
) -> tuple[float, list[dict[str, Any]]]:
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
    last_error: str | None = None
    for _ in range(1, attempts + 1):
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


# ---------------------------------------------------------------------------
# Local callback server
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CallbackServer:
    """Thread-safe local HTTP server that collects /generated POST payloads."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.payloads: stdlib_queue.Queue[dict[str, Any]] = stdlib_queue.Queue()
        self._server: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        payloads = self.payloads

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    payload = _json.loads(body.decode("utf-8"))
                    payloads.put(payload)
                    self.send_response(200)
                except Exception:
                    self.send_response(400)
                self.end_headers()

            def log_message(self, fmt, *args):  # suppress access log noise
                pass

        self._server = http.server.HTTPServer(("127.0.0.1", self.port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()

    def get_next(self, timeout: float = 120.0) -> dict[str, Any] | None:
        try:
            return self.payloads.get(timeout=timeout)
        except stdlib_queue.Empty:
            return None


def profile_poc(
    dtype: str = DEFAULT_DTYPE,
    kv_cache_dtype: str = DEFAULT_KV_CACHE_DTYPE,
) -> None:
    profile_runs = 10
    dist_threshold = POC_PROFILE_DIST_THRESHOLD
    p_mismatch = POC_PROFILE_P_MISMATCH
    fraud_threshold = POC_PROFILE_FRAUD_THRESHOLD

    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    seq_len = 1024
    k_dim = 12
    batch_size = 32
    tp_size = DEFAULT_TP_SIZE
    api_server_count = 1
    max_model_len = 2048

    public_key = PUBLIC_KEY
    block_hash = BLOCK_HASH
    block_height = 2732723

    tp_size, api_server_count, device_slices = _resolve_topology(
        tp_size=tp_size,
        api_server_count=api_server_count,
    )

    ports = [BASE_PORT + i for i in range(api_server_count)]
    base_urls = [f"http://127.0.0.1:{port}" for port in ports]
    _BASE_URL_TO_SERVER_IDX.clear()
    _BASE_URL_TO_SERVER_IDX.update({base_urls[i]: i for i in range(api_server_count)})

    print("=" * 70)
    print("PoC legacy_poc test: /init/generate -> first callback -> /legacy_poc + validation")
    print(f"Model: {model}")
    print(f"TP size: {tp_size}")
    print(f"batch_size: {batch_size}")
    print(f"Profile runs (legacy_poc callbacks to collect): {profile_runs}")
    visible_cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Visible CUDA devices: {visible_cuda_devices}")
    print(f"Device slices: {device_slices}")
    print(f"max_model_len: {max_model_len}")
    print("model_args:")
    print(f"  --max-model-len {max_model_len}")
    print(f"  --dtype {dtype}")
    print(f"  --kv-cache-dtype {kv_cache_dtype}")
    print("  --enable-auto-tool-choice")
    print("  --tool-call-parser hermes")
    print("=" * 70)
    print(f"Using vllm from: {vllm.__file__}")
    print(f"Project root: {PROJECT_ROOT}")
    print(
        f"  dist_threshold={dist_threshold}, "
        f"p_mismatch={p_mismatch}, "
        f"fraud_threshold={fraud_threshold}"
    )

    start_delays = [0, 5]
    logs_dir = Path("logs/profile_poc")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_paths = [logs_dir / f"server_{i + 1}.log" for i in range(api_server_count)]

    with contextlib.ExitStack() as exit_stack:
        log_files = [
            exit_stack.enter_context(
                log_paths[i].open("w", buffering=1, encoding="utf-8")
            )
            for i in range(api_server_count)
        ]

        server_procs: list[subprocess.Popen] = []
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
                        dtype,
                        kv_cache_dtype,
                        log_paths[server_idx],
                        log_files[server_idx],
                        (
                            start_delays[server_idx]
                            if server_idx < len(start_delays)
                            else 0
                        ),
                    )
                    for server_idx in range(api_server_count)
                ]

                ready: list[subprocess.Popen | None] = [None] * api_server_count
                for future in as_completed(futures):
                    try:
                        server_idx, proc, log_path = future.result()
                    except Exception as exc:
                        print(f"\n[ERROR] Server startup failed: {exc}")
                        for idx in range(api_server_count):
                            diag_path = (
                                Path("logs/profile_poc") / f"server_{idx + 1}.log"
                            )
                            if diag_path.exists():
                                print(
                                    "\n[LOG DIAG] server "
                                    f"{idx + 1} root-cause excerpt ({diag_path}):\n"
                                    f"{_extract_root_cause(diag_path)}"
                                )
                        raise

                    ready[server_idx] = proc
                    print(
                        f"  API server ready: {server_idx + 1}/{api_server_count} "
                        f"(port={ports[server_idx]}, log={log_path})"
                    )

            for proc in ready:
                if proc is None:
                    raise RuntimeError("Server startup failed")
                server_procs.append(proc)

            print(f"Parallel server startup finished in {time.time() - start_t0:.1f}s")

            base_url = base_urls[0]

            # ── Local callback server ──────────────────────────────────────────
            cb_port = _find_free_port()
            cb_server = _CallbackServer(cb_port)
            cb_url = cb_server.start()
            print(f"\nCallback server listening on {cb_url}")

            poc_payload = {
                "block_hash": block_hash,
                "block_height": block_height,
                "public_key": public_key,
                "node_id": 0,
                "node_count": 1,
                "group_id": 0,
                "n_groups": 1,
                "batch_size": batch_size,
                "params": {"model": model, "seq_len": seq_len, "k_dim": k_dim},
                "url": cb_url,
            }

            validation_map = _build_validation_map(_load_validation_payload())
            nonces_in_sample = sorted(validation_map)

            def _validate_and_print(label: str, artifacts_raw: list[dict]) -> None:
                matching = [
                    a for a in artifacts_raw if int(a["nonce"]) in validation_map
                ]
                print(
                    f"  {label}: {len(artifacts_raw)} artifacts total, "
                    f"{len(matching)} match nonces "
                    f"{nonces_in_sample[0]}-{nonces_in_sample[-1]}"
                )
                if not matching:
                    print(f"  No matching nonces in {label} callback.")
                    return
                matched_dicts = [
                    {"nonce": int(a["nonce"]), "vector_b64": str(a["vector_b64"])}
                    for a in matching
                ]
                try:
                    val = _run_validation(
                        matched_dicts,
                        validation_map,
                        dist_threshold=dist_threshold,
                        p_mismatch=p_mismatch,
                        fraud_threshold=fraud_threshold,
                        k_dim=k_dim,
                    )
                except Exception as exc:
                    print(f"  [WARN] Validation error: {exc}")
                    return
                print(
                    f"  Validation ({label}): n_total={val['n_total']}, "
                    f"n_mismatch={val['n_mismatch']}, "
                    f"fraud={val['fraud_detected']}, "
                    f"p_value={val['p_value']:.6f}"
                )
                if val.get("mismatch_nonces"):
                    print(f"    mismatch_nonces={val['mismatch_nonces']}")

            # ── Step 1: /init/generate with callback ───────────────────────────
            print("\nCalling /init/generate (scheduler path)...")
            resp = requests.post(
                f"{base_url}/api/v1/pow/init/generate",
                json=poc_payload,
                timeout=30,
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"/init/generate failed: {resp.status_code} {resp.text}"
                )
            print("  /init/generate started.")

            # ── Step 2: Wait for first callback, validate ──────────────────────
            print(
                f"  Waiting for first callback from /init/generate "
                f"(will validate nonces {nonces_in_sample[0]}-{nonces_in_sample[-1]})..."
            )
            t0 = time.time()
            first_cb = cb_server.get_next(timeout=120.0)
            if first_cb is None:
                raise TimeoutError(
                    "No callback received from /init/generate in 120s"
                )
            elapsed_init = time.time() - t0
            print(f"  First callback received in {elapsed_init * 1000:.0f}ms.")
            _validate_and_print("/init/generate", first_cb.get("artifacts", []))

            # ── Step 3: Stop /init/generate ────────────────────────────────────
            requests.post(f"{base_url}/api/v1/pow/stop", timeout=10)
            print("  /init/generate stopped.")

            # Drain any queued init/generate callbacks before legacy_poc starts
            drained = 0
            while not cb_server.payloads.empty():
                cb_server.payloads.get_nowait()
                drained += 1
            if drained:
                print(f"  Drained {drained} leftover init/generate callbacks.")

            # ── Step 4: Start /legacy_poc with callback ────────────────────────
            print("\nStarting /legacy_poc...")
            resp = requests.post(
                f"{base_url}/api/v1/pow/legacy_poc",
                json=poc_payload,
                timeout=30,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"/legacy_poc failed: {resp.status_code} {resp.text}")
            print("  /legacy_poc started.")

            # ── Step 5: Wait for ONE legacy_poc callback, validate ────────────
            print(
                f"\nWaiting for first legacy_poc callback "
                f"(will validate nonces {nonces_in_sample[0]}-{nonces_in_sample[-1]})..."
            )

            t0 = time.time()
            cb_payload = cb_server.get_next(timeout=120.0)
            elapsed = time.time() - t0

            # ── Stop legacy_poc immediately after first callback ───────────────
            requests.post(f"{base_url}/api/v1/pow/stop", timeout=10)
            cb_server.stop()
            print(f"  legacy_poc stopped. First callback in {elapsed * 1000:.0f}ms.")

            if cb_payload is None:
                raise TimeoutError("No callback received from /legacy_poc in 120s")

            _validate_and_print("/legacy_poc", cb_payload.get("artifacts", []))

        finally:
            for proc in server_procs:
                _stop_process(proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile PoC through one or more OpenAI API servers."
    )
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=["auto", "float16", "bfloat16", "float32"],
        help="dtype passed to vllm.entrypoints.openai.api_server --dtype",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default=DEFAULT_KV_CACHE_DTYPE,
        choices=[
            "auto",
            "bfloat16",
            "fp8",
            "fp8_ds_mla",
            "fp8_e4m3",
            "fp8_e5m2",
            "fp8_inc",
        ],
        help="value passed to vllm.entrypoints.openai.api_server --kv-cache-dtype",
    )
    args = parser.parse_args()
    profile_poc(dtype=args.dtype, kv_cache_dtype=args.kv_cache_dtype)
