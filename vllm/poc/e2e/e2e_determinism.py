#!/usr/bin/env python3
"""PoC determinism test across different ``--max-num-batched-tokens`` values.

The test starts a vLLM OpenAI API server **twice** — first with the default
token budget, then with ``--max-num-batched-tokens 32768`` — and validates
that the PoC artifacts produced for the same 10 nonces are **bit-identical**,
using the *same* :func:`validate_artifacts` pipeline that the network uses.

Usage (inside a container with GPUs visible)::

    python -m vllm.poc.e2e.e2e_determinism          # auto-detect topology
    python -m vllm.poc.e2e.e2e_determinism --tp 4   # explicit TP size
    POC_PROFILE_DEVICE_SLICES="0,1,2,3" python -m vllm.poc.e2e.e2e_determinism

Environment variables (optional):
    POC_PROFILE_DEVICE_SLICES   Semicolon-separated CUDA device slices.
    POC_PROFILE_SERVER_STARTUP_TIMEOUT_SEC   Server health-check timeout.
    POC_DETERMINISM_MODEL       Model to use (default: Qwen3-235B FP8).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import requests

import vllm
from vllm.poc.consensus.encoding import decode_vector
from vllm.poc.constants import (
    DEFAULT_FRAUD_THRESHOLD,
    DEFAULT_K_DIM,
    DEFAULT_P_MISMATCH,
)
from vllm.poc.server.models import Artifact, ArtifactValidationStats
from vllm.poc.server.validation import validate_artifacts

# ---------------------------------------------------------------------------
# Buffered stdout/stderr
# ---------------------------------------------------------------------------

stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(line_buffering=True, write_through=True)
stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
if callable(stderr_reconfigure):
    stderr_reconfigure(line_buffering=True, write_through=True)

# ---------------------------------------------------------------------------
# Input parameters — identical for every run.
# ---------------------------------------------------------------------------

BLOCK_HASH = (
    "69B2F6FC38D2BE8181983AF17D7AFAC5B616EF95674F8762A4AB0EAA7F8032A5"
)
BLOCK_HEIGHT = 2489306
PUBLIC_KEY = (
    "02704a4bc225f08a2ef8c19439109bb73ff0833d9d87c78a8d072b85262ecaf074"
)
NODE_ID = 0
NODE_COUNT = 1
GROUP_ID = 1
N_GROUPS = 1
SEQ_LEN = 1024
NONCES = list(range(10))  # 0..9
BATCH_SIZE = 32

MODEL = os.environ.get(
    "POC_DETERMINISM_MODEL",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
)
DEFAULT_TP_SIZE = int(os.environ.get("POC_TP_SIZE", "4"))
DEFAULT_DTYPE = os.environ.get("POC_DTYPE", "float16")
DEFAULT_KV_CACHE_DTYPE = os.environ.get("POC_KV_CACHE_DTYPE", "auto")
DEFAULT_MAX_MODEL_LEN = int(os.environ.get("POC_MAX_MODEL_LEN", "4096"))

BASE_PORT = int(os.environ.get("POC_DETERMINISM_PORT", "8766"))
SERVER_STARTUP_TIMEOUT_SEC = int(
    os.environ.get("POC_PROFILE_SERVER_STARTUP_TIMEOUT_SEC", "1200")
)
SERVER_PROGRESS_SEC = int(
    os.environ.get("POC_PROFILE_SERVER_PROGRESS_SEC", "10")
)

TOKEN_BUDGETS: list[int | None] = [
    None,   # default (vLLM picks)
    32768,  # explicit 32 K
]

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------


def _resolve_project_root() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "vllm").is_dir():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists() and (cwd / "vllm").is_dir():
        return cwd
    return script_path.parent


PROJECT_ROOT = _resolve_project_root()

# ---------------------------------------------------------------------------
# Server log streaming
# ---------------------------------------------------------------------------

_SERVER_LOG_TAIL: deque[str] = deque(maxlen=200)


def _stream_logs(stream: Any, log_file: Any) -> None:
    for line in iter(stream.readline, ""):
        text = line.rstrip("\n")
        if not text:
            continue
        print(f"  [server] {text}")
        _SERVER_LOG_TAIL.append(text)
        with contextlib.suppress(Exception):
            log_file.write(line)
    with contextlib.suppress(Exception):
        stream.close()


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def _build_device_slice(server_idx: int, tp_size: int) -> str:
    start = server_idx * tp_size
    return ",".join(str(i) for i in range(start, start + tp_size))


def _resolve_device_slice(tp_size: int) -> str:
    env_val = os.environ.get("POC_PROFILE_DEVICE_SLICES", "").strip()
    if env_val:
        first = env_val.replace("|", ";").split(";")[0].strip()
        if first:
            return first
    return _build_device_slice(0, tp_size)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _start_server(
    *,
    model: str,
    tp_size: int,
    device_slice: str,
    port: int,
    max_model_len: int,
    dtype: str,
    kv_cache_dtype: str,
    max_num_batched_tokens: int | None,
    log_path: Path,
    log_file: Any,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = device_slice
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT) if not existing_pp else f"{PROJECT_ROOT}:{existing_pp}"
    )

    cmd = [
        sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--tensor-parallel-size", str(tp_size),
        "--max-num-seqs", "1024",
        "--max-model-len", str(max_model_len),
        "--dtype", dtype,
        "--kv-cache-dtype", kv_cache_dtype,
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]
    if max_num_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(max_num_batched_tokens)]

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
            target=_stream_logs,
            args=(proc.stdout, log_file),
            daemon=True,
        ).start()

    return proc


def _wait_for_health(
    proc: subprocess.Popen,
    port: int,
    timeout_sec: int,
    log_path: Path,
) -> None:
    started = time.time()
    next_progress = started + SERVER_PROGRESS_SEC
    while time.time() - started < timeout_sec:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server exited before healthy (port={port}, rc={proc.returncode}).\n"
                f"Log: {log_path}"
            )
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        now = time.time()
        if now >= next_progress:
            elapsed = int(now - started)
            print(f"  waiting for server on :{port} ({elapsed}s/{timeout_sec}s)...")
            next_progress = now + SERVER_PROGRESS_SEC
        time.sleep(1)
    raise TimeoutError(f"Server on port {port} not healthy in {timeout_sec}s")


def _stop_server(proc: subprocess.Popen) -> None:
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


# ---------------------------------------------------------------------------
# PoC generation via /api/v1/pow/generate  (reuses existing server route)
# ---------------------------------------------------------------------------


def _generate_nonces(
    base_url: str,
    model: str,
    nonces: list[int],
    *,
    seq_len: int = SEQ_LEN,
    k_dim: int = DEFAULT_K_DIM,
    batch_size: int = BATCH_SIZE,
) -> list[Artifact]:
    """Call /api/v1/pow/generate and return a list of :class:`Artifact`."""
    payload = {
        "block_hash": BLOCK_HASH,
        "block_height": BLOCK_HEIGHT,
        "public_key": PUBLIC_KEY,
        "node_id": NODE_ID,
        "node_count": NODE_COUNT,
        "nonces": nonces,
        "params": {
            "model": model,
            "seq_len": seq_len,
            "k_dim": k_dim,
        },
        "batch_size": batch_size,
        "wait": True,
    }
    resp = requests.post(
        f"{base_url}/api/v1/pow/generate",
        json=payload,
        timeout=600,
    )
    if resp.status_code != 200:
        detail = resp.text[:500]
        raise RuntimeError(
            f"HTTP {resp.status_code} from /api/v1/pow/generate: {detail}"
        )
    body = resp.json()
    raw_artifacts = body.get("artifacts")
    if raw_artifacts is None:
        raise RuntimeError(f"No artifacts in response: {body}")

    return [
        Artifact(nonce=int(a["nonce"]), vector_b64=str(a["vector_b64"]))
        for a in raw_artifacts
    ]


# ---------------------------------------------------------------------------
# Validation — reuses validate_artifacts / Artifact / decode_vector from
# vllm.poc.server.validation and vllm.poc.consensus.encoding
# ---------------------------------------------------------------------------


def _build_expected_map(artifacts: list[Artifact]) -> dict[int, str]:
    """Build an expected_map {nonce: vector_b64} from a list of Artifacts."""
    return {a.nonce: a.vector_b64 for a in artifacts}


def _cross_validate(
    run_a_artifacts: list[Artifact],
    run_b_artifacts: list[Artifact],
    *,
    label_a: str,
    label_b: str,
    k_dim: int,
    dist_threshold: float = 0.0,
) -> ArtifactValidationStats:
    """Validate run_b against run_a using :func:`validate_artifacts`.

    A dist_threshold of **0.0** means we require exact bit-identity.
    """
    expected_map = _build_expected_map(run_a_artifacts)

    print(f"\n  Validating {label_b} against {label_a} "
          f"(dist_threshold={dist_threshold}):")

    stats = validate_artifacts(
        computed_artifacts=run_b_artifacts,
        expected_map=expected_map,
        dist_threshold=dist_threshold,
        p_mismatch=DEFAULT_P_MISMATCH,
        fraud_threshold=DEFAULT_FRAUD_THRESHOLD,
        k_dim=k_dim,
    )
    return stats


def _print_mismatch_details(
    run_a_artifacts: list[Artifact],
    run_b_artifacts: list[Artifact],
    stats: ArtifactValidationStats,
    *,
    label_a: str,
    label_b: str,
) -> None:
    """Print detailed per-nonce diff for mismatched artifacts."""
    if not stats.mismatch_nonces:
        return

    map_a = {a.nonce: a for a in run_a_artifacts}
    map_b = {a.nonce: a for a in run_b_artifacts}

    for nonce in stats.mismatch_nonces:
        a = map_a.get(nonce)
        b = map_b.get(nonce)
        if a is None or b is None:
            print(f"    nonce {nonce}: missing in "
                  f"{'run_a' if a is None else 'run_b'}")
            continue

        vec_a = decode_vector(a.vector_b64)
        vec_b = decode_vector(b.vector_b64)
        dist = float(np.linalg.norm(vec_a - vec_b))
        print(f"    nonce {nonce}: L2 distance = {dist:.8f}")
        print(f"      {label_a}: {a.vector_b64}")
        print(f"      {label_b}: {b.vector_b64}")


# ---------------------------------------------------------------------------
# Run one configuration
# ---------------------------------------------------------------------------


def _run_one_config(
    *,
    model: str,
    tp_size: int,
    device_slice: str,
    port: int,
    max_model_len: int,
    dtype: str,
    kv_cache_dtype: str,
    max_num_batched_tokens: int | None,
    nonces: list[int],
    k_dim: int,
    label: str,
) -> list[Artifact]:
    """Start server, warmup, generate nonces, stop.  Return Artifact list."""
    budget_str = str(max_num_batched_tokens) if max_num_batched_tokens else "default"
    print(f"\n{'='*70}")
    print(f"Run: {label}  (max_num_batched_tokens={budget_str})")
    print(f"{'='*70}")

    logs_dir = Path("logs/determinism")
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace(" ", "_").lower()
    log_path = logs_dir / f"server_{safe_label}.log"

    with log_path.open("w", buffering=1, encoding="utf-8") as log_file:
        _SERVER_LOG_TAIL.clear()
        proc = _start_server(
            model=model,
            tp_size=tp_size,
            device_slice=device_slice,
            port=port,
            max_model_len=max_model_len,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            max_num_batched_tokens=max_num_batched_tokens,
            log_path=log_path,
            log_file=log_file,
        )
        try:
            print(
                f"  Server starting on :{port} "
                f"(CUDA_VISIBLE_DEVICES={device_slice})..."
            )
            _wait_for_health(proc, port, SERVER_STARTUP_TIMEOUT_SEC, log_path)
            print("  Server is healthy.")

            base_url = f"http://127.0.0.1:{port}"

            # Warmup — one batch to trigger compilations / cache warming.
            print("  Warmup batch (nonces 1000..1009)...")
            _generate_nonces(base_url, model, list(range(1000, 1010)), k_dim=k_dim)
            print("  Warmup done.")

            # Actual run.
            print(f"  Generating {len(nonces)} nonces: {nonces}")
            t0 = time.time()
            artifacts = _generate_nonces(base_url, model, nonces, k_dim=k_dim)
            elapsed = time.time() - t0
            print(f"  Generated {len(artifacts)} artifacts in {elapsed:.2f}s")

            # Print each artifact for traceability.
            for a in artifacts:
                print(f"    nonce={a.nonce}  vector={a.vector_b64}")

            return artifacts

        finally:
            print("  Stopping server...")
            _stop_server(proc)
            print("  Server stopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PoC determinism test: verify that different "
            "--max-num-batched-tokens produce bit-identical artifacts."
        ),
    )
    parser.add_argument("--tp", type=int, default=DEFAULT_TP_SIZE,
                        help="Tensor parallel size")
    parser.add_argument("--model", default=MODEL, help="HF model id")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE,
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--kv-cache-dtype", default=DEFAULT_KV_CACHE_DTYPE)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--port", type=int, default=BASE_PORT)
    parser.add_argument("--k-dim", type=int, default=DEFAULT_K_DIM)
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=None,
        help=(
            "Token budgets to compare. Use 'default' for no flag. "
            "Example: --budgets default 32768"
        ),
    )
    parser.add_argument(
        "--nonces",
        nargs="+",
        type=int,
        default=None,
        help="Nonces to generate (default: 0..9)",
    )
    parser.add_argument(
        "--dist-threshold",
        type=float,
        default=0.0,
        help=(
            "L2 distance threshold for cross-validation. "
            "0.0 = require exact bit-identity (default)."
        ),
    )
    args = parser.parse_args()

    nonces = args.nonces if args.nonces else NONCES
    k_dim = args.k_dim
    budgets: list[int | None]
    if args.budgets:
        budgets = [
            None if b.lower() == "default" else int(b) for b in args.budgets
        ]
    else:
        budgets = TOKEN_BUDGETS

    if len(budgets) < 2:
        print("Need at least 2 budgets to compare.  Use --budgets default 32768")
        sys.exit(1)

    tp_size = args.tp
    device_slice = _resolve_device_slice(tp_size)

    print("=" * 70)
    print("PoC Determinism Test")
    print("=" * 70)
    print(f"  Model:          {args.model}")
    print(f"  TP size:        {tp_size}")
    print(f"  Device slice:   {device_slice}")
    print(f"  Port:           {args.port}")
    print(f"  Max model len:  {args.max_model_len}")
    print(f"  Dtype:          {args.dtype}")
    print(f"  KV cache dtype: {args.kv_cache_dtype}")
    print(f"  k_dim:          {k_dim}")
    print(f"  Nonces:         {nonces}")
    print(f"  Token budgets:  {budgets}")
    print(f"  Dist threshold: {args.dist_threshold}")
    print(f"  vLLM:           {vllm.__file__}")
    print(f"  Project root:   {PROJECT_ROOT}")
    print()

    # -----------------------------------------------------------------------
    # Run each budget configuration sequentially (same GPUs, same port).
    # -----------------------------------------------------------------------
    run_results: dict[str, list[Artifact]] = {}
    for budget in budgets:
        label = f"budget_{budget}" if budget is not None else "budget_default"
        artifacts = _run_one_config(
            model=args.model,
            tp_size=tp_size,
            device_slice=device_slice,
            port=args.port,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            kv_cache_dtype=args.kv_cache_dtype,
            max_num_batched_tokens=budget,
            nonces=nonces,
            k_dim=k_dim,
            label=label,
        )
        run_results[label] = artifacts

    # -----------------------------------------------------------------------
    # Cross-validate all pairs using validate_artifacts from vllm.poc
    # -----------------------------------------------------------------------
    labels = list(run_results.keys())
    all_ok = True

    print(f"\n{'='*70}")
    print("DETERMINISM CROSS-VALIDATION")
    print(f"{'='*70}")

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            artifacts_a = run_results[la]
            artifacts_b = run_results[lb]

            stats = _cross_validate(
                artifacts_a,
                artifacts_b,
                label_a=la,
                label_b=lb,
                k_dim=k_dim,
                dist_threshold=args.dist_threshold,
            )

            print("\n  validate_artifacts result:")
            print(f"    n_total         = {stats.n_total}")
            print(f"    n_mismatch      = {stats.n_mismatch}")
            print(f"    mismatch_nonces = {stats.mismatch_nonces}")
            print(f"    p_value         = {stats.p_value:.8f}")
            print(f"    fraud_detected  = {stats.fraud_detected}")

            if stats.n_mismatch == 0:
                print(
                    f"\n  RESULT: OK — all {stats.n_total} nonces are "
                    f"bit-identical between {la} and {lb}."
                )
            else:
                print(
                    f"\n  RESULT: MISMATCH — {stats.n_mismatch}/{stats.n_total} "
                    f"nonces differ between {la} and {lb}."
                )
                _print_mismatch_details(
                    artifacts_a, artifacts_b, stats,
                    label_a=la, label_b=lb,
                )
                all_ok = False

    # -----------------------------------------------------------------------
    # Final verdict
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    if all_ok:
        print("PASS — all configurations produce deterministic artifacts.")
        sys.exit(0)
    else:
        print("FAIL — determinism broken between configurations.")
        sys.exit(1)


if __name__ == "__main__":
    main()
