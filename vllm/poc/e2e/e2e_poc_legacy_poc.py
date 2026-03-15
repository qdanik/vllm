#!/usr/bin/env python3
"""Start a vLLM server and run a legacy_poc workflow with callback collection.

Flow:
1) Start a local HTTP callback server on a free port to receive artifacts
2) Start vLLM server
3) Healthcheck until ready
4) POST /api/v1/pow/legacy_poc  → url=http://127.0.0.1:{port}/callback
5) Wait --legacy-poc-seconds, polling /api/v1/pow/legacy_poc/status
6) POST /api/v1/pow/stop
7) Validate collected artifacts via /api/v1/pow/generate
8) Stop all
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MODEL = os.environ.get("POC_MODEL", "Qwen/Qwen3-0.6B")
DEFAULT_TP_SIZE = int(os.environ.get("POC_TP_SIZE", "4"))
DEFAULT_TIMEOUT_S = int(os.environ.get("POC_HTTP_STARTUP_TIMEOUT_SEC", "300"))
DEFAULT_BLOCK_HASH = "8d148df1530d06a3412acd3deda4db16bae780eefdd160e081e6f878417de92a"
DEFAULT_PUBLIC_KEY = (
    "02e0f3b6b7f832ead7af2a235b9b27715a4d586b0fa108e735f0676a5086479225"
)
DEFAULT_BLOCK_HEIGHT = 2732723
DEFAULT_POC_SEQ_LEN = 1024
DEFAULT_POC_K_DIM = 12
DEFAULT_POC_NONCES = "1,3,5,7,9"
DEFAULT_LEGACY_POC_SECONDS = int(os.environ.get("POC_LEGACY_POC_SECONDS", "30"))
DEFAULT_STATUS_INTERVAL_S = float(os.environ.get("POC_STATUS_INTERVAL_S", "5.0"))

_API_SERVER_FLAG_CACHE: set[str] | None = None


# ---------------------------------------------------------------------------
# Callback HTTP server — receives ArtifactBatchSchema POSTs from legacy_poc
# ---------------------------------------------------------------------------


class _ArtifactCollector:
    """Thread-safe store for artifacts received via callback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._artifacts: list[dict] = []
        self._batches_received = 0

    def add_batch(self, batch: dict) -> None:
        arts = batch.get("artifacts", [])
        with self._lock:
            self._artifacts.extend(arts)
            self._batches_received += 1

    @property
    def artifacts(self) -> list[dict]:
        with self._lock:
            return list(self._artifacts)

    @property
    def batches_received(self) -> int:
        with self._lock:
            return self._batches_received


def _make_callback_handler(collector: _ArtifactCollector):
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                batch = json.loads(body.decode("utf-8"))
                collector.add_batch(batch)
            except Exception as exc:
                print(f"[Callback] parse error: {exc}", flush=True)
            self.send_response(200)
            self.end_headers()

        def log_message(self, fmt, *args):  # suppress access log noise
            pass

    return _Handler


def _start_callback_server(collector: _ArtifactCollector) -> tuple[http.server.HTTPServer, int]:
    """Start the callback HTTP server on a random free port. Returns (server, port)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    server = http.server.HTTPServer(("127.0.0.1", port), _make_callback_handler(collector))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def _post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout_s: int = 60,
) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, headers: dict[str, str], timeout_s: int = 30) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _health_ok(url: str, headers: dict[str, str], timeout_s: int = 5) -> bool:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status != 200:
                return False
            _ = resp.read()
            return True
    except Exception:
        return False


def _health_status(
    url: str,
    headers: dict[str, str],
    timeout_s: int = 5,
) -> tuple[bool, str]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = resp.status
            _ = resp.read()
            if status == 200:
                return True, "HTTP 200"
            return False, f"HTTP {status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, f"URL error: {exc.reason}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _wait_for_health(
    base_url: str,
    headers: dict[str, str],
    timeout_s: int,
    server: subprocess.Popen,
) -> None:
    deadline = time.time() + timeout_s
    last_err: str | None = None
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                "API server exited before becoming healthy "
                f"(exit_code={server.returncode})"
            )
        ok, status = _health_status(f"{base_url}/health", headers=headers)
        if ok:
            return
        last_err = status
        time.sleep(0.5)
    raise RuntimeError(f"Server did not become healthy: {last_err}")


def _client_base_url(host: str, port: int) -> str:
    client_host = host
    if host in {"0.0.0.0", "::", "[::]"}:
        client_host = "127.0.0.1"
    return f"http://{client_host}:{port}"


def _get_api_server_supported_flags() -> set[str]:
    global _API_SERVER_FLAG_CACHE
    if _API_SERVER_FLAG_CACHE is not None:
        return _API_SERVER_FLAG_CACHE
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        _API_SERVER_FLAG_CACHE = set(re.findall(r"--[a-z0-9][a-z0-9-]*", help_text))
    except Exception:
        _API_SERVER_FLAG_CACHE = set()
    return _API_SERVER_FLAG_CACHE


def _supports_api_server_flag(flag: str) -> bool:
    supported = _get_api_server_supported_flags()
    if not supported:
        return True
    return flag in supported


def _append_optional_flag(
    cmd: list[str],
    flag: str,
    value: str | None = None,
) -> None:
    if not _supports_api_server_flag(flag):
        print(f"Warning: api_server does not support {flag}; skipping.", flush=True)
        return
    cmd.append(flag)
    if value is not None:
        cmd.append(value)


def _start_server(args: argparse.Namespace) -> subprocess.Popen:
    env = os.environ.copy()
    if args.enable_cuda_compatibility:
        env["VLLM_ENABLE_CUDA_COMPATIBILITY"] = "1"
    if args.cuda_compatibility_path:
        env["VLLM_CUDA_COMPATIBILITY_PATH"] = args.cuda_compatibility_path

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
    ]

    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    if args.performance_mode:
        _append_optional_flag(cmd, "--performance-mode", args.performance_mode)

    if args.optimization_level is not None:
        _append_optional_flag(cmd, "--optimization-level", str(args.optimization_level))

    if args.attention_backend:
        _append_optional_flag(cmd, "--attention-backend", args.attention_backend)

    if args.disable_custom_all_reduce:
        _append_optional_flag(cmd, "--disable-custom-all-reduce")

    if args.enforce_eager:
        _append_optional_flag(cmd, "--enforce-eager")

    for extra_arg in args.additional_server_arg:
        cmd.append(extra_arg)

    return subprocess.Popen(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def _parse_nonces(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TP_SIZE)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--poc-block-hash", default=DEFAULT_BLOCK_HASH)
    parser.add_argument("--poc-public-key", default=DEFAULT_PUBLIC_KEY)
    parser.add_argument("--poc-block-height", type=int, default=DEFAULT_BLOCK_HEIGHT)
    parser.add_argument("--poc-seq-len", type=int, default=DEFAULT_POC_SEQ_LEN)
    parser.add_argument("--poc-k-dim", type=int, default=DEFAULT_POC_K_DIM)
    parser.add_argument("--poc-nonces", default=DEFAULT_POC_NONCES)
    parser.add_argument("--poc-node-id", type=int, default=0)
    parser.add_argument("--poc-node-count", type=int, default=1)
    parser.add_argument("--poc-batch-size", type=int, default=None)
    parser.add_argument(
        "--legacy-poc-seconds", type=int, default=DEFAULT_LEGACY_POC_SECONDS
    )
    parser.add_argument(
        "--status-interval-s", type=float, default=DEFAULT_STATUS_INTERVAL_S
    )
    parser.add_argument(
        "--performance-mode",
        choices=["balanced", "interactivity", "throughput"],
        default="",
    )
    parser.add_argument("--attention-backend", default="")
    parser.add_argument("--optimization-level", type=int, default=None)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-cuda-compatibility", action="store_true")
    parser.add_argument("--cuda-compatibility-path", default="")
    parser.add_argument(
        "--h200-preset",
        action="store_true",
        help="Apply H200 defaults: throughput mode, O3, FLASH_ATTN.",
    )
    parser.add_argument(
        "--additional-server-arg",
        action="append",
        default=[],
        help="Repeatable passthrough arg for api_server.",
    )
    args = parser.parse_args()

    if args.h200_preset:
        if not args.performance_mode:
            args.performance_mode = "throughput"
        if args.optimization_level is None:
            args.optimization_level = 3
        if not args.attention_backend:
            args.attention_backend = "FLASH_ATTN"

    base_url = _client_base_url(args.host, args.port)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    server = _start_server(args)
    try:
        _wait_for_health(base_url, headers, args.timeout_s, server)
        print("Server healthy.", flush=True)

        # ── Start local callback server ────────────────────────────────────
        collector = _ArtifactCollector()
        callback_srv, callback_port = _start_callback_server(collector)
        callback_url = f"http://127.0.0.1:{callback_port}/callback"
        print(f"Callback server listening on {callback_url}", flush=True)

        # ── Start legacy_poc ──────────────────────────────────────────────
        print("Calling /api/v1/pow/legacy_poc ...", flush=True)
        legacy_poc_payload: dict = {
            "block_hash": args.poc_block_hash,
            "block_height": args.poc_block_height,
            "public_key": args.poc_public_key,
            "node_id": args.poc_node_id,
            "node_count": args.poc_node_count,
            "group_id": 0,
            "n_groups": 1,
            "params": {
                "model": args.model,
                "seq_len": args.poc_seq_len,
                "k_dim": args.poc_k_dim,
            },
            "url": callback_url,
        }
        if args.poc_batch_size is not None:
            legacy_poc_payload["batch_size"] = args.poc_batch_size

        legacy_poc_resp = _post_json(
            f"{base_url}/api/v1/pow/legacy_poc",
            payload=legacy_poc_payload,
            headers=headers,
            timeout_s=30,
        )
        print("/api/v1/pow/legacy_poc response:")
        print(json.dumps(legacy_poc_resp, indent=2))

        # ── Poll status during legacy_poc ─────────────────────────────────
        deadline = time.time() + args.legacy_poc_seconds
        last_total = 0
        last_poll = time.time()

        while time.time() < deadline:
            now = time.time()
            if now - last_poll >= args.status_interval_s:
                try:
                    status = _get_json(
                        f"{base_url}/api/v1/pow/legacy_poc/status",
                        headers=headers,
                    )
                    total = (status.get("stats") or {}).get("total_processed", 0)
                    delta = total - last_total
                    elapsed_poll = now - last_poll
                    rate = (delta / (elapsed_poll / 60.0)) if elapsed_poll > 0 else 0.0
                    print(
                        f"[legacy_poc status] total={total}  Δ={delta}  "
                        f"rate={rate:.0f}/min  "
                        f"callbacks_received={collector.batches_received}  "
                        f"artifacts_collected={len(collector.artifacts)}",
                        flush=True,
                    )
                    last_total = total
                    last_poll = now
                except Exception as exc:
                    print(f"Status poll error: {exc}", flush=True)
            time.sleep(0.5)

        # ── Stop legacy_poc ───────────────────────────────────────────────
        print("Calling /api/v1/pow/stop ...", flush=True)
        stop_resp = _post_json(
            f"{base_url}/api/v1/pow/stop",
            payload={},
            headers=headers,
            timeout_s=30,
        )
        print("/api/v1/pow/stop response:")
        print(json.dumps(stop_resp, indent=2))

        # ── Final status ──────────────────────────────────────────────────
        try:
            final_status = _get_json(
                f"{base_url}/api/v1/pow/legacy_poc/status",
                headers=headers,
            )
            total = final_status.get("total_processed", 0)
            elapsed_min = args.legacy_poc_seconds / 60.0
            rate = total / elapsed_min if elapsed_min > 0 else 0.0
            print(
                f"legacy_poc finished: {total} nonces in {args.legacy_poc_seconds}s "
                f"({rate:.0f}/min)  "
                f"artifacts_via_callback={len(collector.artifacts)}",
                flush=True,
            )
        except Exception as exc:
            print(f"Final status error: {exc}", flush=True)

        callback_srv.shutdown()

        # ── Validate artifacts collected via callback ─────────────────────
        collected = collector.artifacts[:256]
        if not collected:
            print(
                "No artifacts received via callback — skipping validation.",
                flush=True,
            )
            return 0

        print(
            f"Validating first {len(collected)} artifacts (of {len(collector.artifacts)} received) ...",
            flush=True,
        )
        validate_payload = {
            "block_hash": args.poc_block_hash,
            "block_height": args.poc_block_height,
            "public_key": args.poc_public_key,
            "node_id": args.poc_node_id,
            "node_count": args.poc_node_count,
            "nonces": [a["nonce"] for a in collected],
            "params": {
                "model": args.model,
                "seq_len": args.poc_seq_len,
                "k_dim": args.poc_k_dim,
            },
            "wait": True,
            "url": None,
            "validation": {"artifacts": collected},
            "stat_test": None,
        }
        validate_resp = _post_json(
            f"{base_url}/api/v1/pow/generate",
            payload=validate_payload,
            headers=headers,
            timeout_s=120,
        )
        print("/api/v1/pow/generate (validation) response:")
        print(json.dumps(validate_resp, indent=2))

        return 0

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        print(f"HTTP error: {exc.code} {exc.reason}\n{body}")
        return 2
    finally:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
