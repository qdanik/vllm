#!/usr/bin/env python3
"""Start a vLLM server and run a PoC + inference workflow.

Flow:
1) Start server
2) Healthcheck for 10 seconds
3) Optionally send inference every 1 second; after 5 inferences, call /api/v1/pow/init/generate
4) Wait 30 seconds and call /api/v1/pow/stop
5) Validate nonces via /api/v1/pow/generate
6) Stop all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
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
DEFAULT_GENERATION_SECONDS = int(os.environ.get("POC_GENERATION_SECONDS", "30"))
DEFAULT_INFERENCE_INTERVAL_S = float(os.environ.get("POC_INFERENCE_INTERVAL_S", "1.0"))
DEFAULT_INFERENCE_MAX_TOKENS = int(os.environ.get("POC_INFERENCE_MAX_TOKENS", "16"))

_API_SERVER_FLAG_CACHE: set[str] | None = None


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
            _ = resp.read()  # Consume body to avoid connection reuse issues.
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
    except Exception as exc:  # noqa: BLE001
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
        try:
            ok, status = _health_status(f"{base_url}/health", headers=headers)
            if ok:
                return
            raise RuntimeError(status)
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(0.5)
    raise RuntimeError(f"Server did not become healthy: {last_err}")


def _client_base_url(host: str, port: int) -> str:
    client_host = host
    if host in {"0.0.0.0", "::", "[::]"}:
        client_host = "127.0.0.1"
    return f"http://{client_host}:{port}"


def _healthcheck_loop(base_url: str, headers: dict[str, str], seconds: int) -> None:
    print("Starting 10s healthcheck loop...", flush=True)
    for i in range(seconds):
        try:
            if _health_ok(f"{base_url}/health", headers=headers):
                print(f"Healthcheck {i + 1}/{seconds}: OK")
            else:
                print(f"Healthcheck {i + 1}/{seconds}: FAILED (non-200)")
        except Exception as exc:  # noqa: BLE001
            print(f"Healthcheck {i + 1}/{seconds}: FAILED ({exc})")
        time.sleep(1.0)


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
    except Exception:  # noqa: BLE001
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
    env.setdefault("VLLM_USE_V1", "1")
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


def _send_inference_loop(
    base_url: str,
    headers: dict[str, str],
    model: str,
    inference_interval_s: float,
    inference_max_tokens: int,
    phase_state: dict[str, str],
    inference_enabled: bool,
    inference_during_pow: bool,
    stop_event: threading.Event,
    counter: dict[str, int],
    lock: threading.Lock,
) -> None:
    idx = 0
    while not stop_event.is_set():
        current_phase = phase_state.get("phase", "warmup")
        if not inference_enabled:
            time.sleep(0.1)
            continue

        if not inference_during_pow and current_phase in {"pow", "stop", "validate"}:
            time.sleep(0.1)
            continue

        start = time.time()
        try:
            if idx % 2 == 0:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are concise."},
                        {"role": "user", "content": "Ping from chat."},
                    ],
                    "max_tokens": inference_max_tokens,
                    "temperature": 0.2,
                }
                _post_json(
                    f"{base_url}/v1/chat/completions",
                    payload,
                    headers,
                    timeout_s=10,
                )
            else:
                payload = {
                    "model": model,
                    "prompt": "Ping from completions.",
                    "max_tokens": inference_max_tokens,
                    "temperature": 0.2,
                }
                _post_json(
                    f"{base_url}/v1/completions",
                    payload,
                    headers,
                    timeout_s=10,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"Inference error: {exc}")

        with lock:
            counter["count"] += 1
        idx += 1

        elapsed = time.time() - start
        sleep_s = max(0.0, inference_interval_s - elapsed)
        time.sleep(sleep_s)


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
    parser.add_argument(
        "--generation-seconds",
        type=int,
        default=DEFAULT_GENERATION_SECONDS,
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
        "--inference-interval-s",
        type=float,
        default=DEFAULT_INFERENCE_INTERVAL_S,
    )
    parser.add_argument(
        "--inference-max-tokens",
        type=int,
        default=DEFAULT_INFERENCE_MAX_TOKENS,
    )
    parser.add_argument(
        "--disable-inference-loop",
        action="store_true",
        help=(
            "Disable background chat/completions loop to maximize pure "
            "PoCV2 throughput."
        ),
    )
    parser.add_argument(
        "--disable-inference-during-pow",
        action="store_true",
        help=(
            "Keep inference warmup before PoCV2 start, then pause inference "
            "for PoCV2 stop/validate phases."
        ),
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help=(
            "Run a pure PoCV2 flow without starting any background "
            "chat/completions requests or waiting for inference warmup."
        ),
    )
    parser.add_argument(
        "--h200-preset",
        action="store_true",
        help=(
            "Apply practical H200 defaults: throughput mode, O3, FLASH_ATTN, "
            "lighter background inference load."
        ),
    )
    parser.add_argument(
        "--additional-server-arg",
        action="append",
        default=[],
        help=(
            "Repeatable passthrough arg for api_server, e.g. "
            "--additional-server-arg=--max-num-seqs "
            "--additional-server-arg=64"
        ),
    )
    args = parser.parse_args()

    if args.h200_preset:
        if not args.performance_mode:
            args.performance_mode = "throughput"
        if args.optimization_level is None:
            args.optimization_level = 3
        if not args.attention_backend:
            args.attention_backend = "FLASH_ATTN"
        if args.inference_interval_s == DEFAULT_INFERENCE_INTERVAL_S:
            args.inference_interval_s = 1.5
        if args.inference_max_tokens == DEFAULT_INFERENCE_MAX_TOKENS:
            args.inference_max_tokens = 8

    if args.inference_interval_s <= 0:
        raise ValueError("--inference-interval-s must be > 0")
    if args.inference_max_tokens <= 0:
        raise ValueError("--inference-max-tokens must be > 0")

    if args.generate_only:
        args.disable_inference_loop = True
        args.disable_inference_during_pow = True

    base_url = _client_base_url(args.host, args.port)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    server = _start_server(args)
    try:
        _wait_for_health(base_url, headers, args.timeout_s, server)
        _healthcheck_loop(base_url, headers, seconds=10)
        inference_enabled = not args.disable_inference_loop

        counter = {"count": 0}
        lock = threading.Lock()
        phase_state = {"phase": "warmup"}
        stop_event = threading.Event()
        worker: threading.Thread | None = None
        if inference_enabled:
            print("Starting inference loop...", flush=True)
            worker = threading.Thread(
                target=_send_inference_loop,
                args=(
                    base_url,
                    headers,
                    args.model,
                    args.inference_interval_s,
                    args.inference_max_tokens,
                    phase_state,
                    inference_enabled,
                    not args.disable_inference_during_pow,
                    stop_event,
                    counter,
                    lock,
                ),
                daemon=True,
            )
            worker.start()

            wait_deadline = time.time() + 120
            while True:
                with lock:
                    current = counter["count"]
                if current >= 5:
                    break
                if time.time() > wait_deadline:
                    print(
                        "Warning: did not reach 5 inferences within 120s; continuing."
                    )
                    break
                time.sleep(0.1)
        else:
            print(
                "Inference loop disabled; starting pure PoCV2 generate flow.",
                flush=True,
            )

        print("Calling /api/v1/pow/init/generate ...", flush=True)
        phase_state["phase"] = "pow"
        init_payload = {
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
            "url": None,
        }
        init_resp = _post_json(
            f"{base_url}/api/v1/pow/init/generate",
            payload=init_payload,
            headers=headers,
            timeout_s=30,
        )
        print("/api/v1/pow/init/generate response:")
        print(json.dumps(init_resp, indent=2))

        time.sleep(float(args.generation_seconds))
        print("Calling /api/v1/pow/stop ...", flush=True)
        phase_state["phase"] = "stop"
        stop_resp = _post_json(
            f"{base_url}/api/v1/pow/stop",
            payload={},
            headers=headers,
            timeout_s=30,
        )
        print("/api/v1/pow/stop response:")
        print(json.dumps(stop_resp, indent=2))

        nonces = _parse_nonces(args.poc_nonces)
        generate_payload = {
            "block_hash": args.poc_block_hash,
            "block_height": args.poc_block_height,
            "public_key": args.poc_public_key,
            "node_id": args.poc_node_id,
            "node_count": args.poc_node_count,
            "nonces": nonces,
            "params": {
                "model": args.model,
                "seq_len": args.poc_seq_len,
                "k_dim": args.poc_k_dim,
            },
            "wait": True,
            "url": None,
            "validation": None,
            "stat_test": None,
        }
        print("Calling /api/v1/pow/generate ...", flush=True)
        phase_state["phase"] = "validate"
        generate_resp = _post_json(
            f"{base_url}/api/v1/pow/generate",
            payload=generate_payload,
            headers=headers,
            timeout_s=60,
        )
        print("/api/v1/pow/generate response:")
        print(json.dumps(generate_resp, indent=2))

        artifacts = (
            generate_resp.get("artifacts", [])
            if isinstance(generate_resp, dict)
            else []
        )
        validate_payload = {
            **generate_payload,
            "validation": {"artifacts": artifacts},
        }
        print("Calling /api/v1/pow/generate (validation) ...", flush=True)
        validate_resp = _post_json(
            f"{base_url}/api/v1/pow/generate",
            payload=validate_payload,
            headers=headers,
            timeout_s=60,
        )
        print("/api/v1/pow/generate (validation) response:")
        print(json.dumps(validate_resp, indent=2))

        stop_event.set()
        if worker is not None:
            worker.join(timeout=5)

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
