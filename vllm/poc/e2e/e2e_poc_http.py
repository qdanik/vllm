#!/usr/bin/env python3
"""Start a vLLM server and run a PoC + inference workflow.

Flow:
1) Start server
2) Healthcheck for 10 seconds
3) Send inference every 1 second; after 5 inferences, call /api/v1/pow/init/generate
4) Wait 30 seconds and call /api/v1/pow/stop (keep sending inference)
5) Validate nonces via /api/v1/pow/generate (keep sending inference)
6) Stop all
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_TP_SIZE = int(os.environ.get("POC_TP_SIZE", "4"))
DEFAULT_TIMEOUT_S = int(os.environ.get("POC_HTTP_STARTUP_TIMEOUT_SEC", "300"))
DEFAULT_BLOCK_HASH = "8d148df1530d06a3412acd3deda4db16bae780eefdd160e081e6f878417de92a"
DEFAULT_PUBLIC_KEY = (
    "02e0f3b6b7f832ead7af2a235b9b27715a4d586b0fa108e735f0676a5086479225"
)
DEFAULT_BLOCK_HEIGHT = 2732723
DEFAULT_POC_SEQ_LEN = 16
DEFAULT_POC_K_DIM = 8
DEFAULT_POC_NONCES = "1,3,5,7,9"


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


def _start_server(args: argparse.Namespace) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("VLLM_USE_V1", "1")

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
    stop_event: threading.Event,
    counter: dict[str, int],
    lock: threading.Lock,
) -> None:
    idx = 0
    while not stop_event.is_set():
        start = time.time()
        try:
            if idx % 2 == 0:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are concise."},
                        {"role": "user", "content": "Ping from chat."},
                    ],
                    "max_tokens": 16,
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
                    "max_tokens": 16,
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
        sleep_s = max(0.0, 1.0 - elapsed)
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
    args = parser.parse_args()

    base_url = _client_base_url(args.host, args.port)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    server = _start_server(args)
    try:
        _wait_for_health(base_url, headers, args.timeout_s, server)
        _healthcheck_loop(base_url, headers, seconds=10)
        print("Starting inference loop...", flush=True)

        counter = {"count": 0}
        lock = threading.Lock()
        stop_event = threading.Event()
        worker = threading.Thread(
            target=_send_inference_loop,
            args=(base_url, headers, args.model, stop_event, counter, lock),
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
                print("Warning: did not reach 5 inferences within 120s; continuing.")
                break
            time.sleep(0.1)

        print("Calling /api/v1/pow/init/generate ...", flush=True)
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

        time.sleep(30.0)
        print("Calling /api/v1/pow/stop ...", flush=True)
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
