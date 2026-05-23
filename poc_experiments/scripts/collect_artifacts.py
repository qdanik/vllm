#!/usr/bin/env python3 -u
"""Collect PoC artifacts: 1000 nonces + 100 inference logprobs.

Uses callback-based PoC API:
  /api/v1/pow/init/generate  - start continuous generation
  /api/v1/pow/stop           - stop generation
  Callback server receives batches at /generated
"""
import argparse
import json
import time
import requests
import sys
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.stdout.reconfigure(line_buffering=True)

CALLBACK_PORT = 9998  # different from bench's 9999

# --- Callback server to receive nonce batches ---
_proof_batches = []
_proof_lock = threading.Lock()
_server_instance = None


class CallbackHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, d, s=200):
        self.send_response(s)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(d).encode())

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "OK"})
        elif self.path == "/stats":
            with _proof_lock:
                total = sum(
                    len(b.get("artifacts", b.get("nonces", [])))
                    for b in _proof_batches
                )
            self._json({"total_nonces": total, "batch_count": len(_proof_batches)})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(cl).decode() if cl > 0 else "{}"
        data = json.loads(body)
        if self.path == "/generated":
            with _proof_lock:
                _proof_batches.append(data)
            self._json({"message": "OK"})
        elif self.path == "/clear":
            with _proof_lock:
                _proof_batches.clear()
            self._json({"message": "Cleared"})
        else:
            self._json({"error": "not found"}, 404)


def start_callback_server():
    global _server_instance
    _server_instance = HTTPServer(("0.0.0.0", CALLBACK_PORT), CallbackHandler)
    t = threading.Thread(target=_server_instance.serve_forever, daemon=True)
    t.start()
    for _ in range(20):
        try:
            r = requests.get(f"http://localhost:{CALLBACK_PORT}/health", timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False


def get_total_nonces():
    with _proof_lock:
        total = 0
        for b in _proof_batches:
            total += len(b.get("artifacts", b.get("nonces", [])))
        return total


def get_all_artifacts():
    with _proof_lock:
        all_arts = []
        for b in _proof_batches:
            arts = b.get("artifacts", [])
            if arts:
                all_arts.extend(arts)
            else:
                # fallback: batch itself might be the artifact list
                nonces = b.get("nonces", [])
                if nonces:
                    all_arts.extend(nonces)
        return all_arts


# --- Main logic ---

def generate_nonces_callback(
    base_url, model_name, total=1000, batch_size=8,
    block_hash="artifact_collection_block_v1",
    public_key="artifact_collection_pk_v1",
):
    """Generate PoC nonces using callback-based API."""
    callback_url = f"http://127.0.0.1:{CALLBACK_PORT}"

    # Clear any previous batches
    with _proof_lock:
        _proof_batches.clear()

    payload = {
        "block_hash": block_hash,
        "block_height": 100,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "batch_size": batch_size,
        "url": callback_url,
        "params": {
            "model": model_name,
            "seq_len": 1024,
            "k_dim": 12,
        },
    }

    print(f"  Starting PoC generation (batch_size={batch_size})...", flush=True)
    r = requests.post(f"{base_url}/api/v1/inference/pow/init/generate", json=payload, timeout=60)
    r.raise_for_status()
    print(f"  Init response: {r.json()}", flush=True)

    # Wait for enough nonces
    t0 = time.time()
    while True:
        n = get_total_nonces()
        elapsed = time.time() - t0
        if n >= total:
            print(f"  [{n}/{total}] nonces collected in {elapsed:.1f}s", flush=True)
            break
        if elapsed > 600:  # 10 min timeout
            print(f"  TIMEOUT: only {n}/{total} nonces after {elapsed:.1f}s", flush=True)
            break
        if int(elapsed) % 10 == 0 and int(elapsed) > 0:
            rate = n / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{n}/{total}] {rate:.0f} nonces/min ({elapsed:.0f}s)", flush=True)
        time.sleep(1)

    # Stop generation
    try:
        requests.post(f"{base_url}/api/v1/inference/pow/stop", json={}, timeout=10)
    except:
        pass
    time.sleep(1)

    artifacts = get_all_artifacts()
    elapsed = time.time() - t0
    rate = len(artifacts) / elapsed * 60 if elapsed > 0 else 0

    return {
        "block_hash": block_hash,
        "public_key": public_key,
        "seq_len": 1024,
        "k_dim": 12,
        "total_nonces": len(artifacts),
        "artifacts": artifacts[:total],  # trim to requested count
        "generation_time_sec": elapsed,
        "nonces_per_min": rate,
    }


def generate_logprobs(base_url, model_name, count=100):
    """Generate inference completions with logprobs."""
    url = f"{base_url}/v1/chat/completions"
    results = []

    prompts = [
        "What is 2+2?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about the ocean.",
        "What is the capital of France?",
        "Translate 'hello world' to Japanese.",
        "What is the speed of light?",
        "Name three prime numbers.",
        "What year did World War II end?",
        "Define entropy in physics.",
        "What is pi to 5 decimal places?",
    ]

    for i in range(count):
        prompt = prompts[i % len(prompts)]
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 5,
        }
        try:
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            results.append({
                "prompt": prompt,
                "response": data,
                "index": i,
            })
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{count}] completions done", flush=True)
        except Exception as e:
            print(f"  ERROR completion {i}: {e}", flush=True)
            results.append({"prompt": prompt, "error": str(e), "index": i})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="vLLM base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--nonces", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="nonces per forward pass (32 matches 0.17.0 default)")
    parser.add_argument("--block-hash", default="artifact_collection_block_v1",
                        help="seeds generate_inputs; must match across runs you "
                             "want to L2-compare (cluster A used 'TEST_BLOCK')")
    parser.add_argument("--public-key", default="artifact_collection_pk_v1",
                        help="seeds generate_inputs alongside --block-hash "
                             "(cluster A used 'test_pub_keys')")
    parser.add_argument("--logprobs-count", type=int, default=100)
    parser.add_argument("--gpu", default="unknown", help="GPU name")
    parser.add_argument("--vllm-version", default="unknown", help="vLLM version")
    parser.add_argument("--startup-cmd", default="", help="vLLM startup command")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Start callback server
    print("Starting callback server...", flush=True)
    if not start_callback_server():
        print("ERROR: Failed to start callback server", flush=True)
        sys.exit(1)
    print(f"Callback server listening on port {CALLBACK_PORT}", flush=True)

    # Health check
    try:
        r = requests.get(f"{args.url}/health", timeout=5)
        print(f"vLLM health: {r.status_code}", flush=True)
    except Exception as e:
        print(f"Health check failed: {e}", flush=True)
        sys.exit(1)

    # Config
    config = {
        "gpu": args.gpu,
        "vllm_version": args.vllm_version,
        "model": args.model,
        "url": args.url,
        "batch_size": args.batch_size,
        "block_hash": args.block_hash,
        "public_key": args.public_key,
        "seq_len": 1024,
        "k_dim": 12,
        "startup_command": args.startup_cmd,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {args.output_dir}/config.json", flush=True)

    # Generate nonces
    print(f"\n=== Generating {args.nonces} PoC nonces ===", flush=True)
    nonces_data = generate_nonces_callback(
        args.url, args.model,
        total=args.nonces, batch_size=args.batch_size,
        block_hash=args.block_hash, public_key=args.public_key,
    )
    with open(os.path.join(args.output_dir, "nonces_1000.json"), "w") as f:
        json.dump(nonces_data, f)
    print(
        f"Nonces saved: {nonces_data['total_nonces']} "
        f"({nonces_data['nonces_per_min']:.0f}/min) "
        f"in {nonces_data['generation_time_sec']:.1f}s",
        flush=True,
    )

    # Generate logprobs
    print(f"\n=== Generating {args.logprobs_count} inference logprobs ===", flush=True)
    t0 = time.time()
    logprobs_data = generate_logprobs(
        args.url, args.model,
        count=args.logprobs_count,
    )
    t_logprobs = time.time() - t0
    with open(os.path.join(args.output_dir, "logprobs_100.json"), "w") as f:
        json.dump({"completions": logprobs_data, "time_sec": t_logprobs}, f)
    print(f"Logprobs saved: {len(logprobs_data)} in {t_logprobs:.1f}s", flush=True)

    print(f"\n=== Done: {args.output_dir} ===", flush=True)


if __name__ == "__main__":
    main()
