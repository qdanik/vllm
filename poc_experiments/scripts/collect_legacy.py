#!/usr/bin/env python3
"""Collect PoC artifacts via /api/v1/pow/legacy_poc callback flow.

This exercises legacy_poc_runner.execute_legacy_poc_forward_multi_batch,
which is where POC_USE_AOT_COMPILED_WORKAROUND has effect.
"""
import argparse, http.server, json, socket, sys, threading, time, urllib.request

BLOCK_HASH = "8d148df1530d06a3412acd3deda4db16bae780eefdd160e081e6f878417de92a"
PUBLIC_KEY = "02e0f3b6b7f832ead7af2a235b9b27715a4d586b0fa108e735f0676a5086479225"


class Collector:
    def __init__(self):
        self.lock = threading.Lock()
        self.artifacts: list[dict] = []
        self.batches = 0

    def add(self, batch):
        with self.lock:
            self.artifacts.extend(batch.get("artifacts", []))
            self.batches += 1

    @property
    def count(self):
        with self.lock:
            return len(self.artifacts)


def make_handler(collector):
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            try:
                collector.add(json.loads(body.decode()))
            except Exception as e:
                print(f"[cb] parse err: {e}", flush=True)
            self.send_response(200); self.end_headers()
        def log_message(self, *a):
            pass
    return H


def start_callback(collector):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0)); port = s.getsockname()[1]
    srv = http.server.HTTPServer(("0.0.0.0", port), make_handler(collector))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def post(url, payload, timeout=60):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--callback-host", default="172.17.0.1",
                   help="host that container can reach")
    p.add_argument("--model", required=True)
    p.add_argument("--target", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--k-dim", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-wait-s", type=int, default=600)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    collector = Collector()
    srv, port = start_callback(collector)
    cb_url = f"http://{args.callback_host}:{port}/callback"
    print(f"Callback server on port {port}, container will POST to {cb_url}", flush=True)

    payload = {
        "block_hash": BLOCK_HASH,
        "block_height": 2732723,
        "public_key": PUBLIC_KEY,
        "node_id": 0,
        "node_count": 1,
        "group_id": 0,
        "n_groups": 1,
        "batch_size": args.batch_size,
        "params": {
            "model": args.model,
            "seq_len": args.seq_len,
            "k_dim": args.k_dim,
        },
        "url": cb_url,
    }
    print(f"POSTing /api/v1/pow/legacy_poc, target={args.target} artifacts...", flush=True)
    t0 = time.time()
    resp = post(f"{args.base_url}/api/v1/pow/legacy_poc", payload)
    print(f"legacy_poc started: {resp}", flush=True)

    deadline = time.time() + args.max_wait_s
    last = 0
    while time.time() < deadline:
        c = collector.count
        if c >= args.target:
            break
        if c != last:
            print(f"  collected {c}/{args.target}  ({(c-last)/((time.time()-t0)/60.0):.0f}/min)", flush=True)
            last = c
        time.sleep(2)

    elapsed = time.time() - t0
    print(f"Got {collector.count} artifacts in {elapsed:.1f}s", flush=True)

    print("Stopping...", flush=True)
    post(f"{args.base_url}/api/v1/pow/stop", {})
    srv.shutdown()

    artifacts = collector.artifacts[: args.target]
    with open(args.out, "w") as f:
        json.dump({"artifacts": artifacts, "elapsed_s": elapsed}, f)
    print(f"Saved {len(artifacts)} artifacts to {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
