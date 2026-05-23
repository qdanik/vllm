#!/usr/bin/env python3
"""Collect PoC artifacts from running vLLM server. Saves JSON to disk."""
import argparse, json, sys, time, urllib.request

BLOCK_HASH = "8d148df1530d06a3412acd3deda4db16bae780eefdd160e081e6f878417de92a"
PUBLIC_KEY = "02e0f3b6b7f832ead7af2a235b9b27715a4d586b0fa108e735f0676a5086479225"


def post(url, payload, timeout=900):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--nonces", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--k-dim", type=int, default=12)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    nonces = list(range(args.nonces))
    payload = {
        "block_hash": BLOCK_HASH,
        "block_height": 2732723,
        "public_key": PUBLIC_KEY,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {"model": args.model, "seq_len": args.seq_len, "k_dim": args.k_dim},
        "wait": True,
        "url": None,
        "validation": None,
        "stat_test": None,
    }
    print(f"POSTing /api/v1/pow/generate with {len(nonces)} nonces...", flush=True)
    t0 = time.time()
    resp = post(f"{args.base_url}/api/v1/pow/generate", payload)
    dt = time.time() - t0
    arts = resp.get("artifacts") or []
    print(f"Got {len(arts)} artifacts in {dt:.1f}s ({len(arts)/dt:.1f}/s)", flush=True)
    with open(args.out, "w") as f:
        json.dump({"artifacts": arts, "elapsed_s": dt}, f)
    print(f"Saved to {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
