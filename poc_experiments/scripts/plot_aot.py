#!/usr/bin/env python3
"""Compare AOT=1 vs AOT=0 artifacts and render scatter + histogram of per-nonce L2."""
import argparse, base64, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def decode(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float32)


def load(path: str) -> dict[int, np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    return {int(a["nonce"]): decode(a["vector_b64"]) for a in data["artifacts"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aot1", required=True)
    p.add_argument("--aot0", required=True)
    p.add_argument("--out-png", required=True)
    p.add_argument("--out-csv", default=None)
    args = p.parse_args()

    a1 = load(args.aot1)
    a0 = load(args.aot0)
    common = sorted(set(a1) & set(a0))
    print(f"AOT=1: {len(a1)} artifacts, AOT=0: {len(a0)}, common: {len(common)}")

    dists, cos = [], []
    for n in common:
        v1, v0 = a1[n], a0[n]
        dists.append(float(np.linalg.norm(v1 - v0)))
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v0))
        cos.append(1.0 - float(np.dot(v1, v0)) / denom if denom > 0 else 0.0)
    dists = np.asarray(dists)
    cos = np.asarray(cos)

    print(f"L2  : mean={dists.mean():.4f}  median={np.median(dists):.4f}  "
          f"min={dists.min():.4f}  max={dists.max():.4f}  std={dists.std():.4f}")
    print(f"1-cos: mean={cos.mean():.6f}  median={np.median(cos):.6f}  "
          f"min={cos.min():.6f}  max={cos.max():.6f}")

    # Threshold check (matches POC_PROFILE_DIST_THRESHOLD default 0.4)
    thr = 0.4
    n_above = int((dists > thr).sum())
    print(f"L2 > {thr}: {n_above} / {len(dists)} ({100.0*n_above/max(len(dists),1):.1f}%)")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    ax = axes[0]
    ax.scatter(np.arange(len(common)), dists, s=12, alpha=0.6,
               label=f"L2(AOT=1, AOT=0)  (mean={dists.mean():.3f})")
    ax.axhline(thr, color="red", linestyle="--", linewidth=0.8,
               label=f"POC dist_threshold = {thr}")
    ax.set_title("Per-nonce L2 distance between AOT=1 and AOT=0 PoC artifacts")
    ax.set_xlabel("nonce index")
    ax.set_ylabel("L2 distance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(dists, bins=50, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.3)
    ax.axvline(thr, color="red", linestyle="--", linewidth=0.8, label=f"threshold={thr}")
    ax.set_xlabel("L2 distance")
    ax.set_ylabel("count")
    ax.set_title(f"Distribution (n={len(dists)})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=120)
    print(f"Saved plot to {args.out_png}")

    if args.out_csv:
        with open(args.out_csv, "w") as f:
            f.write("nonce,l2,one_minus_cos\n")
            for n, d, c in zip(common, dists, cos):
                f.write(f"{n},{d:.6f},{c:.8f}\n")
        print(f"Saved CSV to {args.out_csv}")


if __name__ == "__main__":
    main()
