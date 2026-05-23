#!/usr/bin/env python3
"""All valid pair L2 comparisons (kaitakuai 0.15.1, Qwen2.5-7B).

Datasets split into TWO PoC-input clusters by ``(block_hash, public_key)``:
  * cluster A — A100, B200 → block_hash='TEST_BLOCK'
  * cluster B — H100, RTX  → block_hash='artifact_collection_block_v1'

PoC inputs (the synthetic embeddings) are seeded from these values, so cross-cluster
artifacts are computed over different inputs and cannot be compared meaningfully.
Only within-cluster pairs are rendered: C(4,2)=6 per cluster, 12 total.

Outputs:
  * all_pairs_histograms.png — 2×6 grid of per-nonce L2 histograms.
  * summary_heatmaps.png — two 4×4 matrices of mean L2 (one per cluster).
"""
import base64, json
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

ROOT = Path(__file__).parent
THR = 0.4

CLUSTERS = {
    "TEST_BLOCK (A100/B200)": [
        ("A100·AOT=1", "a100_aot1.json"),
        ("A100·AOT=0", "a100_aot0_eager.json"),
        ("B200·AOT=1", "b200_aot1.json"),
        ("B200·AOT=0", "b200_aot0_eager.json"),
    ],
    "artifact_collection_block_v1 (H100/RTX)": [
        ("H100·AOT=1", "h100_aot1.json"),
        ("H100·AOT=0", "h100_aot0_eager.json"),
        ("RTX·AOT=1",  "rtx_aot1.json"),
        ("RTX·AOT=0",  "rtx_aot0_eager.json"),
    ],
}


def decode(b64): return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float32)

def load(p):
    with open(ROOT / p) as f: data = json.load(f)
    return {int(a["nonce"]): decode(a["vector_b64"]) for a in data["artifacts"]}

def diff(a, b):
    common = sorted(set(a) & set(b))
    return np.array([float(np.linalg.norm(a[n] - b[n])) for n in common])

def category(la, lb):
    """within_gpu | cross_gpu_same_aot | cross_gpu_cross_aot"""
    ga, ma = la.split("·"); gb, mb = lb.split("·")
    if ga == gb: return "within_gpu"
    return "cross_gpu_same_aot" if ma == mb else "cross_gpu_cross_aot"


CAT_COLOR = {
    "within_gpu":           "#f2c94c",   # yellow — AOT effect (workaround neutrality)
    "cross_gpu_same_aot":   "#56a0d3",   # blue   — pure GPU drift
    "cross_gpu_cross_aot":  "#eb5757",   # red    — both vary
}
CAT_LABEL = {
    "within_gpu":           "within GPU (AOT effect)",
    "cross_gpu_same_aot":   "cross-GPU, same AOT (GPU drift)",
    "cross_gpu_cross_aot":  "cross-GPU + cross-AOT",
}


print("Loading datasets…")
data = {}
for ds_list in CLUSTERS.values():
    for label, path in ds_list:
        data[label] = load(path)

# Compute pairs and global x-axis
pair_results = []  # (cluster_idx, la, lb, l2)
for ci, (cname, ds_list) in enumerate(CLUSTERS.items()):
    for (la, _), (lb, _) in combinations(ds_list, 2):
        l2 = diff(data[la], data[lb])
        pair_results.append((ci, cname, la, lb, l2))

assert len(pair_results) == 12, len(pair_results)
xlim = max(arr.max() for _, _, _, _, arr in pair_results) * 1.05
bins = np.linspace(0, xlim, 50)

# ---- 1. histograms grid (2 rows × 6 cols) -----------------------------------
fig, axes = plt.subplots(2, 6, figsize=(20, 7), sharex=True)
for row, (ci, cname) in enumerate(((0, list(CLUSTERS.keys())[0]),
                                   (1, list(CLUSTERS.keys())[1]))):
    cluster_pairs = [r for r in pair_results if r[0] == ci]
    for col, (_, _, la, lb, l2) in enumerate(cluster_pairs):
        ax = axes[row, col]
        cat = category(la, lb)
        n_above = int((l2 > THR).sum())
        n_zero = int((l2 == 0.0).sum())
        ax.hist(l2, bins=bins, color=CAT_COLOR[cat], alpha=0.85,
                edgecolor="black", linewidth=0.25)
        ax.axvline(THR, color="red", ls="--", lw=0.7, alpha=0.8)
        ax.axvline(l2.mean(), color="black", ls=":", lw=1.0)
        ax.set_title(f"{la}  vs  {lb}", fontsize=10, fontweight="bold", pad=3)
        stats = (f"n={len(l2)}\n"
                 f"μ={l2.mean():.4f}\n"
                 f"med={np.median(l2):.4f}\n"
                 f"max={l2.max():.3f}\n"
                 f"std={l2.std():.4f}\n"
                 f">thr:{n_above}\n"
                 f"L2=0:{n_zero}")
        ax.text(0.97, 0.97, stats, transform=ax.transAxes, fontsize=8,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          alpha=0.88, lw=0.3))
        ax.tick_params(axis="both", labelsize=8)
        ax.set_xlim(0, xlim)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("count", fontsize=9)
        if row == 1:
            ax.set_xlabel("L2 distance", fontsize=9)
    # cluster row label on the far left
    axes[row, 0].text(-0.28, 0.5, cname, transform=axes[row, 0].transAxes,
                      rotation=90, fontsize=10, fontweight="bold",
                      ha="right", va="center")

handles = [plt.Rectangle((0, 0), 1, 1, fc=CAT_COLOR[c], ec="black", lw=0.4) for c in CAT_LABEL]
labels = [CAT_LABEL[c] for c in CAT_LABEL]
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, 1.01), frameon=True)
fig.suptitle(
    "kaitakuai 0.15.1  ·  Qwen2.5-7B  ·  1000 nonces  ·  PoC threshold = 0.4\n"
    "12 within-cluster pairs only (cross-cluster pairs invalidated by different block_hash/pk)",
    fontsize=12, y=1.06,
)
plt.tight_layout()
out = ROOT / "all_pairs_histograms.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved {out}")

# ---- 2. summary heatmaps (one per cluster) ----------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (cname, ds_list) in zip(axes2, CLUSTERS.items()):
    labels = [d[0] for d in ds_list]
    n = len(labels)
    m = np.zeros((n, n))
    for i, (la, _) in enumerate(ds_list):
        for j, (lb, _) in enumerate(ds_list):
            if i == j: continue
            m[i, j] = diff(data[la], data[lb]).mean()
    vmax = max(m.max(), 0.001)
    norm = mcolors.PowerNorm(gamma=0.6, vmin=0, vmax=vmax)
    im = ax.imshow(m, cmap="RdYlBu_r", norm=norm)
    for i in range(n):
        for j in range(n):
            v = m[i, j]
            color = "white" if v > vmax * 0.6 else "black"
            ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                    fontsize=10, color=color)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(cname, fontsize=11, pad=10)
    plt.colorbar(im, ax=ax, label="mean L2", shrink=0.85)
fig2.suptitle("Mean per-nonce L2 — within each PoC-input cluster",
              fontsize=12, y=1.02)
plt.tight_layout()
out2 = ROOT / "summary_heatmaps.png"
plt.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved {out2}")

# ---- 3. console summary -----------------------------------------------------
print("\n=== Per-pair L2 stats (12 valid pairs) ===")
print(f"{'cluster':<42}  {'pair':<28}  {'mean':>8}  {'median':>8}  {'max':>7}  {'>thr':>5}")
for _, cname, la, lb, l2 in pair_results:
    n_above = int((l2 > THR).sum())
    print(f"{cname:<42}  {la+' vs '+lb:<28}  {l2.mean():>8.4f}  "
          f"{np.median(l2):>8.4f}  {l2.max():>7.3f}  {n_above:>5}")
