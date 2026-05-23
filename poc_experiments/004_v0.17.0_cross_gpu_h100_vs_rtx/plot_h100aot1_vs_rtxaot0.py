#!/usr/bin/env python3
"""H100 AOT=1 vs RTX AOT=0 (cross-GPU, cross-AOT)."""
import base64, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
THR = 0.4


def decode(b64): return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float32)

def load(p):
    with open(ROOT / p) as f: data = json.load(f)
    return {int(a["nonce"]): decode(a["vector_b64"]) for a in data["artifacts"]}


h100_aot1 = load("h100_qwen3_aot1.json")
rtx_aot0 = load("qwen3_aot0.json")
common = sorted(set(h100_aot1) & set(rtx_aot0))
l2 = np.array([float(np.linalg.norm(h100_aot1[n] - rtx_aot0[n])) for n in common])
n_above = int((l2 > THR).sum())
n_zero = int((l2 == 0.0).sum())

print(f"H100 AOT=1 vs RTX AOT=0  (Qwen3-0.6B)")
print(f"  L2 mean={l2.mean():.6f}  median={np.median(l2):.6f}  max={l2.max():.6f}  std={l2.std():.6f}")
print(f"  identical (L2=0): {n_zero}/{len(l2)}  |  >{THR}: {n_above}/{len(l2)}")

fig, axes = plt.subplots(2, 1, figsize=(13, 7))

ax = axes[0]
label = (f"L2(H100·AOT=1, RTX·AOT=0)  |  mean={l2.mean():.4f}  max={l2.max():.4f}  "
         f"std={l2.std():.4f}\n>{THR}: {n_above}/{len(l2)}  |  identical: {n_zero}/{len(l2)}")
ax.scatter(np.arange(len(common)), l2, s=14, alpha=0.7, c="tab:purple", label=label)
ax.axhline(THR, color="red", ls="--", lw=0.7, alpha=0.6, label=f"PoC threshold = {THR}")
ax.set_ylim(-0.02, max(THR * 1.15, l2.max() * 1.1))
ax.set_title("H100 PCIe (AOT=1) vs RTX PRO 6000 Blackwell (AOT=0) — Qwen3-0.6B — cross-GPU + cross-AOT",
             fontsize=12, fontweight="bold")
ax.set_xlabel("nonce index"); ax.set_ylabel("L2 distance")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(l2, bins=60, alpha=0.8, color="tab:purple", edgecolor="black", linewidth=0.3)
ax.axvline(THR, color="red", ls="--", lw=0.7, label=f"threshold={THR}")
ax.axvline(l2.mean(), color="black", ls=":", lw=1.0, label=f"mean={l2.mean():.4f}")
ax.set_xlabel("L2 distance"); ax.set_ylabel("count")
ax.set_title(f"Distribution (n={len(l2)})", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = ROOT / "h100aot1_vs_rtxaot0.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
