#!/usr/bin/env python3
"""Dedicated AOT=1 vs AOT=0 comparison — same GPU, same model, only env var differs."""
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

def diff(a, b):
    common = sorted(set(a) & set(b))
    l2 = np.array([float(np.linalg.norm(a[n] - b[n])) for n in common])
    return common, l2


PAIRS = [
    ("RTX PRO 6000 Blackwell  —  Qwen3-0.6B  —  AOT=1 vs AOT=0",
     "qwen3_aot1.json", "qwen3_aot0.json", "tab:blue"),
    ("H100 PCIe  —  Qwen3-0.6B  —  AOT=1 vs AOT=0",
     "h100_qwen3_aot1.json", "h100_qwen3_aot0.json", "tab:green"),
]

fig, axes = plt.subplots(len(PAIRS), 1, figsize=(13, 3.5 * len(PAIRS)))
if len(PAIRS) == 1: axes = [axes]

for ax, (title, fa, fb, color) in zip(axes, PAIRS):
    a, b = load(fa), load(fb)
    nonces, l2 = diff(a, b)
    n_above = int((l2 > THR).sum())
    n_zero = int((l2 == 0.0).sum())
    label = (f"L2(AOT=1, AOT=0)  |  mean={l2.mean():.6f}  max={l2.max():.6f}  "
             f"std={l2.std():.6f}\n"
             f"identical (L2=0): {n_zero}/{len(l2)}  |  >{THR}: {n_above}/{len(l2)}")
    ax.scatter(np.arange(len(nonces)), l2, s=14, alpha=0.7, c=color, label=label)
    ax.axhline(THR, color="red", ls="--", lw=0.7, alpha=0.6, label=f"PoC threshold = {THR}")
    ax.set_ylim(-0.05, max(THR * 1.15, l2.max() * 1.1, 0.5))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("nonce index"); ax.set_ylabel("L2 distance")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    print(f"{title}\n  L2 mean={l2.mean():.8f}  max={l2.max():.8f}  zero: {n_zero}/{len(l2)}\n")

plt.suptitle("Effect of POC_USE_AOT_COMPILED_WORKAROUND on PoC artifacts (legacy_poc path)",
             fontsize=13, y=1.00)
plt.tight_layout()
out = ROOT / "aot_only_comparison.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
