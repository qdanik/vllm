#!/usr/bin/env python3
"""Compare H100 and RTX PRO 6000 PoC artifacts across multiple configurations."""
import base64, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent


def decode(b64): return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float32)

def load(p):
    with open(ROOT / p) as f: data = json.load(f)
    return {int(a["nonce"]): decode(a["vector_b64"]) for a in data["artifacts"]}

def diff(a, b):
    common = sorted(set(a) & set(b))
    l2 = np.array([float(np.linalg.norm(a[n] - b[n])) for n in common])
    return common, l2


PAIRS = [
    ("Qwen3-0.6B  RTX  vs  H100  (AOT=1, with workaround)",
     "qwen3_aot1.json", "h100_qwen3_aot1.json", "tab:blue"),
    ("Qwen3-0.6B  RTX  vs  H100  (AOT=0, no workaround, input_ids=None)",
     "qwen3_aot0.json", "h100_qwen3_aot0.json", "tab:orange"),
    ("Qwen3-0.6B  H100  AOT=1  vs  AOT=0  (within-GPU, workaround effect)",
     "h100_qwen3_aot1.json", "h100_qwen3_aot0.json", "tab:green"),
    ("Qwen2.5-7B  RTX  vs  H100  (AOT=1; Qwen2 hard-requires workaround)",
     "aot1_legacy.json", "h100_qwen25_aot1.json", "tab:red"),
]

THR = 0.4
fig, axes = plt.subplots(len(PAIRS), 1, figsize=(13, 3.0 * len(PAIRS)))

for ax, (title, fa, fb, color) in zip(axes, PAIRS):
    try:
        a, b = load(fa), load(fb)
    except FileNotFoundError as e:
        ax.text(0.5, 0.5, f"missing: {e.filename}", ha="center", transform=ax.transAxes)
        continue
    nonces, l2 = diff(a, b)
    n_above = int((l2 > THR).sum())
    label = (f"L2 per-nonce  |  mean={l2.mean():.4f}  median={np.median(l2):.4f}  "
             f"max={l2.max():.4f}  std={l2.std():.4f}  |  >{THR}: {n_above}/{len(l2)}")
    ax.scatter(np.arange(len(nonces)), l2, s=10, alpha=0.6, c=color, label=label)
    ax.axhline(THR, color="red", ls="--", lw=0.7, label=f"PoC threshold = {THR}")
    ax.set_ylim(bottom=-max(0.02, l2.max() * 0.05) if l2.max() > 0 else -0.02,
                top=max(THR * 1.1, l2.max() * 1.1, 0.05))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("nonce index"); ax.set_ylabel("L2")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    print(f"{title}\n  L2 mean={l2.mean():.6f}  max={l2.max():.6f}  >thr: {n_above}/{len(l2)}\n")

plt.tight_layout()
out = ROOT / "h100_vs_rtx_comparison.png"
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
