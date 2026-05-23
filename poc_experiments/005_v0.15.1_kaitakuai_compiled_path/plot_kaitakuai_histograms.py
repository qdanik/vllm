#!/usr/bin/env python3
"""4 histograms of per-nonce L2 distance — kaitakuai branch (vLLM 0.15.1), Qwen2.5-7B.

NOTE: AOT=0 with compilation crashes on 0.15.1 with
  AttributeError: 'NoneType' object has no attribute 'size'
in torch._dynamo.utils.call_size (CUDA graph tracing of input_ids.size()).
This validates that kaitakuai's dummy_input_ids fix is necessary.

For comparison we use AOT=0 + --enforce-eager (no compilation) as the "no workaround" branch.
"""
import base64, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent / "kaitakuai_results"
THR = 0.4


def decode(b64): return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float32)

def load(p):
    with open(ROOT / p) as f: d = json.load(f)
    return {int(a["nonce"]): decode(a["vector_b64"]) for a in d["artifacts"]}

def diff(a, b):
    common = sorted(set(a) & set(b))
    return np.array([float(np.linalg.norm(a[n] - b[n])) for n in common])


rtx_aot1 = load("rtx_aot1.json")
rtx_aot0 = load("rtx_aot0_eager.json")
h100_aot1 = load("h100_aot1.json")
h100_aot0 = load("h100_aot0_eager.json")

PAIRS = [
    ("RTX·AOT=1  vs  H100·AOT=1\n(cross-GPU, both compiled+workaround)",
     rtx_aot1, h100_aot1, "tab:blue"),
    ("RTX·AOT=0(eager)  vs  H100·AOT=0(eager)\n(cross-GPU, both eager+no workaround)",
     rtx_aot0, h100_aot0, "tab:orange"),
    ("H100·AOT=1  vs  RTX·AOT=0(eager)\n(cross-GPU + cross-AOT)",
     h100_aot1, rtx_aot0, "tab:green"),
    ("H100·AOT=0(eager)  vs  RTX·AOT=1\n(cross-GPU + cross-AOT, reversed)",
     h100_aot0, rtx_aot1, "tab:red"),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.flatten()

all_max = max(max(diff(a, b)) for _, a, b, _ in PAIRS)
xlim = max(THR * 1.05, all_max * 1.1)
bins = np.linspace(0, xlim, 60)

for ax, (title, a, b, color) in zip(axes, PAIRS):
    l2 = diff(a, b)
    n_above = int((l2 > THR).sum())
    n_zero = int((l2 == 0.0).sum())
    label = (f"n={len(l2)}\n"
             f"mean={l2.mean():.4f}\n"
             f"median={np.median(l2):.4f}\n"
             f"max={l2.max():.4f}\n"
             f"std={l2.std():.4f}\n"
             f">{THR}: {n_above}\n"
             f"L2=0: {n_zero}")
    ax.hist(l2, bins=bins, color=color, alpha=0.75, edgecolor="black", linewidth=0.3)
    ax.axvline(THR, color="red", ls="--", lw=0.8, label=f"PoC threshold = {THR}")
    ax.axvline(l2.mean(), color="black", ls=":", lw=1.0, label=f"mean")
    ax.set_xlim(0, xlim)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("L2 distance"); ax.set_ylabel("count")
    ax.text(0.98, 0.97, label, transform=ax.transAxes, fontsize=9,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    print(f"{title.replace(chr(10), ' | ')}")
    print(f"  n={len(l2)}  mean={l2.mean():.6f}  median={np.median(l2):.6f}  "
          f"max={l2.max():.6f}  std={l2.std():.6f}  L2=0: {n_zero}  >thr: {n_above}\n")

plt.suptitle("kaitakuai/vllm  fix/poc-dummy-input-ids  —  Qwen2.5-7B  —  per-nonce L2 histograms",
             fontsize=13, y=1.00)
plt.tight_layout()
out = ROOT.parent / "kaitakuai_histograms.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
