# 004 — Cross-GPU: H100 vs RTX, vLLM 0.17.0

**Date:** 2026-05-22
**Status:** ✅ Cross-GPU drift bounded; AOT toggle still neutral; one Qwen2.5 outlier.

## What was tested

- Branch: `qdanik refactor/0.17.0` + env-var patch
- Endpoint: `POST /api/v1/pow/legacy_poc`
- Models: Qwen/Qwen3-0.6B (both AOT modes), Qwen/Qwen2.5-7B (AOT=1 only — AOT=0 crashes per 002)
- GPUs: RTX PRO 6000 Blackwell (`shadeform@31.22.104.207`), H100 PCIe (`shadeform@69.19.137.240`)
- 500 nonces each, seq_len=1024, k_dim=12

## Result

| comparison | n | mean L2 | max L2 | > thr 0.4 |
|---|---:|---:|---:|---:|
| Qwen3-0.6B  RTX·AOT=1  vs  H100·AOT=1 | 500 | 0.0375 | 0.111 | 0 |
| Qwen3-0.6B  RTX·AOT=0  vs  H100·AOT=0 | 500 | 0.0375 | 0.111 | 0 |
| Qwen3-0.6B  H100·AOT=1  vs  AOT=0 (same GPU) | 500 | 0.000 | 0.000 | 0 |
| Qwen2.5-7B  RTX·AOT=1  vs  H100·AOT=1 | 500 | 0.0196 | **0.840** | **1** |

Cross-GPU drift on the eager path is ~0.04 L2 mean (Blackwell vs Hopper FP16 accumulation paths). Workaround does NOT alter this — same number both AOT modes. Qwen2.5-7B run has one nonce outlier above threshold (~0.84) worth investigating further.

## How to reproduce

Same Docker invocation as 002, swap `--model Qwen/Qwen3-0.6B`. Run on both servers, collect into matching filenames. Then run a plot script from this folder.

## Files

### Raw artifacts (H100)
- `h100_qwen3_aot1.json`, `h100_qwen3_aot0.json`, `h100_qwen25_aot1.json`

### Plots
- `aot_only_comparison.png` — AOT=1 vs AOT=0 panels for both GPUs (showing flat zero)
- `h100_vs_rtx_comparison.png` — 4-panel cross-GPU comparison incl. Qwen2.5
- `h100aot1_vs_rtxaot0.png` — cross-GPU + cross-AOT (extra sanity check)

### Plot scripts
- `plot_aot_only.py`, `plot_h100_vs_rtx.py`, `plot_h100aot1_vs_rtxaot0.py`

(These scripts hard-code filenames relative to the repo root — they expect the JSONs in the same folder when re-run, so paths may need a tweak after the reorg.)

## See also

- 003 — RTX-only Qwen3 baseline this is compared against
- 005 — equivalent comparison but on kaitakuai 0.15.1 (compiled path) — much bigger drift
