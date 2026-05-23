# 001 — `/generate` path (false start)

**Date:** 2026-05-22
**Status:** ❌ wrong endpoint — kept for the record.

## What was tested

- Branch: `qdanik refactor/0.17.0`
- Endpoint: `POST /api/v1/pow/generate` (one-shot, scheduler-based via `compute_artifacts_pipelined`)
- Model: Qwen/Qwen2.5-7B
- GPU: RTX PRO 6000 Blackwell (`shadeform@31.22.104.207`)
- 500 nonces, seq_len=1024, k_dim=12
- Two modes: `POC_USE_AOT_COMPILED_WORKAROUND=1` vs `=0`

## Result

Bit-exact identical artifacts in both modes (L2 = 0 across all 500 nonces).

## Why it doesn't mean anything

The `/generate` endpoint goes through the **scheduler path** (`vllm/poc/server/compute.py::compute_artifacts_pipelined`), which does NOT call the file we patched (`vllm/poc/server/legacy_poc_runner.py`). The env var had no effect on the executed code, so identical output is a tautology.

The correct endpoint that exercises `legacy_poc_runner.py` is `/api/v1/pow/legacy_poc` — see experiments 002+.

## Files

- `aot1.json` (500), `aot0.json` (500) — raw artifacts
- `aot_comparison.png`, `aot_per_nonce.csv` — comparison output (all zeros)
