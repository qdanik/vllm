# 003 — `/legacy_poc`, Qwen3-0.6B on RTX

**Date:** 2026-05-22
**Status:** ✅ both modes succeeded, AOT toggle has zero numerical effect.

## What was tested

- Branch: `qdanik refactor/0.17.0` + env-var patch
- Endpoint: `POST /api/v1/pow/legacy_poc`
- Model: Qwen/Qwen3-0.6B (chosen because qwen3.* has no `assert input_ids is not None`, unlike qwen2.*)
- GPU: RTX PRO 6000 Blackwell (`shadeform@31.22.104.207`)
- 500 nonces, seq_len=1024, k_dim=12

## Result

| AOT=1 vs AOT=0 | mean L2 | max L2 | > threshold 0.4 |
|---|---:|---:|---:|
| 500 nonces | **0.000000** | 0.000000 | 0/500 |

**Bit-exact identical** between the two modes. Confirms theoretical expectation: `legacy_poc_runner.py` runs with `skip_compiled=True` (always eager), and `inputs_embeds` is always provided → the model's `forward` does `hidden_states = inputs_embeds` and the value of `input_ids` (None or zeros) is never read. See [`vllm/model_executor/models/qwen2.py:426-429`](../../vllm/model_executor/models/qwen2.py#L426-L429) for the dispatch.

## Files

- `qwen3_aot1.json`, `qwen3_aot0.json` — raw artifacts
- `qwen3_aot_comparison.png`, `qwen3_aot_per_nonce.csv` — comparison output (flat line at zero)
