# 002 — `/legacy_poc`, Qwen2.5-7B on RTX

**Date:** 2026-05-22
**Status:** ✅ AOT=1 succeeded · ❌ AOT=0 crashed (as designed — workaround mandatory).

## What was tested

- Branch: `qdanik refactor/0.17.0` (commit `e2c8839 feat: rename sprint to legacy poc`) + our env-var patch
- Endpoint: `POST /api/v1/pow/legacy_poc` (collective_rpc → `execute_legacy_poc_forward_multi_batch`)
- Model: Qwen/Qwen2.5-7B
- GPU: RTX PRO 6000 Blackwell (`shadeform@31.22.104.207`)
- 500 nonces, seq_len=1024, k_dim=12
- Two modes: `POC_USE_AOT_COMPILED_WORKAROUND=1` (placeholder zeros) vs `=0` (`input_ids=None`)

## Result

| mode | result |
|---|---|
| AOT=1 | 500 artifacts in 34.0s (~880/min) → `aot1_legacy.json` |
| AOT=0 | crashed with `AssertionError: assert input_ids is not None` at `vllm/model_executor/models/qwen2.py:583` → `aot0_legacy.json` is the empty placeholder (49 bytes, no artifacts) |

Only `qwen2.py` has this hard assert (grep across `vllm/model_executor/models/qwen*.py`). qwen3.* family accepts `None`. So on 0.17.0 the workaround is required for any Qwen2/Qwen2.5 model.

`--enforce-eager` does NOT bypass the assert (the assert is in the model `forward`, not in the compiled path).

## How to reproduce

```bash
ssh shadeform@31.22.104.207
docker run -d --gpus all --name poc-test \
  --add-host=host.docker.internal:host-gateway -p 8000:8000 \
  -e POC_USE_AOT_COMPILED_WORKAROUND=1 \
  -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  poc-aot:test \
  --model Qwen/Qwen2.5-7B --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --tensor-parallel-size 1
# wait /health, then:
python3 ~/collect_legacy.py --model Qwen/Qwen2.5-7B --target 500 \
  --out ~/poc-results/aot1_legacy.json
```

Swap `=1` → `=0` for the AOT=0 attempt; expect crash.

## Files

- `aot1_legacy.json` — 500 artifacts from AOT=1 run
- `aot0_legacy.json` — empty (crash payload)
