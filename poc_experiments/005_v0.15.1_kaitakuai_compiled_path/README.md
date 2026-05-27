# 005 — kaitakuai 0.15.1: compiled PoC path

**Date:** 2026-05-22 / 23 (RTX, H100); 23 (A100, B200 added)
**Status:** ✅ 12 valid pairs analysed across 4 GPUs × 2 AOT modes. 16 cross-cluster pairs invalidated — see caveat below.

## ⚠️ Data caveat — two PoC-input clusters AND two models

The 8 datasets in this folder were collected with **two different `block_hash` / `public_key` pairs** AND **two different models**. Datasets with different seeds OR different models produce conceptually different artifacts and cannot be compared across the divide.

| cluster | block_hash | public_key | model | datasets |
|---|---|---|---|---|
| A | `TEST_BLOCK` | `test_pub_keys` | **Qwen3-235B-A22B-Instruct-2507-FP8** | `a100_aot{0,1}.json`, `b200_aot{0,1}.json` |
| B | `artifact_collection_block_v1` | `artifact_collection_pk_v1` | **Qwen2.5-7B** | `h100_aot{0,1}.json`, `rtx_aot{0,1}.json` |

The model identity for cluster A was confirmed retroactively (2026-05-23) by an independent B200 run on the same `(TEST_BLOCK, test_pub_keys)` seed pair with Qwen3-235B-A22B-Instruct-2507-FP8: mean L2 vs `b200_aot0_eager.json` = 0.085 (within the same-model cross-run eager envelope). If the cluster A datasets had been Qwen2.5-7B, the new run would have shown L2 ≈ 1.4 (independent uniform-on-sphere distributions).

Comparisons across the cluster boundary (A↔B) give mean L2 ≈ 1.4 with 98% above threshold — that's now over-determined: both the inputs and the model differ. **Only within-cluster pairs (12 of 28) are reported.**

If we need a full 8×8 matrix, re-run one cluster against the other's `(block_hash, pk, model)` triple — not just the seeds.

## What was tested

- Branch: `kaitakuai fix/poc-dummy-input-ids` (commit `48c09f5`, vLLM 0.15.1 base) + our env-var patch
- Endpoint: `POST /api/v1/pow/init/generate` (callback-based continuous loop → `engine_client.poc_request("generate_artifacts")` → `execute_poc_forward` in `vllm/poc/poc_model_runner.py`)
- Models:
  - **Cluster B (H100, RTX)** — `Qwen/Qwen2.5-7B`. This is the run we executed end-to-end on the rented boxes.
  - **Cluster A (A100, B200)** — `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`. Added later, identified retroactively via the 2026-05-23 cross-check.
- GPUs: RTX PRO 6000 Blackwell, H100 PCIe (cluster B); A100 80GB, B200 (cluster A, added 2026-05-23)
- 1000 nonces per run (matching kaitakuai's `collect_artifacts.py` default)
- Used kaitakuai's `collect_artifacts.py` after `sed -i 's|/api/v1/inference/pow|/api/v1/pow|g'` (their script targets a reverse-proxy prefix that our containers don't expose)

## Key difference from 0.17.0

Their `poc_model_runner.py` doesn't set `skip_compiled` per-call → PoC runs through the **compiled** model. This makes the workaround structurally necessary: `input_ids=None` crashes inside CUDA graph capture (`AttributeError: 'NoneType' object has no attribute 'size'` in `torch._dynamo.utils.call_size`) — exactly the bug kaitakuai's `dummy_input_ids` fix addresses.

Since pure AOT=0 crashes, the "no workaround" baseline was collected with `--enforce-eager` instead (disables compilation entirely).

## Result (12 valid pairs)

### Cluster A — TEST_BLOCK (A100, B200)

| pair | mean L2 | median | max | > thr 0.4 |
|---|---:|---:|---:|---:|
| A100·AOT=1 vs A100·AOT=0 (within A100) | **1.2968** | 1.3299 | 1.96 | **982/1000** |
| A100·AOT=1 vs B200·AOT=1 (cross-GPU, compiled) | **0.1368** | 0.1303 | 0.43 | **3/1000** |
| A100·AOT=0 vs B200·AOT=0 (cross-GPU, eager)    | **0.0862** | 0.0784 | 0.48 | 1/1000 |
| B200·AOT=1 vs B200·AOT=0 (within B200) | 1.3164 | 1.3507 | 1.96 | 984/1000 |
| A100·AOT=1 vs B200·AOT=0 (cross-AOT, cross-GPU) | 1.2978 | 1.3275 | 1.96 | 982/1000 |
| A100·AOT=0 vs B200·AOT=1 (cross-AOT, cross-GPU) | 1.3156 | 1.3547 | 1.97 | 984/1000 |

### Cluster B — artifact_collection_block_v1 (H100, RTX)

| pair | mean L2 | median | max | > thr 0.4 |
|---|---:|---:|---:|---:|
| H100·AOT=1 vs H100·AOT=0 (within H100) | 1.3338 | 1.3681 | 1.99 | 986/1000 |
| H100·AOT=1 vs RTX·AOT=1 (cross-GPU, compiled) | **0.2162** | 0.1958 | 1.53 | **53/1000** |
| H100·AOT=0 vs RTX·AOT=0 (cross-GPU, eager)    | **0.0235** | 0.0184 | 0.55 | 1/1000 |
| RTX·AOT=1 vs RTX·AOT=0 (within RTX) | 1.3364 | 1.3728 | 1.99 | 986/1000 |
| H100·AOT=1 vs RTX·AOT=0 (cross-AOT, cross-GPU) | 1.3345 | 1.3704 | 1.99 | 986/1000 |
| H100·AOT=0 vs RTX·AOT=1 (cross-AOT, cross-GPU) | 1.3356 | 1.3710 | 1.99 | 986/1000 |

### Findings

1. **Eager mode is bit-stable cross-GPU within each cluster.** A100↔B200 (Qwen3-235B): 0.086 mean L2, 1/1000 over thr. H100↔RTX (Qwen2.5-7B): 0.024 mean L2, 1/1000 over thr. Both far under threshold 0.4. The 3.5× gap between the two clusters is likely the model — Qwen3-235B has ~94 layers + MoE routing vs Qwen2.5-7B's 28 layers, so more numerical accumulation per forward.
2. **Compiled mode introduces real cross-GPU drift, scale depends on the GPU pair.** A100↔B200: 0.137 mean L2, 3 outliers over thr. H100↔RTX: 0.216 mean L2, 53 outliers (5% false-positive fraud rate). Compilation paths differ in numerical behaviour between same-architecture GPUs.
3. **Compiled vs eager are completely different code paths.** Within-GPU AOT=1 vs AOT=0: mean L2 ≈ 1.3, 98%+ over threshold. torch.compile reorders ops sufficiently that artifacts have no useful overlap.
4. **0.17.0's `skip_compiled=has_poc` is the right design** — keep the model compiled for normal inference, but route PoC through eager so artifacts stay GPU-portable.

## How to reproduce (cluster B — Qwen2.5-7B)

The bash below reproduces cluster B (H100 / RTX). For cluster A (A100 / B200) swap the model to `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` and the input seeds to `--block-hash TEST_BLOCK --public-key test_pub_keys` (using the post-006 `collect_artifacts.py`); also expect more VRAM and longer model load.

Disk + container hygiene matters — base image is `vllm/vllm-openai:v0.15.1` (different from 0.17.0). On H100 had to `docker system prune -a -f` before building, ~12 GB base + ~30 GB built image + ~14 GB Qwen HF cache.

```bash
# on the server
git clone --depth 1 -b fix/poc-dummy-input-ids https://github.com/kaitakuai/vllm.git ~/vllm-repo
cd ~/vllm-repo && git apply ~/kaitakuai_aot_env.patch
docker build -f Dockerfile.quick -t poc-aot:test .

sed -i 's|/api/v1/inference/pow|/api/v1/pow|g' ~/collect_artifacts.py

# AOT=1 (compiled+workaround)
docker run -d --gpus all --name poc-test --net=host \
  -e POC_USE_AOT_COMPILED_WORKAROUND=1 -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  poc-aot:test --model Qwen/Qwen2.5-7B --host 0.0.0.0 --port 8000 --max-model-len 4096
# wait /health, then
python3 ~/collect_artifacts.py --url http://127.0.0.1:8000 --model Qwen/Qwen2.5-7B \
  --output-dir ~/poc-results/rtx_aot1 --nonces 1000 --logprobs-count 0

# AOT=0 (eager) — REQUIRES --enforce-eager, otherwise CUDA-graph capture crashes
docker run -d --gpus all --name poc-test --net=host \
  -e POC_USE_AOT_COMPILED_WORKAROUND=0 -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  poc-aot:test --model Qwen/Qwen2.5-7B --host 0.0.0.0 --port 8000 --max-model-len 4096 \
  --enforce-eager
```

## Files

### Raw artifacts (1000 nonces each)
- Cluster A — `Qwen3-235B-A22B-Instruct-2507-FP8`, seeds `(TEST_BLOCK, test_pub_keys)`: `a100_aot1.json`, `a100_aot0_eager.json`, `b200_aot1.json`, `b200_aot0_eager.json`
- Cluster B — `Qwen2.5-7B`, seeds `(artifact_collection_block_v1, artifact_collection_pk_v1)`: `h100_aot1.json`, `h100_aot0_eager.json`, `rtx_aot1.json`, `rtx_aot0_eager.json`

### Plots
- `all_pairs_histograms.png` — 2×6 grid of 12 within-cluster L2 histograms
- `summary_heatmaps.png` — two 4×4 mean-L2 heatmaps (one per cluster)
- `kaitakuai_histograms.png` — earlier 4-panel chart (Cluster B only, kept for reference)

### Plot scripts
- `plot_all_pairs.py` — current 12-pair analysis
- `plot_kaitakuai_histograms.py` — earlier 4-pair plot (H100/RTX only)

## Open questions

1. **Test `skip_compiled=True`** for kaitakuai's `poc_model_runner.py:236` — add per-call to `set_forward_context(...)`. Expected: cross-GPU drift drops to eager baseline while compiled inference stays for non-PoC requests. Tracking as `006_…`.
2. **Unify block_hash/pk** between clusters if we want a full 8×8 — re-run one side. A↔B cross-cluster pairs currently meaningless.
