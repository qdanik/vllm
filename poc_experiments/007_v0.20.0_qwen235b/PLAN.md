# 007 — Qwen3-235B-A22B-Instruct-2507-FP8 perf tuning

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify perf knobs we missed in 006 H100 work and quantify their effect on Qwen3-235B-A22B-Instruct-2507-FP8 PoC throughput + L2 stability. Each phase = one knob toggle, isolated A/B against the running baseline, measured against the same reference (`b300_kaitakuai_test.json`).

**Context:** 007's `b300_kaitakuai_test.json` (shared by another node, 2026-05-23) is our cross-GPU intra-version reference. **L2 metric:** mean L2 vs `b300_kaitakuai_test.json` (cluster A seeds, same model). 005 cluster A within-eager A100↔B200 = 0.086 mean → that's the "good" envelope; anything ≤ ~0.10 mean is fine, anything ≥ 0.4 mean is a regression.

**Reference baseline (007 kaitakuai B300, same GPU model as us):**
- Model: `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`
- Seeds: cluster A (`TEST_BLOCK` / `test_pub_keys`)
- Reported: **1280 nonces/min** (kaitakuai table); in-JSON: 1088 nonces in 52.0s = 1255 nonces/min
- File: [b300_kaitakuai_test.json](b300_kaitakuai_test.json)
- TP=1, vLLM 0.20.0, B300
- **Their docker run command not shared yet — ASK for it.** Knowing their exact flags would short-cut this whole plan.

**The gap:** our B300 baseline 864 n/min vs kaitakuai 1280 = we're at **67% of their speed**. Need to find the flags they use that we don't.

**Our setup (Phase 0):**
- Box: `shadeform@95.133.252.45` — NVIDIA B300 SXM6 AC, 275GB VRAM, driver 580.126.09, 1.5TB disk
- Branch: `poc-v2-0.20.0` @ `7f8f9ccbd update`
- Container: `poc-006:test` (overlay on `vllm/vllm-openai:v0.20.0`)
- TP=1 (single B300 fits 235B FP8 with room for KV cache)

---

## Test methodology — fast iteration cycle

Same as 006: edit locally → `scp` + `docker cp` into running container → `docker restart` → wait `/health` → run collect → scp result → analyze.

Container default startup:
```bash
docker run -d --gpus all --name poc-test --net=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  poc-006:test --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --gpu-memory-utilization 0.88
```

Collect default:
```bash
python3 ~/collect_artifacts.py --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --output-dir ~/poc-results/<task_label> \
  --nonces 1000 --logprobs-count 0 \
  --block-hash TEST_BLOCK --public-key test_pub_keys \
  --gpu B300_SXM6 --vllm-version 0.20.0
```

Analysis script (run locally per task):
```python
import json, base64, numpy as np
def load(p):
    d = json.load(open(p))
    return {int(a['nonce']): np.frombuffer(base64.b64decode(a['vector_b64']), dtype='<f2').astype(np.float32) for a in d['artifacts']}, d
new, mn = load('poc_experiments/007_v0.20.0_qwen235b/b300_<task>.json')
ref, mr = load('poc_experiments/007_v0.20.0_qwen235b/b300_kaitakuai_test.json')
common = sorted(set(new) & set(ref))
d = np.array([np.linalg.norm(new[n]-ref[n]) for n in common])
print(f'L2 vs b200 baseline: mean={d.mean():.4f} max={d.max():.4f} >0.4: {(d>0.4).sum()}/1000')
print(f'nonces/min: ours={mn["nonces_per_min"]:.1f} b200={mr["nonces_per_min"]:.1f}')
```

---

## Phase 0 — Baseline (B300, default flags)

Establishes the control row. Everything else compares against this.

- [x] **Phase 0 launched** — container up, collect in progress (2026-05-23)
- [ ] **Step 1: scp result back**
  ```bash
  scp shadeform@95.133.252.45:~/poc-results/b300_baseline/nonces_1000.json \
      poc_experiments/007_v0.20.0_qwen235b/b300_baseline.json
  ```
- [ ] **Step 2: compute L2 vs 007 b200, record in RESULTS Table A row "0 baseline"**

---

## Phase 1 — Env-only knobs (no rebuild, container restart)

Cheapest to test. Container-level env toggles. PoC code unchanged. Run each in isolation (restart container with ONE new env per task).

### Task 1.1: `VLLM_USE_DEEP_GEMM=1`

**Why:** DeepSeek-optimized FP8 kernels for MoE forward. Qwen3-235B is MoE (22B active / 235B total) → potentially significant for both chat AND PoC forward (PoC goes through model.forward which has MoE layers). On Blackwell with FP8 KV cache it can be a 1.3–2× kernel speedup; can also be slower on H20-class.

**Steps:**
- [ ] **Step 1:** restart container with `-e VLLM_USE_DEEP_GEMM=1` added
- [ ] **Step 2:** wait `/health`, verify in logs (grep for "DeepGEMM")
- [ ] **Step 3:** collect `b300_t11_deepgemm.json` (default everything else)
- [ ] **Step 4:** scp + L2 + nonces/min vs Phase 0 baseline (same B300) + vs 007 b200 (cross-GPU)
- [ ] **Step 5:** decision gate
  - same nonces/min ± 3% AND L2 unchanged → keep off (default), it's inert here
  - faster + L2 same as baseline → **WIN**, document
  - L2 shifts but stays within envelope (≤ 0.10 mean vs b200) → acceptable trade
  - L2 over envelope OR slower → revert

### Task 1.2: `VLLM_USE_TRTLLM_ATTENTION=1`

**Why:** TRT-LLM attention through FlashInfer wrappers on SM100+ (B200, B300, RTX Blackwell). vLLM auto-may-pick FA4 on Blackwell already; TRT-LLM path is an alternative that some workloads benefit from. Incompatible with `VLLM_BATCH_INVARIANT=1` — don't combine.

**Steps:**
- [ ] **Step 1:** restart container with `-e VLLM_USE_TRTLLM_ATTENTION=1` (drop DeepGEMM if testing in isolation, OR keep it if 1.1 was a win)
- [ ] **Step 2:** wait `/health`, grep logs for "trtllm" or "TRT-LLM"
- [ ] **Step 3:** collect `b300_t12_trtllm.json`
- [ ] **Step 4:** L2 + nonces/min vs Phase 0 baseline + vs b200
- [ ] **Step 5:** decision gate (same as 1.1)

### Task 1.3: combo `VLLM_USE_DEEP_GEMM=1 + VLLM_USE_TRTLLM_ATTENTION=1`

**Why:** Stack the winners from 1.1 and 1.2 (if both were wins). Sometimes they help independently but conflict together — verify.

- [ ] **Step 1:** only run if both 1.1 AND 1.2 were wins individually
- [ ] **Step 2:** collect `b300_t13_combo.json`
- [ ] **Step 3:** L2 + nonces/min — should add up; if it doesn't, pick the best single knob

---

## Phase 2 — CLI flag knobs (no rebuild, container restart)

### Task 2.1: `--async-scheduling`

**Why:** Overlaps vLLM scheduler bookkeeping with GPU forward execution. Recommended in official Qwen3-235B vLLM recipe for B200. Expected to mostly help chat tps (more decode iterations per second), but may also help PoC if scheduler overhead is non-trivial. Won't affect numerics.

**Steps:**
- [ ] **Step 1:** restart container with `--async-scheduling` appended to vllm serve args (after `--gpu-memory-utilization 0.88`)
- [ ] **Step 2:** wait `/health`
- [ ] **Step 3:** collect `b300_t21_async.json`
- [ ] **Step 4:** L2 + nonces/min vs Phase 0 baseline
- [ ] **Step 5:** decision gate — expect bit-identical L2 (no math change), perf may shift

### Task 2.2: `--compilation-config '{"compile_sizes":[8]}'`

**Why:** PoC uses HTTP `batch_size=8` → server processes 8 nonces per RPC forward. By default vLLM auto-detects and compiles for a set of sizes; explicitly specifying 8 in `compile_sizes` lets Inductor optimize for our exact PoC shape. **However**: this affects the AOT-compiled inference path. Our PoC uses `skip_compiled=True` so it shouldn't touch PoC artifacts. Mostly a chat-throughput knob.

**Steps:**
- [ ] **Step 1:** restart container with `--compilation-config '{"compile_sizes":[8]}'`
- [ ] **Step 2:** verify Inductor compiled size 8 in logs at startup
- [ ] **Step 3:** collect `b300_t22_compile8.json`
- [ ] **Step 4:** L2 must be bit-identical (skip_compiled bypasses this); nonces/min may shift slightly

---

## Phase 3 — PoC code-level (requires patch + docker cp + restart)

These actually touch our PoC code. Higher risk for L2 shifts.

### Task 3.1: `@torch.compile` on `apply_haar_rotation`

**Why:** [apply_haar_rotation](../../vllm/poc/gpu_random.py) loops k-1=11 times over the tensor: `_batched_normal` → norm → cast → dot → update. 11 separate kernel launches per PoC forward. Wrapping with `@torch.compile(mode="max-autotune")` lets Inductor fuse them into 1-2 kernels.

**Risk:** Inductor may reorder reductions → tiny numerics drift. Need L2 check.

**Patch:**
```python
# In vllm/poc/gpu_random.py, before def apply_haar_rotation:
_HAAR_COMPILED = None

def _compile_haar():
    global _HAAR_COMPILED
    if _HAAR_COMPILED is None and os.getenv("POC_COMPILE_HAAR", "1") == "1":
        _HAAR_COMPILED = torch.compile(_apply_haar_inner, mode="reduce-overhead")
    return _HAAR_COMPILED

def _apply_haar_inner(y, v_batch_list):
    """Inner Haar loop, torch.compile-able. Takes pre-generated v vectors."""
    for v in v_batch_list:
        dot = (y * v).sum(dim=-1, keepdim=True)
        y = y - 2 * dot * v
    return y

# Then in apply_haar_rotation, separate out the seed/random generation
# (which has stateful murmur3 calls Inductor can't trace) from the per-step
# Householder math (which is pure tensor ops).
```

(Exact patch shape: pre-generate all v_batches outside compile region, then call compiled inner with the list.)

**Steps:**
- [ ] **Step 1:** apply patch locally, `py_compile`, scp + docker cp + restart
- [ ] **Step 2:** wait `/health`
- [ ] **Step 3:** collect `b300_t31_compile_haar.json`
- [ ] **Step 4:** L2 vs Phase 0 baseline (same box, only this changed): expect **bit-identical** if Inductor doesn't reorder, OR small drift
- [ ] **Step 5:** decision gate
  - bit-identical AND faster → **WIN**
  - drift but within envelope vs b200 AND faster → **acceptable** (artifact bytes change, but consensus holds)
  - drift over envelope → revert (we already discussed: "L2 b200-aot0 это правильные данные")

### Task 3.2: `cudagraph_runtime_mode=PIECEWISE` in `set_forward_context`

**Why:** vLLM's `set_forward_context` accepts a `cudagraph_runtime_mode` kwarg (see [vllm/forward_context.py:251](../../vllm/forward_context.py#L251)). Currently we pass `skip_compiled=True` but NOT this — meaning our PoC forward runs eager with **per-kernel CPU dispatch overhead** on every layer.

Setting `cudagraph_runtime_mode=PIECEWISE` (or `FULL`) tells vLLM to capture this forward in a CUDA Graph and replay it. Same eager kernels, but one `cudaGraphLaunch` instead of thousands of cuLaunchKernel calls.

**Risk:**
- vLLM internal CUDA Graph capture infrastructure may conflict with our manual call.
- First call captures (slow), subsequent calls replay (fast).
- Tensors must be pre-allocated at fixed addresses — our current code allocates `positions`, `input_ids`, `inputs_embeds` per call. Need static buffers.

**Steps:**
- [ ] **Step 1:** Investigate vLLM 0.20.0's PIECEWISE cudagraph semantics in [vllm/forward_context.py](../../vllm/forward_context.py) and surrounding code. **This is research-first, not patch-first.**
- [ ] **Step 2:** If feasible, patch poc_model_runner.py to:
  - pre-allocate `input_ids`, `positions`, `inputs_embeds_buffer` once per `(batch_size, seq_len)` shape
  - call `set_forward_context(..., skip_compiled=True, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE)`
  - On second+ call, replay should be fast
- [ ] **Step 3:** collect `b300_t32_cudagraph.json`
- [ ] **Step 4:** L2 must stay bit-identical (same kernels, same addresses)
- [ ] **Step 5:** nonces/min — expected significant uplift (literature: 20-30%)

---

## What NOT to do

- **Don't `VLLM_USE_DEEP_GEMM=1` blindly** — known to slow down some hardware. A/B test required.
- **Don't combine `VLLM_USE_TRTLLM_ATTENTION=1` with `VLLM_BATCH_INVARIANT=1`** — incompatible.
- **Don't `--enable-expert-parallel`** on TP=1 — needs multi-GPU sharding (we have single B300).
- **Don't `torch.compile` `_batched_normal` or `_seed_from_string`** — they use murmur3 hashing which Inductor can't trace.
- **Don't `-O3`** without `-O2` baseline first — same currently but may diverge.
- **Don't change `--max-model-len`** between tasks (affects KV cache block count → indirectly attention math).

---

## RESULTS

### Table A — B300 Qwen3-235B perf matrix (fill in after each task)

All collects: cluster A seeds (`TEST_BLOCK` / `test_pub_keys`), 1000-1500 nonces. HTTP `--batch-size` varies per row. L2 vs `b300_kaitakuai_test.json` (reference, same GPU model + same model + same seeds).

| task | env / change | HTTP bs | nonces/min | Δ vs baseline | L2 mean vs ref | max L2 | >0.4/1000 | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 0 | baseline (default flags, no env, default container) | 8 | **864** | — | **0.0862** | 0.778 | 1/1000 | control. ✅ L2 in envelope; 33% slower than kaitakuai (1280) |
| K-cold | full kaitakuai env+CLI cfg, 1st call (JIT cold) | 8 | 697 | -19% | 0.0850 | 0.404 | 1/1000 | mode-1 STOCK_TORCH_COMPILE cold-warmup cost |
| K-warm | full kaitakuai cfg, 2nd call (warmed) | 8 | 1137 | +32% | 0.0850 | 0.404 | 1/1000 | 89% of kaitakuai 1280 |
| K-bs32-cold | kt cfg, bs=32 1st call | 32 | 640 | -26% | (same) | — | — | new shape → recompile cost |
| K-bs32-warm | kt cfg, bs=32 2nd call | 32 | 1218 | +41% | 0.0850 | 0.404 | 1/1000 | 95% of kaitakuai 1280 |
| **K-bs64** | **kt cfg, bs=64 (1500 nonces, amortized warmup)** | 64 | **1280** | **+48%** | **0.0850** | 0.404 | 1/1000 | ✅ **MATCH KAITAKUAI PEAK** |
| K-bs128 | kt cfg, bs=128 | 128 | — | — | — | — | — | ❌ **CUDA OOM** (needs 8.12 GiB, free 5.06 GiB) — bs=64 is the ceiling at gpu-mem-util 0.95 |
| H-eager-cold | + Haar refactor patch (POC_COMPILE_HAAR=0, default eager) | 64 | 873 | -2% (vs baseline; warmup) | 0.0000 vs K-bs64 | 0 | 0 | ✅ refactor bit-identical (1500/1500 same bytes vs prev K-bs64) |
| H-eager-warm | Haar refactor eager, warmed | 64 | **1280** | +48% | 0.0850 | 0.404 | 1/1000 | refactor adds zero overhead in eager path |
| H-compile-cold | + `POC_COMPILE_HAAR=1` (torch.compile inner Householder loop), 1st call | 64 | 881 | -36% (cold) | 0.0850 | 0.404 | 1/1000 | cold = compile cost |
| H-compile-warm | `POC_COMPILE_HAAR=1`, warmed | 64 | **1280** | +48% | 0.0850 | 0.404 | 1/1000 | **same as eager — Haar NOT the bottleneck; ❌ DON'T SHIP this flag** |
| KV-batch-cold | + Phase 3.3 kv_scratch batched fill, 1st call | 64 | **1143** | **+32% vs prior cold** | 0.000001 vs prev | (1459/1500 bit-id) | ✅ **WIN on cold start** |
| KV-batch-warm | Phase 3.3, warmed | 64 | 1280 | 0 vs prev warm | 0.000001 | same ceiling — model.forward dominates steady-state |

### Phase 3.3 — batched kv_scratch fill (today, 2026-05-24)

`poc_model_runner.py:execute_poc_forward` (kv_scratch path) was doing **per-nonce** `_normal()` loop. Phase 1.1 batched only `generate_inputs` (gpu_random.py), but the KV-scratch fast path bypasses `generate_inputs` entirely. Fixed: collapse the per-nonce loop into one `_batched_normal` call.

| task | env / change | HTTP bs | nonces/min | Δ | L2 mean vs ref | notes |
|---|---|---:|---:|---:|---:|---|
| KV-batch-cold | kt cfg + Phase 3.3 patch, 1st call | 64 | **1143** | **+30% vs prior cold ~880** | 0.000001 vs kt_bs64 (1459/1500 bit-id) | Cold-start meaningful win |
| KV-batch-warm | Phase 3.3, warmed | 64 | 1280 | 0 vs prev warm | 0.000001 | **steady-state still capped at 1280 — model.forward dominates** |

**Verdict:** Phase 3.3 reduces input-generation overhead measurably on cold-start (first ~50s of a PoC session), but does NOT lift the steady-state ceiling. model.forward (Qwen3-235B-A22B MoE, 94 layers × bs=64 × seq_len=1024) is the only remaining lever. Ship the patch — it's a real cold-start improvement and bit-near-identical (max L2 0.0005, well under threshold) — but don't expect a steady-state delta.

### Phase 4 — Profile execute_poc_forward (2026-05-24) — CONCLUSIVE

Instrumented `poc_model_runner.py:execute_poc_forward` with `torch.cuda.synchronize()` checkpoints (`POC_PROFILE=1`). Ran 1500-nonce collect at bs=64 with parallel `nvidia-smi dmon -s pucvmet` + `mpstat -P ALL` for system-level utilization.

**Steady-state breakdown (bs=64, total ~2900 ms/chunk, ~22 chunks measured):**

| segment | time (ms) | % of total | notes |
|---|---:|---:|---|
| attn_meta | 4.5 | 0.2% | KV-cache metadata build |
| positions | 0.0 | 0.0% | tensor.arange |
| inputs_ready | 24.4 | 0.8% | Phase 3.3 batched seed→normal fill (formerly per-nonce loop) |
| **model_forward** | **2876** | **98.9%** | **`self.model(positions, inputs_embeds)` — Qwen3-235B-A22B MoE, 94 layers** |
| post_processing | 3.6 | 0.1% | logprob extraction + Haar rotation + Householder |

**Cold first chunk:** 35.3s total (31.3s = model_forward JIT/compile, 3.97s = post_processing first-call sampler/FlashInfer warmup). Steady-state stabilizes at ~2.9s/chunk from chunk #2 onward.

**System utilization during steady-state PoC run:**

| metric | value | meaning |
|---|---:|---|
| GPU SM % | **100%** | GPU is doing compute continuously, not waiting on host |
| GPU memory bandwidth % | 20-23% | NOT memory-bandwidth-bound; FP8 GEMM compute-bound |
| GPU power | **1050-1085 W** | **at TDP cap (~1100W B300)** |
| GPU pviol (power throttling) | **100%** | **silicon is power-throttling — already over budget** |
| GPU clock | 1900-1980 MHz | reduced from 2032 MHz idle due to pviol |
| memory clock | 3996 MHz | pegged at max |
| CPU total %usr (avg over 90s) | **1.64%** | **CPU effectively idle**, single hot core 43% (HTTP server) |

**VERDICT — we are at the silicon ceiling.**

1. **CPU is NOT the bottleneck.** Host work (attn_meta + positions + inputs_ready + post_processing) = **1.1% of total time**. Even cutting it to zero would buy us <1% throughput.
2. **GPU is the bottleneck and is power-throttled at the TDP cap.** SM at 100%, power at 1050-1085W with pviol=100% means the chip cannot do more FP8 GEMM per unit time — it's thermally/electrically capped, not waiting on data movement (memBW only 22%) and not waiting on host (CPU 1.6% busy).
3. **1280 n/min IS the hardware wall for Qwen3-235B on a single B300** in our config. kaitakuai hit the same ceiling because the same model on the same silicon runs the same forward pass.

**What cannot improve perf (proven by data):**
- More CPU-side batching / pipelining — host has nothing to do
- CUDA graphs — kernel launches are not the bottleneck (SM is 100% busy doing compute, not idle between launches)
- Faster `_batched_normal` / Haar — already at 0.8% + 0.1%, can't get below zero
- More aggressive kv_scratch tricks — input prep is 0.8%

**What could theoretically improve perf (but each has cost):**
- **Lower-precision MoE kernels** (e.g., FP4 quant if model supports) — changes the model
- **Raise GPU power cap via BIOS** — if datacenter allows, would lift pviol
- **Different MoE backend with higher arithmetic intensity** — kaitakuai already chose FlashInfer latency-mode which is the fastest known for B300
- **Multi-GPU TP=2** — splits the 22B active params across two B300s → ~half model_forward → ~2× throughput per pair, but uses twice the HW; kaitakuai's 8-GPU table shows linear scaling (10240/8 = 1280)

**Net recommendation for the Gonka network:**
- **Ship current branch as the B300 single-GPU production config.** It matches kaitakuai's published peak (1280 n/min) — there's no leftover code-side perf to chase.
- The cold-start improvement from Phase 3.3 (batched kv_scratch) is real (+30% on chunk 1, ~5-10% on first 60s amortized) — ship it.
- Phase 3.1 (torch.compile Haar) ships off-by-default — pure overhead with no warm benefit, +0.5% only on cold (insignificant given Phase 3.3 dominates cold).
- For higher absolute throughput, the next move is HW (multi-GPU TP/EP scaling), not code.

### Phase 3.1 verdict — Haar post-processing is not the bottleneck

vs `b300_haar_eager.json` (eager control on same warmed container):
- compile-haar: bit-identical 1459/1500 bytes; L2 mean=0.000001, max=0.00049 (Inductor reorders reductions slightly)
- nonces/min identical to eager (1280)

Model.forward (Qwen3-235B, 94 layers MoE × cur_bs × seq_len tokens) dominates ~99%+ of PoC time. Optimizing the k_dim=12 Householder loop (11 small GPU ops on [B, 12] tensors) gives no measurable speedup. **Don't enable POC_COMPILE_HAAR in production** — pure compile overhead for no win.

### Table B — L2 envelope context (from 005 same-cluster eager runs, for reference)

| reference pair | mean L2 | notes |
|---|---:|---|
| 007 b300_kaitakuai_test vs 005 b200_aot0_eager (Qwen3-235B same model, same seeds, same GPU model) | 0.085 | "good" envelope ceiling for same-model cross-run |
| 005 A100 ↔ B200 within-cluster eager | 0.086 | within-eager cross-GPU drift for Qwen3-235B |
| 005 H100 ↔ RTX within-cluster eager | 0.024 | within-eager cross-GPU drift for Qwen2.5-7B (3.5× smaller — fewer layers) |

Any 007 task with mean L2 > ~0.10 vs `b300_kaitakuai_test.json` is suspicious. > 0.4 mean → unship.

---

## Sources

Deep-research findings that drove this plan (2026-05-23):
- [vLLM Optimization and Tuning docs](https://docs.vllm.ai/en/stable/configuration/optimization/) — `-O` levels, compile_sizes, max_num_batched_tokens
- [vLLM Recipes Qwen3-VL on B200](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html) — `--async-scheduling`, `--enable-expert-parallel`, `--mm-encoder-tp-mode data`
- [vLLM CUDA Graphs design](https://docs.vllm.ai/en/stable/design/cuda_graphs/) — `cudagraph_capture_sizes`, FULL_AND_PIECEWISE, max_cudagraph_capture_size
- [vLLM Environment Variables](https://docs.vllm.ai/en/stable/configuration/env_vars/) — `VLLM_USE_DEEP_GEMM`, `VLLM_USE_TRTLLM_ATTENTION`, `VLLM_BATCH_INVARIANT`
- [vLLM torch.compile blog (Aug 2025)](https://vllm.ai/blog/2025-08-20-torch-compile) — compile_sizes, compile cache, piecewise CUDA Graphs
- [DeepWiki: vLLM env vars deep dive](https://deepwiki.com/vllm-project/vllm/2.3-environment-variables-system) — DeepGEMM, TRTLLM hardware support matrix
- [QuantTrio Qwen3-VL-235B-A22B-Instruct-FP8 HF](https://huggingface.co/QuantTrio/Qwen3-VL-235B-A22B-Instruct-FP8) — model card (note: VL variant, not Instruct-2507 specifically)
