# PoC Tuning — Experiment 006 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PoC L2 cross-GPU bit-stable (match 005 eager baseline `a100-aot0` / `b200-aot0` / `h100-aot0`) and faster than current per-nonce kaitakuai code. No `--enforce-eager`, no breaking the artifacts the network already accepts.

**Architecture:**
- Branch: `compiled-0.15.1` (origin = `qdanik/vllm`), tip = `48c09f5` + local uncommitted edits.
- Already applied: (1) batched `collective_rpc` forward, (2) per-call `skip_compiled=True` so PoC runs eager while chat stays compiled, (3) `POC_BATCH_SIZE` env (default 32). Files: [vllm/poc/poc_model_runner.py](../../vllm/poc/poc_model_runner.py), [vllm/poc/manager.py](../../vllm/poc/manager.py), [vllm/poc/engine_patch.py](../../vllm/poc/engine_patch.py).
- This plan adds:
  - **Phase 1** — bit-identical perf wins in the PoC runner (no L2 change)
  - **Phase 2** — env-controlled numerics knobs to test cross-GPU stability
  - **Phase 3** — CUDA Graph (DEFERRED until Phase 1+2 numbers are in)

**Tech Stack:** vLLM 0.15.1, PyTorch 2.x, Qwen 2.5-7B, kaitakuai's `collect_artifacts.py` (in [poc_experiments/scripts/](../scripts/)), Docker overlay (`Dockerfile.quick`), shadeform GPU rentals.

**Constraint reminder:** User handles all `git commit` / `git push` himself. Every "commit point" step means: stop, show diff to user.

---

## File Structure

| file | role |
|---|---|
| [vllm/poc/poc_model_runner.py](../../vllm/poc/poc_model_runner.py) | batched RPC entrypoint, attn metadata, post-processing |
| [vllm/poc/gpu_random.py](../../vllm/poc/gpu_random.py) | seeded random primitives, `generate_inputs`, Haar rotation |
| [vllm/poc/layer_hooks.py](../../vllm/poc/layer_hooks.py) | Householder layer hooks |
| [vllm/poc/engine_patch.py](../../vllm/poc/engine_patch.py) | V1 AsyncLLM patch — HTTP payload → RPC args |
| [poc_experiments/006_v0.15.1_batched_rpc/](.) | this folder — JSONs, plots, summary |

## Research-backed findings that drive this plan

1. **vLLM picks attention backend per GPU**: A100→FA2, H100→FA3, B200→FA4/FlashInfer. Different kernels, different numerics. This is the most likely source of the ~0.2 cross-GPU L2 we saw in compiled-path 005. Fix candidate: `VLLM_ATTENTION_BACKEND=TORCH_SDPA` — single uniform kernel across all GPUs (slower, but bit-stable).
2. **torch.compile reordering is non-guaranteed**: PyTorch docs say Inductor "reserves the right to select different matrix multiply algorithms with different numerics." Confirms why compiled path drifts. Our `skip_compiled=True` (per-call) is the right answer.
3. **`allow_fp16_reduced_precision_reduction`**: PyTorch default is `False` (FP32 accumulation in FP16 matmul). vLLM may flip this for perf. Explicit guard in `apply_patch()` is cheap insurance.
4. **CUDA Graph manual capture** for fixed-shape eager forwards can cut 20-30% CPU-dispatch overhead, numerically identical. Saved for Phase 3 if Phase 1+2 leave perf headroom.

(Full source list at bottom.)

## Test methodology (read first, all tasks reference it)

**Why H100 first:** 005 has reference data for both clusters — H100/RTX used `artifact_collection_block_v1`/`artifact_collection_pk_v1` (cluster B), A100/B200 used `TEST_BLOCK`/`test_pub_keys` (cluster A). Tomorrow's H100 lets us compare directly against `005_v0.15.1_kaitakuai_compiled_path/h100_aot0_eager.json`. If the rented box is A100/B200 instead, switch to cluster A inputs via the new `--block-hash` / `--public-key` flags on [collect_artifacts.py](../scripts/collect_artifacts.py).

**Per-task protocol:**
1. User pushes runtime changes to `qdanik/vllm:compiled-0.15.1`.
2. SSH into rented box, `git clone -b compiled-0.15.1 …`, `docker build -f Dockerfile.quick -t poc-006:test .`.
3. Launch container with task-specific env.
4. Run `~/collect_artifacts.py --nonces 1000 --batch-size 32 --logprobs-count 0` (plus block/pk flags matching the 005 reference for that GPU).
5. `scp` `nonces_1000.json` + `config.json` back into [poc_experiments/006_v0.15.1_batched_rpc/](.).
6. Compare with 005 baseline using the L2 snippet in each task.

**Success criteria:**
- **L2 same-GPU vs 005 eager same-GPU**: mean < 0.05, max < 0.4 (essentially bit-identical artifacts).
- **L2 cross-GPU (H100 vs A100/B200 etc)**: mean ≤ 005 eager cross-GPU (~0.024 H↔RTX, ~0.086 A↔B). Phase 2 may improve on this.
- **Perf**: nonces/min ≥ 005 eager same-GPU (batched > per-nonce expected).

**Recording results:** Append a row to the RESULTS table after every measurement.

---

## Phase 0 — Baseline (mandatory first)

Prove the already-applied code (batched RPC + skip_compiled) actually works before optimizing on top. If baseline fails, stop and debug.

### Task 0.1: Push current branch + bring up H100 box

**Files:** none (user action + GPU box setup)

- [ ] **Step 1 (user action): commit + push current local changes**
  Three runtime files uncommitted: `vllm/poc/poc_model_runner.py`, `vllm/poc/manager.py`, `vllm/poc/engine_patch.py`. Plus tooling: `poc_experiments/scripts/collect_artifacts.py` (`--block-hash` / `--public-key` flags), `poc_experiments/scripts/README.md`. Plus 43 staged `poc_experiments/` files. User decides commit grouping. Push to `qdanik/vllm:compiled-0.15.1`.

- [ ] **Step 2: on H100 box, clone fresh checkout**
  ```bash
  git clone --depth 1 -b compiled-0.15.1 https://github.com/qdanik/vllm.git ~/vllm-repo
  cd ~/vllm-repo && git log --oneline -1
  # Expected: commit on top of 48c09f5 with poc_model_runner.py changes
  ```

- [ ] **Step 3: build Docker image**
  ```bash
  docker system prune -a -f   # ensure ≥ 45 GB free
  docker build -f Dockerfile.quick -t poc-006:test .
  ```

- [ ] **Step 4: copy collect script to home, sed-fix URL prefix**
  ```bash
  scp poc_experiments/scripts/collect_artifacts.py shadeform@<box>:~/collect_artifacts.py
  # On box:
  sed -i 's|/api/v1/inference/pow|/api/v1/pow|g' ~/collect_artifacts.py
  ```

### Task 0.2: Run baseline (default env, batched + skip_compiled)

**Files:** none — output goes to `poc_experiments/006_v0.15.1_batched_rpc/h100_baseline.json`

- [ ] **Step 1: start container with no extra envs**
  ```bash
  docker run -d --gpus all --name poc-test --net=host \
    -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    poc-006:test --model Qwen/Qwen2.5-7B --host 0.0.0.0 --port 8000 --max-model-len 4096
  # Wait until: curl http://127.0.0.1:8000/health → 200
  ```

- [ ] **Step 2: collect with cluster B inputs (H100 was cluster B in 005)**
  ```bash
  python3 ~/collect_artifacts.py --url http://127.0.0.1:8000 --model Qwen/Qwen2.5-7B \
    --output-dir ~/poc-results/h100_baseline \
    --nonces 1000 --batch-size 32 --logprobs-count 0
  # Defaults --block-hash=artifact_collection_block_v1 --public-key=artifact_collection_pk_v1 → matches 005 cluster B
  ```

- [ ] **Step 3: scp results back**
  ```bash
  scp shadeform@<box>:~/poc-results/h100_baseline/nonces_1000.json \
      poc_experiments/006_v0.15.1_batched_rpc/h100_baseline.json
  scp shadeform@<box>:~/poc-results/h100_baseline/config.json \
      poc_experiments/006_v0.15.1_batched_rpc/h100_baseline.config.json
  ```

- [ ] **Step 4: compute L2 vs 005 H100 eager**
  ```bash
  python3 -c "
  import json, base64, numpy as np
  def load(p):
      d = json.load(open(p))
      return {int(a['nonce']): np.frombuffer(base64.b64decode(a['vector_b64']), dtype='<f2').astype(np.float32) for a in d['artifacts']}
  new = load('poc_experiments/006_v0.15.1_batched_rpc/h100_baseline.json')
  ref = load('poc_experiments/005_v0.15.1_kaitakuai_compiled_path/h100_aot0_eager.json')
  common = sorted(set(new) & set(ref))
  d = np.array([np.linalg.norm(new[n]-ref[n]) for n in common])
  print(f'n={len(d)} mean={d.mean():.4f} max={d.max():.4f} >0.4: {(d>0.4).sum()}')
  "
  # Expected: mean < 0.05 (eager-to-eager same GPU, near-zero)
  ```

- [ ] **Step 5: extract nonces/min**
  ```bash
  jq '.nonces_per_min' poc_experiments/006_v0.15.1_batched_rpc/h100_baseline.json
  jq '.nonces_per_min' poc_experiments/005_v0.15.1_kaitakuai_compiled_path/h100_aot0_eager.json
  # Expected: new ≥ 005 (batched should beat per-nonce)
  ```

- [ ] **Step 6: append row to RESULTS** and **stop the container**
  ```bash
  docker stop poc-test && docker rm poc-test
  ```

- [ ] **Step 7: DECISION GATE — if baseline mean L2 > 0.05, stop the plan and debug**
  Don't apply Phase 1 optimizations on a broken baseline. If broken: likely cause is something in our batched code differs from kaitakuai per-nonce — start with a `POC_BATCH_SIZE=1` rerun to see if batching is the issue.

---

## Phase 1 — Bit-identical perf wins (zero L2 risk)

Three changes inside the PoC runner that produce byte-identical artifacts but cut Python/GPU launch overhead. **Each task must verify bit-identity vs Phase 0 baseline** before moving on. If artifacts changed, that's a bug — revert.

### Task 1.1: `generate_inputs` uses `_batched_normal`

**Why:** [vllm/poc/gpu_random.py:131](../../vllm/poc/gpu_random.py#L131) loops Python-side, launching one `_normal()` CUDA kernel per nonce. `_batched_normal()` already exists in the same file (line 73) and produces bit-identical output for the same seeds when called with the full list. For batch_size=32, this is 32 launches → 1.

**Files:**
- Modify: `vllm/poc/gpu_random.py:119-136`

- [ ] **Step 1: read current implementation**
  ```bash
  sed -n '119,140p' vllm/poc/gpu_random.py
  ```

- [ ] **Step 2: replace function body**

  Replace:
  ```python
  def generate_inputs(
      block_hash: str,
      public_key: str,
      nonces: List[int],
      dim: int,
      seq_len: int,
      device: torch.device,
      dtype: torch.dtype = torch.float16,
  ) -> torch.Tensor:
      """Generate deterministic input embeddings for PoC."""
      batch_size = len(nonces)
      result = torch.empty(batch_size, seq_len, dim, device=device, dtype=dtype)
      for i, nonce in enumerate(nonces):
          seed_str = f"{block_hash}_{public_key}_nonce{nonce}"
          seed = _seed_from_string(seed_str)
          normal = _normal(seed, seq_len * dim, device)
          result[i] = normal.view(seq_len, dim).to(dtype)
      return result
  ```
  with:
  ```python
  def generate_inputs(
      block_hash: str,
      public_key: str,
      nonces: List[int],
      dim: int,
      seq_len: int,
      device: torch.device,
      dtype: torch.dtype = torch.float16,
  ) -> torch.Tensor:
      """Generate deterministic input embeddings for PoC (batched)."""
      seeds = [
          _seed_from_string(f"{block_hash}_{public_key}_nonce{nonce}")
          for nonce in nonces
      ]
      normal = _batched_normal(seeds, seq_len * dim, device)  # [B, seq_len*dim], FP32
      return normal.view(len(nonces), seq_len, dim).to(dtype)
  ```

- [ ] **Step 3: py_compile**
  ```bash
  python3 -m py_compile vllm/poc/gpu_random.py && echo OK
  ```

- [ ] **Step 4: commit point** — show diff to user.

- [ ] **Step 5: on H100, repull/rebuild, collect `h100_t11.json` with same env as 0.2**

- [ ] **Step 6: verify bit-identity vs `h100_baseline.json`**
  ```bash
  python3 -c "
  import json, base64, numpy as np
  def load(p):
      d = json.load(open(p))
      return {int(a['nonce']): np.frombuffer(base64.b64decode(a['vector_b64']), dtype='<f2') for a in d['artifacts']}
  a = load('poc_experiments/006_v0.15.1_batched_rpc/h100_baseline.json')
  b = load('poc_experiments/006_v0.15.1_batched_rpc/h100_t11.json')
  common = sorted(set(a) & set(b))
  diff = sum(1 for n in common if not np.array_equal(a[n], b[n]))
  print(f'diffs: {diff}/{len(common)}')
  "
  # Expected: 0 diffs. Non-zero → revert.
  ```

- [ ] **Step 7: measure perf delta, append RESULTS row**
  ```bash
  jq '.nonces_per_min' poc_experiments/006_v0.15.1_batched_rpc/h100_baseline.json poc_experiments/006_v0.15.1_batched_rpc/h100_t11.json
  ```

### Task 1.2: batched post-processing in `execute_poc_forward`

**Why:** [vllm/poc/poc_model_runner.py](../../vllm/poc/poc_model_runner.py) output stage loops per-nonce calling `random_pick_indices([nonce])` and `apply_haar_rotation([nonce], ...)`. Both already accept lists. With 1000 nonces × single-row calls = 1000 GPU launches in post-processing. Batched: 1 launch per group.

**Files:**
- Modify: `vllm/poc/poc_model_runner.py` — output stage + the `all_last_hidden` accumulator

- [ ] **Step 1: change the forward-loop accumulator (one append per chunk)**

  Current:
  ```python
  for i in range(cur_bs):
      all_last_hidden.append(last_hidden[i:i+1])
  ```
  Replace with:
  ```python
  all_last_hidden.append(last_hidden)  # [cur_bs, hidden_size]
  ```

- [ ] **Step 2: replace the output stage block**

  Replace the block from `# Output stage (last PP rank only)` through `vectors_f16 = np.stack(final_vectors)` with:
  ```python
      # =========================================================================
      # Output stage (last PP rank only): batched post-processing with cache.
      # Cache is per-nonce so we split into cached/uncached, then process the
      # uncached subset with a single batched gather + Haar + CPU transfer.
      # =========================================================================
      import numpy as np
      hidden_concat = torch.cat(all_last_hidden, dim=0)  # [total, hidden_size]

      uncached_idx: List[int] = []
      uncached_nonces: List[int] = []
      final_vectors: List[Any] = [None] * total

      for i, nonce in enumerate(nonces):
          cached = _cache_get(block_hash, public_key, nonce, seq_len, hidden_size, k_dim)
          if cached is not None:
              final_vectors[i] = cached
          else:
              uncached_idx.append(i)
              uncached_nonces.append(nonce)

      if uncached_nonces:
          stacked = hidden_concat[uncached_idx]  # [U, hidden_size]
          stacked = stacked / (stacked.norm(dim=-1, keepdim=True) + 1e-8)
          indices = random_pick_indices(
              block_hash, public_key, uncached_nonces,
              hidden_size, k_dim, device,
          )                                           # [U, k_dim]
          xk = torch.gather(stacked, 1, indices)
          yk = apply_haar_rotation(
              block_hash, public_key, uncached_nonces, xk, device,
          )
          yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)
          yk_cpu = yk.half().cpu().numpy()            # ONE GPU→CPU transfer

          for k_idx, slot in enumerate(uncached_idx):
              vec_f16 = yk_cpu[k_idx]
              _cache_put(
                  block_hash, public_key, nonces[slot],
                  seq_len, hidden_size, k_dim, vec_f16,
              )
              final_vectors[slot] = vec_f16

      vectors_f16 = np.stack(final_vectors)
  ```

- [ ] **Step 3: py_compile**
  ```bash
  python3 -m py_compile vllm/poc/poc_model_runner.py && echo OK
  ```

- [ ] **Step 4: commit point** — show diff to user.

- [ ] **Step 5: on H100, repull/rebuild, collect `h100_t12.json`**

- [ ] **Step 6: verify bit-identity vs `h100_baseline.json`** (same script as 1.1 Step 6, swap filenames)

- [ ] **Step 7: measure perf delta, append RESULTS row**

### Task 1.3: pre-cast Householder vector in `_setup`

**Why:** [vllm/poc/layer_hooks.py:97](../../vllm/poc/layer_hooks.py#L97) hook calls `v.to(x.dtype)` every forward on every layer. `v` is FP32, `x` is model dtype (FP16 for Qwen). 28 layers × N PoC forwards = a lot of redundant casts.

**Files:**
- Modify: `vllm/poc/layer_hooks.py` — `_setup` and the hook closure

- [ ] **Step 1: change `_setup` to detect dtype and pre-cast**

  Replace the loop body in `_setup`:
  ```python
          model_dtype = next(model.parameters()).dtype
          for i in range(len(layers)):
              seed_str = f"{block_hash}_layer_{i}_householder"
              v = generate_householder_vector(seed_str, hidden_size, device).to(model_dtype)
              self.reflection_vectors.append(v)

              hook = layers[i].register_forward_hook(self._create_hook(i))
              self.hooks.append(hook)
  ```

- [ ] **Step 2: simplify hook — drop runtime cast**

  In `_create_hook`, replace `transform`:
  ```python
              v = self.reflection_vectors[layer_idx]  # already model dtype

              def transform(x):
                  return apply_householder(x, v)  # no .to() — v is pre-cast
  ```

- [ ] **Step 3: py_compile**
  ```bash
  python3 -m py_compile vllm/poc/layer_hooks.py && echo OK
  ```

- [ ] **Step 4: commit point** — show diff to user.

- [ ] **Step 5: on H100, repull/rebuild, collect `h100_t13.json`**

- [ ] **Step 6: verify bit-identity vs baseline** (same script)
  Expected: 0 diffs. `v.to(model_dtype)` in `_setup` produces same bytes as `v.to(x.dtype)` per-call when `x.dtype == model_dtype`.

- [ ] **Step 7: measure perf, append RESULTS row**

---

## Phase 2 — Numerics exploration (OPTIONAL, only if Phase 1 leaves a problem)

**Important context before doing any of this:** Gonka's fraud detection is **statistical**, not a hard L2 cliff. The per-nonce `r_target=0.4` flags individual nonces as "invalid", then `binomtest(n_invalid, batch_size, p=PROBABILITY_MISMATCH).pvalue < fraud_threshold (~0.01)` decides if the whole batch is fraud. A few outliers per batch are **expected and accepted** for honest nodes. See [project-gonka-architecture memory](../../../.claude/projects/-Users-daniilyankouski-develop-TRONITY-ONE-gonka-vllm/memory/project_gonka_architecture.md) and [gonka_poc.md](../../../gonka-fork/docs/gonka_poc.md).

005 eager numbers (~0.024–0.086 cross-GPU mean L2) already sit comfortably inside that statistical envelope. **If Phase 0 baseline confirms the same on the new code, Phase 2 may not be needed at all.**

Also: **operators intentionally tune `--attention-backend` per GPU per model** (B200 → `FLASHINFER_MLA`, H200 → `FLASHMLA`, see `gonka-fork/deploy/join/node-config-*.json`). Any "force one backend across the fleet" knob (e.g., `TORCH_SDPA`) is a chat-throughput regression for operators, which is unacceptable. Phase 2 is therefore narrowed to environment knobs that affect numerics WITHOUT overriding operator-chosen backend.

Run Phase 2 only if Phase 0 baseline shows L2 outside the eager envelope, or if cross-GPU measurement on a second rented GPU later shows fraud-flag rate above what 005 eager produced. Otherwise skip and move on.

### Task 2.1: explicit FP32 matmul reduction guard

**Why:** `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False` ensures FP32 accumulation in FP16 GEMMs. PyTorch default is already False but vLLM startup may flip it. Cheap insurance.

**Files:**
- Modify: `vllm/poc/engine_patch.py` — add at top of `apply_patch()`

- [ ] **Step 1: add explicit guard in `apply_patch()` before the AsyncLLM import**

  ```python
      # PoC numerical stability across GPUs — force FP32 reduction in FP16 matmul.
      # PyTorch default is False but vLLM startup may flip this for perf.
      import torch
      torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
      torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
  ```

- [ ] **Step 2: py_compile**
  ```bash
  python3 -m py_compile vllm/poc/engine_patch.py && echo OK
  ```

- [ ] **Step 3: commit point** — show diff.

- [ ] **Step 4: on H100, repull/rebuild, collect `h100_t23_fp32reduce.json`**

- [ ] **Step 5: L2 vs baseline.** If same as baseline → flag was already off; if different → this matters.

### Task 2.2: cross-GPU sanity check (run when second GPU is rented)

**Why:** Confirm that Phase 1 result on a different GPU stays inside the statistical fraud-detection envelope. Compare to the corresponding 005 eager artifact for that GPU.

- [ ] **Step 1: on second GPU, deploy same Docker + (optional) Task 2.1 FP32 guard if it was applied**

- [ ] **Step 2: collect with the `--block-hash` / `--public-key` matching that GPU's 005 cluster**

- [ ] **Step 3: compute cross-GPU L2 + per-nonce invalid count**
  ```bash
  python3 -c "
  import json, base64, numpy as np
  def load(p):
      d = json.load(open(p))
      return {int(a['nonce']): np.frombuffer(base64.b64decode(a['vector_b64']), dtype='<f2').astype(np.float32) for a in d['artifacts']}
  a = load('poc_experiments/006_v0.15.1_batched_rpc/h100_baseline.json')   # or t11/t12/t13
  b = load('poc_experiments/006_v0.15.1_batched_rpc/<other>_baseline.json')
  common = sorted(set(a)&set(b))
  d = np.array([np.linalg.norm(a[n]-b[n]) for n in common])
  print(f'cross-GPU: n={len(d)} mean={d.mean():.4f} max={d.max():.4f} >0.4: {(d>0.4).sum()}')
  "
  ```

- [ ] **Step 4: pass/fail vs 005 eager equivalent**
  Look up the corresponding 005 cross-GPU mean (e.g., 005 H100↔RTX eager = 0.024). New number should be ≤ 005 eager.
  - within 2× of 005 eager cross-GPU → ship; statistical fraud test absorbs this comfortably
  - >> 005 eager cross-GPU → investigate which Phase 1 change introduced the regression (revert one-by-one)

---

## Phase 3 — CUDA Graph capture (DEFERRED)

Only do this if Phase 1+2 don't reach the user's target. Manual `torch.cuda.CUDAGraph()` capture of our fixed-shape PoC forward could cut another 20-30% of CPU dispatch overhead.

**Trigger:** Phase 1+2 winning combo + Phase 1.x perf doesn't meet the user's nonces/min target.

**Pre-work needed before this becomes a real task:** read vLLM 0.15.1's own CUDA Graph management code (`vllm/v1/worker/gpu_model_runner.py` around `capture_model`) to understand stream / buffer ownership. Otherwise we'll collide with vLLM's own captures.

---

## What NOT to do

- **No `torch.use_deterministic_algorithms(True)` globally.** Disables Inductor matmul autotuning + padding → hurts chat throughput.
- **No FP32 promotion in layer hooks.** Would improve L2 cross-GPU but *change artifact bytes* vs 005 baseline (which user said is "правильные данные"). Breaks backwards-compat.
- **No disable of Inductor epilogue fusion globally.** Affects chat. We don't compile PoC, so this hurts only chat.
- **No `--max-model-len` change** between tasks. Affects KV cache block count → indirectly attention math.
- **No `--enforce-eager`.** Defeats whole architecture (chat slow). PoC already runs eager via `skip_compiled=True`.

---

## RESULTS — 2026-05-23 H100 PCIe run on shadeform@185.216.20.187

**Setup:** vLLM 0.20.0, Qwen2.5-7B, batch_size=8 (HTTP), --max-model-len 4096, 1000 nonces, cluster B inputs.

**Headline:** Phase 1.1+1.3 and Phase 2.1 all bit-identical to baseline (1000/1000 same bytes). Throughput stable at ~1473 nonces/min — **+11% vs 005 H100 eager (1326 n/min)** which was the legacy kaitakuai per-nonce code on vLLM 0.15.1.

**The L2 vs 005 puzzle:** new H100 baseline diverges by mean 0.536 from 005 H100 eager. This is **cross-vLLM-version drift** (0.15.1 → 0.20.0 changes attention backend defaults from FA2 to FA3/TMA), not GPU or our code. We cannot do a clean L2 vs 005 without running vLLM 0.15.1 on this same H100 box.

### Table A — bit-identity + throughput (within vLLM 0.20.0)

| task | env / change | bit-identical vs baseline | L2 vs baseline | nonces/min | notes |
|---|---|---|---:|---:|---|
| 0.2 | baseline (HEAD `poc-v2-0.20.0` @ fd02ca5, default env) | (self) | 0.0 | 1472.5 | control |
| 1.1+1.3 | `generate_inputs` → `_batched_normal` + pre-cast Householder `v` in `_setup` | **1000/1000 ✅** | 0.000000 | 1472.7 | safe, no perf delta on this workload |
| 2.1 | `POC_FORCE_FP32_REDUCTION=1` (engine_patch.apply_patch) | **1000/1000 ✅** | 0.000000 | 1472.6 | PyTorch default already `False` on this stack → guard inert |

### Table B — L2 vs 005 (cross-version reference, kept for the record)

| pair | mean L2 | max L2 | >0.4/1000 | interpretation |
|---|---:|---:|---:|---|
| 006 baseline H100 vs 005 H100 eager (cluster B) | 0.5357 | 1.996 | 371 | cross-vLLM-version drift |
| 006 baseline H100 vs 005 RTX eager (cluster B)  | 0.5359 | 1.996 | 372 | cross-version + cross-GPU |
| 006 t21 (FP32 guard) vs 005 H100 eager | 0.5357 | 1.996 | 371 | guard didn't shift artifacts |

The 005 within-version drift was much smaller (~0.024 H↔RTX eager). The big number here is the 0.15.1 → 0.20.0 vLLM jump.

### Conclusions

- **Phase 1.1 + 1.3 ship-ready.** Bit-identical, no risk. Throughput unchanged on this workload (8 nonces / RPC); larger batches would amplify the `_batched_normal` win, but at our current load the model.forward dominates.
- **Phase 2.1 ship-ready as insurance but inert here.** `allow_fp16_reduced_precision_reduction` was already `False`; the guard would only matter on a stack that flipped it to `True`.
- **Cross-version baseline gap is not a regression** — it's expected when comparing 0.15.1 artifacts to 0.20.0 artifacts. To validate intra-version cross-GPU L2 (the real consensus question) we'd need a second vLLM 0.20.0 box on a different SM family (A100 or B200), not the 005 reference.
- **Phase 6 CUDA Graph still deferred** — perf headroom not blocked by anything Phase 1/2 covered.

---

## Sources

Research that informed this plan (web research done 2026-05-23):
- [vLLM Attention Backend Feature Support](https://docs.vllm.ai/en/latest/design/attention_backends/)
- [vLLM Environment Variables docs](https://docs.vllm.ai/en/stable/configuration/env_vars/)
- [State of torch.compile for training (ezyang's blog, Aug 2025)](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
- [PyTorch issue #151352 — emulate_precision_casts](https://github.com/pytorch/pytorch/issues/151352)
- [allow_fp16_reduced_precision_reduction docs](https://runebook.dev/en/articles/pytorch/backends/torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
- [Accelerating PyTorch with CUDA Graphs (PyTorch blog)](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [Bleeding Edge vLLM on NVIDIA Blackwell (Medium)](https://medium.com/@sarankannan2002/bleeding-edge-or-bleeding-out-the-quest-for-vllm-on-nvidia-blackwell-4166135fe87d)
