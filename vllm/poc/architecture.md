# PoC Redesign — Discrete Decision-Boundary Fingerprint

**Status:** Design proposal (approved direction, pre-implementation)
**Date:** 2026-06-14
**Scope:** Replace the continuous L2 hidden-state artifact of PoC v2 with a discrete,
decision-boundary fingerprint that cross-validates cleanly across heterogeneous GPUs while
remaining hard to forge by an adversarial prover.

> Written in English to match the gonka codebase / `gonka-fork` docs convention so it can be
> shared with the network team.

---

## 1. Problem & motivation

Gonka PoC measures useful-work capacity: each node runs a deterministic, chain-seeded forward
pass through model `M` for many nonces; nonce count in a window → consensus weight (faster GPU →
more nonces → more weight). Validators on *other* GPUs recompute and compare to detect cheating.

The current PoC v2 commits to a **continuous** 12-dim projection of the last-token hidden state
and compares prover-vs-validator vectors under an **L2 threshold** (`dist_threshold`, governed
per model: Qwen3-235B `0.2`, Kimi-K2.6 `0.4`, MiniMax-M2 `0.75`) with a binomial fraud test.
Code: [poc_model_runner.py](poc_model_runner.py), [validation.py](validation.py),
[data.py](data.py), [gpu_random.py](gpu_random.py).

Two coupled problems:

1. **Cross-GPU drift forces a loose threshold.** Identical models produce different numerics on
   different GPUs/backends/TP sizes because floating-point addition is non-associative and
   reduction order changes with batch size, tensor-parallel size, kernel/backend, and GPU
   microarchitecture. The network *deliberately* runs the same model with different optimal
   configs per (GPU, model) — e.g. Kimi-K2.6 on B200 (`FLASHINFER_MLA`, tp=4) vs H200
   (`FLASHMLA`, tp=8). The loose threshold needed to tolerate this is both:
   - a **perverse incentive** — `VLLM_MOE_USE_DEEP_GEMM=1` lowers a node's drift (DeepGEMM is more
     deterministic), helping it pass PoC, but it degrades inference decode throughput (~−51%); and
   - a **forgeability surface** — a wide tolerance band leaves room for less-than-honest compute.

2. **The literature confirms bitwise cross-GPU determinism is unsolved.** Every determinism
   technique (batch-invariant kernels; TBIK [arXiv:2511.17826]; LLM-42 [arXiv:2601.17768]) fixes
   same-hardware or cross-TP/cross-framework determinism on a *single* architecture; TBIK's own
   paper lists cross-architecture (wgmma on Hopper) as **unsolved**. FP32 reduces but does not
   robustly eliminate cross-GPU drift. **We must design for drift, not eliminate it.**

The standout primitive that *tolerates* drift is the TOPLOC pattern [arXiv:2501.16007]: commit to
top-k **high-magnitude** activations (which avoid catastrophic cancellation) with tolerance
thresholds, robust across GPU types. This design recombines that idea into a **discrete** form
rather than adopting TOPLOC wholesale.

## 2. Requirements

1. **Capacity / useful work.** Nonce throughput in a window → weight. Faster GPU = more weight.
   Must be preserved; per-nonce overhead must stay small relative to the forward.
2. **Cross-GPU validation.** A validator on a *different* GPU must validate a prover's nonces for
   the same model.
3. **Heterogeneous configs allowed.** Per-(GPU, model) optimal vLLM params (different backend/TP)
   are intended. We cannot mandate a single canonical compute path.
4. **Model-identity binding.** Must detect a node that passed PoC on a *different* (cheaper) model.
5. **Anti-substitution.** Substituting/forging the work result during PoC must be infeasible.
6. **Universal.** Must support any model (dense + MoE). Current critical set is all MoE:
   Qwen3-235B, MiniMax-M2.6, Kimi-K2.6.

## 3. Trust & threat model

- **Prover: fully untrusted, runs arbitrary code.** Any prover-side patch is revertable and
  therefore security-irrelevant. Security cannot rely on the prover running our code.
- **Validators: semi-trusted via consensus.** Many validators recompute and emit a signed verdict;
  acceptance is majority-by-weight ([`ComputeNewWeights`](../../../gonka-fork/inference-chain/x/inference/module/chainvalidation.go)).
  A dishonest minority cannot flip the result.
- **Chain governance: the trust anchor.** PoC spec parameters are pushed from chain (today via
  `StatTestParamsFromChain`) and cannot be reverted by one participant.

**The forgery-resistance comes from the math, not from code enforcement:** the validator's
recompute + a tight discrete threshold passes *only* if the prover reproduced honest-`M`'s
high-margin decisions, so the prover is forced to actually compute `M` regardless of what code it
runs.

Precompute is already blocked: the seed is the hash of the block that *opens* the generation
phase, revealed only at challenge time (verified in `decentralized-api/broker` + `epoch_context`).
No challenge-time entropy injection is required.

## 4. Why discrete fingerprints

Stop comparing continuous vectors by L2 (loose threshold, forgeable, backend-sensitive). Commit to
**discrete signals that only change at decision boundaries**, not on every ULP of drift:

- **Universal:** top-k token/logit identities (argmax over the output) — model-specific, present
  in every model.
- **MoE bonus:** expert-routing identities (which experts fired) — a discrete fingerprint of `M`'s
  *learned routing*, an internal signal that is much harder to imitate with a distilled proxy than
  outputs.

Comparison becomes exact-match / Hamming over a statistical population, not a continuous threshold.
Discrete decisions survive backend/TP/architecture drift (they flip only near ties), bind model
identity strongly (a different model routes/predicts differently), and resist forgery (must
reproduce `M`'s real decisions). This directly targets the literature's open questions: discrete
routing as a forgery-resistant commitment, and forgery-resistance against a backend-choosing
adversary.

## 5. Design

### 5.1 Core — what we capture per nonce

**Input (per nonce):** a deterministic, chain-seeded sequence (as today: `block_hash + public_key
+ nonce`), one forward through model `M`. The capacity work is this forward, unchanged. Batching
of nonces is retained for throughput.

**Captured signals — both are "free riders" already computed inside the forward:**

1. **Universal — high-margin top-k token/logit IDs.** Over a seeded set of positions, take
   argmax/top-k of the output logits; **keep only decisions whose gap between top-1 and top-2
   (resp. top-k and top-(k+1)) ≥ `τ_margin`.** Works for any model.
2. **MoE bonus — high-margin expert-routing IDs.** Over a seeded set of MoE layers, take the
   selected expert indices; keep those whose router-score margin ≥ `τ_route`.

**The high-margin filter is the key trick** (discrete analog of TOPLOC's high-magnitude selection):
high-margin decisions do not flip under numerical drift; near-tie decisions are discarded. This is
what makes the fingerprint cross-GPU stable *and* tight.

**Per-nonce commitment:** a compact list of `(seeded_position/layer_id → discrete ID)` over the
high-margin decisions, plus their margins. **Candidate positions/layers are chosen by the seed,
not by the prover** — preventing cherry-picking; the validator checks both that a decision is
genuinely high-margin and that its ID matches.

A single forward yields **many** decisions (routing at every MoE layer × every position is already
computed), so `|S|` per nonce is large for free, giving strong per-nonce statistical power and
binding the **full trajectory** (not just the last token — closing the "surrogate of one
projection" shortcut).

### 5.2 Cross-GPU comparison & statistical test

Validator takes the same seeds, runs the same forward on its (possibly different) GPU/config, and
compares decision-wise.

**Hysteresis comparison set (the cross-GPU cleanliness trick).** The per-signal selection
thresholds from §5.1 are `τ_route` (routing) and `τ_margin` (logits); call the active one `τ_sel`.
- Candidate positions/layers are fixed by the seed.
- Prover commits IDs for decisions with margin ≥ `τ_sel`.
- Validator recomputes and scores **only** decisions where *its own* margin ≥ `τ_sel + Δ_hyst`
  (a hysteresis band above the selection threshold). The validator is the anchor; the prover
  cannot cherry-pick.
- In the scored set `S`: a decision is a mismatch iff `ID_prover ≠ ID_validator`.

**Statistical test:** `binomtest(n_mismatch, |S|, p=p0)`, fraud if `pvalue < threshold` — same
structure as today, but `p0` (honest flip rate) is now tiny because only robustly-high-margin
decisions are scored. A tight threshold becomes possible → model substitution yields ~100%
mismatch → caught with overwhelming statistical power.

**Weight / capacity:** = count of valid nonces in the window (`count × time-norm`, as today).

### 5.3 Model-identity binding & anti-substitution

The honest boundary: **tolerate** "same model, different backend/GPU/precision" (cross-GPU
validation); **catch** "different/cheaper model." High-margin discrete decisions draw exactly this
line — stable under numerics, divergent under model change.

| Attack | Outcome |
|---|---|
| (a) Different / smaller model `M′` | routing + tokens diverge → ~100% mismatch → **caught** |
| (b) Same `M`, cheaper kernel/precision | high-margin decisions preserved → **passes, and that's OK** (it really ran `M`) |
| (c) Partial / early-exit compute | decisions seeded across many *deep* layers; routing at layer L depends on all prior layers → cannot skip → **caught** |
| (d) Replay / sharing nonces | seed includes `public_key` → nonces unique per node key → **not replayable** |
| (e) **Distillation / proxy** | ⚠️ **the main residual risk — see below** |

**Distillation (e)** is the only serious hole, and it hits MoE and dense differently. Distillation
trains a student to match the teacher's *outputs* → a cheap student could reproduce top-**tokens**.
But reproducing `M`'s internal **expert-routing** across all layers via distillation is hard:
routing is an internal learned structure of a *different* architecture, not an output. So:
- **MoE (the three critical models):** routing is the anti-distillation anchor → strong.
- **Dense (no routing):** the token fingerprint is weaker against distillation; mitigated by
  breadth (many deep high-margin decisions the student must match *all* of) plus the fact that a
  faithful distillation of a large model is itself nearly as expensive. **Honest position: strong
  for MoE, partial for dense.** Acceptable because the network's economic core is MoE. Optional
  future hardening for dense: high-margin internal activation-coordinate argmax (kept in reserve).

### 5.4 Capacity & cost

| Signal | Cost | `|S|` |
|---|---|---|
| Expert-routing ID (MoE) | ≈ **0** — router top-k indices already computed in the forward; capture = memory read | huge (all layers × positions), free |
| top-k logit ID (universal) | LM-head matmul per position (vocab is large) → not free | taken at a **few** seeded positions |

- **MoE:** routing supplies the bulk of `|S|` for free → logits minimal or omitted → near-zero
  overhead. Good for any GPU including small ones.
- **Dense (e.g. Qwen2.5-7B on L4):** `|S|` comes from logits → number of seeded positions is the
  "statistical power ↔ cost" knob; small dense models have small vocab heads, so a few LM-head
  matmuls per nonce are cheap relative to the forward. Reserve: argmax over a seeded subset of
  internal coordinates to pad `|S|` without LM-head.

Overhead is uniform and small → nonce-count ranking still measures raw forward throughput; economics
unchanged. Commitments are tiny (bytes per decision), stored via the existing MMR commit. Validation
of one sampled nonce = one forward + cheap compare.

### 5.5 Validation flow & trust anchor

| Layer | Trust | Holds |
|---|---|---|
| Governance / chain (un-revertable) | anchor | PoC spec: seeded candidate-selection rule, `τ_route`/`τ_margin`/`Δ_hyst`/`p0`/`fraud_threshold`, signal set per (model[, GPU-class]) |
| Prover (untrusted, revertable code) | none | produces a commitment; modified vLLM doesn't help — the validator's recompute checks the decisions |
| Validators (official build, honest majority) | consensus | recompute the seeded forward, apply the same governance spec, emit a signed verdict; majority-by-weight |

- **Cross-GPU:** prover on GPU-A in its optimal config; validator recomputes `M` in *its* optimal
  config on GPU-B — backends need not match; high-margin decisions agree by construction.
- **Model binding:** validator recomputes against the **governance-canonical weights of `M`**; a
  different model → decisions don't match → fraud.
- **Honest limit (as today):** decision-wise comparison runs off-chain in each validator's node;
  the chain sees only the signed verdict. Security = honest-majority-by-weight running the official
  check. The discrete commitment is small and validates faster than inference → more validators can
  afford to verify → easier to accumulate honest weight. Fully on-chain verification would require
  ZK proof of the forward pass, which is impractical at LLM scale.

### 5.6 Calibration experiment (implementation step 1, gating)

The viability of the whole design rests on one assumption the literature has *not* measured for our
regime: **the cross-GPU flip rate of high-margin decisions drops to ~0 at an achievable `τ` while
retaining enough `|S|`.** This must be measured before any production code.

Measure (sweeping `τ`):
1. **Honest cross-GPU flip-rate(τ):** same model, same seeds, GPU-A vs GPU-B (different arch +
   different optimal configs/backend/TP). Fraction of margin-≥-τ decisions that flip → curve for
   choosing `τ_select/τ_compare/p0`.
2. **Coverage |S|(τ):** how many high-margin decisions remain per nonce. Find τ where flip-rate is
   tiny *and* `|S|` suffices for power.
3. **Cross-model divergence:** a different model `M′` recomputes `M`'s decisions → mismatch should
   be ~100% (confirms substitution is caught).
4. **Routing vs logits separately:** per-signal flip-rate and divergence → mix weights.
5. **Input choice:** gaussian noise vs real seeded tokens — which gives peakier/more stable
   high-margin decisions.
6. *(optional)* **Distillation probe:** a smaller same-family model — how much routing/token it
   reproduces (quantifies the §5.3 dense residual).

Setup (reuse the Docker + rented-GPU methodology): iterate cheaply on a smaller MoE (e.g.
Qwen3-30B-A3B) + a dense model (Qwen2.5-7B), at least two distinct architectures from the fleet
(e.g. RTX PRO 6000 vs H100 PCIe, then B300/A100), in their **real heterogeneous** configs. Dump
per-(layer, position) routing top-k + scores and top-k logits + scores at seeded positions →
margins → cross-GPU-pair comparison → sweep τ.

**Decision fork:** low flip-rate at a usable τ → design viable; lock thresholds into a per-model
governance table. High flip-rate → blend in a magnitude layer (fall back toward the hybrid) only
where needed.

### 5.7 Implementation phasing (vLLM first; gonka-fork only after validation)

Hard contract: **all development stays in the `vllm` repo until the hypotheses are confirmed.**
No `gonka-fork` (chain / governance / validator) change is made until the calibration experiment
gates the design.

- **Phase 0 — vLLM only (this repo).** Implement discrete-signal capture in `vllm/poc/` behind a
  flag — per-(layer, position) expert-routing IDs + margins, top-k logit IDs + margins at seeded
  positions — plus the offline comparison + analytics tooling. Run the §5.6 calibration experiment
  on rented heterogeneous GPUs and collect the data (flip-rate vs τ, `|S|`, cross-model divergence,
  routing-vs-logits, input choice). **Nothing touches consensus.**
- **Gate.** Hypotheses confirmed (low honest cross-GPU flip-rate at a usable τ with sufficient `|S|`,
  near-100% cross-model divergence) → proceed. Otherwise iterate, or blend a magnitude layer where
  needed, still in vLLM.
- **Phase 1 — gonka-fork (only after the gate).** Add the `PoCModelConfig` governance fields (§6),
  validator-side spec enforcement, chain plumbing, and load the calibrated per-model threshold
  tables. This is where it becomes consensus-affecting.

## 6. Governance parameters (new)

Per (model[, GPU-class]), pushed from chain like today's `StatTest`:
`signal_set` (routing / logits / both), seeded candidate-selection rule + counts,
per-signal selection thresholds `τ_route` and `τ_margin`, the validator hysteresis `Δ_hyst`,
`p0`, `fraud_threshold`. A new `PoCModelConfig` field set; values come from the §5.6 calibration.

## 7. Open risks & future work

- **Distillation against dense models** (§5.3) — partial residual; reserve hardening exists.
- **Calibration generalization** — thresholds measured on a GPU subset must hold for unseen GPU
  pairs; revisit when new architectures join the fleet.
- **Off-chain verdict** (§5.5) — not on-chain-verifiable; mitigated by cheaper verification → more
  honest weight, not eliminated.
- **Flat-distribution inputs** — if a model yields few high-margin decisions on the seeded input,
  raise candidate count or switch input (§5.6 item 5).

## 8. References

**Literature (adversarially verified via deep research):**
- Cross-GPU determinism is an open problem — batch-invariant kernels (Thinking Machines);
  TBIK [arXiv:2511.17826] (cross-arch unsolved); LLM-42 [arXiv:2601.17768]; drift quantified
  [arXiv:2506.09501].
- Drift-tolerant commitment pattern — TOPLOC [arXiv:2501.16007] (top-k high-magnitude + per-dtype
  thresholds; ~0.0009% FP at 4M scale; not 100%).
- Verifiable-ML taxonomy [arXiv:2502.18535]; proof-of-learning [arXiv:2103.05633] (weaker, forgeable).

**Current code:**
- [poc_model_runner.py](poc_model_runner.py) — `execute_poc_forward`
- [gpu_random.py](gpu_random.py) — seeded input / Householder / pick / Haar primitives
- [validation.py](validation.py), [data.py](data.py) — V2 agreement comparator + binomtest
- [layer_hooks.py](layer_hooks.py) — per-layer hooks, `is_poc_forward_active`
