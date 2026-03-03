# Continuous PoC (Proof of Compute) — scheduler-native (v0.15.1 / v1 engine)

This document describes **how PoC runs continuously through the vLLM v1 scheduler loop** (as used in the vLLM v0.15.1 server integration in this repo), **how it coexists with chat**, and **where exactly PoC is integrated into the engine/scheduler/worker pipeline**.

The most important design constraint is that PoC is a **first-class request kind** inside the normal v1 scheduling loop:

- No separate “PoC scheduler”.
- No background GPU stream/thread dedicated to PoC.
- PoC and chat share the same batch/token budget.
- PoC yields to chat via priority scheduling.
- PoC is **prefill-only** and **KV-less** (no KV blocks allocated; PAD slot-mapping).

---

## Critical Hardening (strict-scope)

This repo includes a targeted hardening pass for scheduler-native PoC, scoped to:

- #1 timeout/QoS ambiguity (no “late PoC result streamed as text”)
- #2 idempotency / duplicate suppression (single execution per PoC identity)
- #3 chunked-prefill starvation analysis (see below)
- #6 KV-less PAD safety (assertions at worker boundary)
- #8 one-step contract fragility (guards + tests)

### Lifecycle (conceptual)

Conceptually, PoC is:
**accepted → emitted → resolved**.

Notes:

- `RESOLVED` is tracked client-side (frontend); engine only tracks enough to suppress delivery after abort.
- Abort is **idempotent** and must suppress any later delivery for that `request_id`.

### Output routing hardening (abuse vector #1)

Hard requirement: **PoC outputs must never reach the normal token streaming path**.

Implementation:

- Engine tags PoC outputs with `kind=POC`.
- `AsyncLLM` output handler resolves PoC futures and **drops orphan PoC outputs** (e.g. if the client timed out and removed the waiter).

This prevents the “future timed out → output later appears as generated text” failure mode.

### Engine-side idempotency + dedup (abuse vector #2)

PoC requests are deduplicated in `EngineCore` by a canonical identity key:

- Identity tuple: `(block_hash, public_key, block_height, nonce, seq_len, k_dim)`
- Identity key: `poc:<sha256(identity_tuple)>`

Registry behavior:

- First request for an identity becomes **canonical** (executed once).
- Later duplicates while canonical is in-flight are **rejected immediately** (not scheduled), and the engine emits a PoC error output for that duplicate `request_id`.
- No TTL cache and no replay: once canonical completes, the identity can be accepted again.

### KV-less PAD safety assertions (abuse vector #6)

Worker-side guards enforce KV-less invariants:

- Any request with `poc_params != None` must have **all KV group `block_ids` empty**.
- For extra paranoia, set env `VLLM_POC_ASSERT_KVLESS=1` to assert that the worker block-table row is fully `-1` (PAD) for PoC.

This catches any accidental regression where the scheduler attempts to allocate/append KV blocks for PoC.

### One-step contract guards (abuse vector #8)

PoC is a finish-in-one-step compute path:

- PoC `EngineCoreOutput` must have `new_token_ids == []`.
- PoC output must be terminal (`finish_reason != None`).
- Frontend resolves at most once per `request_id` and drops duplicates.


---

## Chunked prefill feasibility under KV-less + one-step constraints (abuse vector #3)

Chunked prefill is not compatible with the PoC invariants as implemented here.

Why:

- Chunked prefill relies on **carrying partial prefill state across steps**.
- In vLLM, the mechanism for carrying state across steps is the **KV cache**.
- PoC is explicitly **KV-less** (no KV blocks allocated; slot mapping is PAD), so there is no safe place to store intermediate attention state.
- PoC is also a **one-step** contract (finish in the step it executes), which forbids multi-step completion by design.

Mitigation strategy (within current invariants):

- Treat PoC as an **atomic prefill-only job**: it either schedules the full `seq_len` in one step or waits.
- Prevent starvation through priority policy + backoff (client-side) and by bounding PoC `seq_len` / batch token budgets.

If future work wants chunked PoC prefill, at least one invariant must be relaxed:

- allow KV allocation/writes for PoC, or
- introduce a dedicated non-KV persistence mechanism (new feature), or
- allow multi-step PoC completion.

---

## Glossary

- **EngineCore**: the v1 “engine loop” that repeatedly pulls input requests, calls scheduler, runs the model, and emits outputs.
- **Scheduler**: chooses which requests/tokens run this step and prepares `SchedulerOutput` for workers.
- **Worker / ModelRunner**: builds model inputs for a step, runs forward (and normally sampling/decoding), and returns `ModelRunnerOutput`.
- **PoC request**: a `Request` / `EngineCoreRequest` whose `kind == EngineCoreRequestKind.POC` and carries `PoCParams`.
- **KV-less**: scheduler sends empty KV block ids; worker builds PAD slot mappings so the step does not update KV cache.

---

## End-to-end flow (HTTP → scheduler → GPU → HTTP)

### 1) HTTP endpoints

PoC endpoints are mounted under the OpenAI-compatible server.

- Router: `vllm/poc/server/routes.py`
- Prefix: `/api/v1/pow`

Key endpoints:

- `POST /api/v1/pow/init/generate`
  - Starts a **continuous generation loop** in the API process.
  - Optionally starts a callback sender.

- `POST /api/v1/pow/generate`
  - Either queues a job (default) or performs immediate compute (`wait=true`).
  - Calls the engine client’s `poc_compute` for each nonce.

The router is included into the server app from `vllm/entrypoints/openai/api_server.py`.

### 2) API compute: “submit nonce(s) as PoC requests”

The API side does **not** call any special worker path. It calls the engine client.

- `vllm/poc/server/compute.py::compute_artifact()`
  - For each nonce, generates a `request_id` and calls:
    - `engine_client.poc_compute(request_id=..., block_hash=..., nonce=..., seq_len=..., k_dim=..., priority=POC_REQUEST_PRIORITY)`
  - Uses a local timeout loop to avoid blocking forever when the engine is busy.

### 3) AsyncLLM: submit request and await result

The v1 async engine API is implemented on `AsyncLLM`:

- `vllm/v1/engine/async_llm.py::AsyncLLM.poc_compute()`
  - Ensures `output_handler` is running.
  - Delegates to `vllm/poc/engine/bridge.py::poc_compute_impl()`.

- `vllm/poc/engine/bridge.py::poc_compute_impl()`
  - Creates a Future and stores it in `AsyncLLM._poc_waiters[request_id]`.
  - Builds an `EngineCoreRequest(kind=POC, poc_params=PoCParams(...))`.
  - Calls `engine_core.add_request_async(req)`.
  - Awaits the Future (optionally with timeout).
  - On timeout/cancel: best-effort aborts the request.

### 4) EngineCore: busy loop + scheduler step

EngineCore repeatedly:

1. Pulls new requests from its input queue and pushes them into scheduler.
2. Calls `scheduler.schedule()` to produce `SchedulerOutput`.
3. Worker runs a model step for the scheduled batch.
4. Calls `scheduler.update_from_output(model_runner_output)` to produce `EngineCoreOutput`s.
5. Emits outputs to `AsyncLLM.output_handler`.

The PoC request rides exactly the same loop.

### 5) Scheduler: PoC is prefill-only and KV-less

Scheduler-level behavior for PoC:

- **Prefill-only**: PoC schedules only “context tokens”, no decode loop.
- **No chunked prefill** for PoC: it tries to schedule the full `seq_len` if the token budget allows.
- **No prefix-cache reads**: PoC disables prefix caching reads because PoC uses nonce-derived embeddings while its token IDs are dummy/low-entropy; cache hits would be incorrect.
- **KV-less**:
  - Scheduler does **not** allocate KV blocks (`KVCacheManager.allocate_slots` is skipped).
  - Scheduler sends empty KV block ids in `NewRequestData.block_ids`.
  - The worker’s block table row becomes KV-less and produces PAD slot mappings.

This is the critical part for correctness and memory safety:

- If PoC allocated KV blocks but finished via the PoC-fast-path, it could leak KV allocations or pollute cache.
- KV-less scheduling ensures PoC never consumes KV cache capacity.

### 6) Worker/GPU model runner: fill embeddings, run forward, extract result

On the worker, PoC adds two hooks around the normal forward:

- **Before forward**: replace dummy token-id inputs with PoC-generated embeddings.
  - `vllm/v1/worker/gpu_model_runner.py::_fill_poc_inputs_embeds()` delegates to
  - `vllm/poc/engine/gpu.py::fill_poc_inputs_embeds()`

- **After forward**: extract a result vector from hidden states.
  - `vllm/v1/worker/gpu_model_runner.py` calls `extract_poc_results()` (also delegated).

The output is placed into `ModelRunnerOutput.poc_results[request_id]`.

### 7) Scheduler update: finish immediately and emit `EngineCoreOutput.poc_result`

In `scheduler.update_from_output(...)`, PoC requests are finished in the same step:

- Scheduler reads `model_runner_output.poc_results[req_id]`.
- Builds `EngineCoreOutput(request_id=req_id, finished=True, poc_result=...)`.
- Frees request state.

### 8) AsyncLLM output handler: resolve PoC futures

`AsyncLLM._run_output_handler()` pulls `EngineCoreOutputs` and:

- For each `EngineCoreOutput`:
  - If there is a matching future in `_poc_waiters`, it resolves it and removes that output from normal text-output processing.

So PoC results never go through the standard text token streaming path.

---

## Scheduler loop (v1) — detailed step-by-step, with PoC branches

This section is a “walkthrough” of what happens during one EngineCore iteration.

### A) Inputs enter EngineCore

1. API thread/task calls `AsyncLLM.poc_compute()`.
2. `poc_compute_impl()` calls `engine_core.add_request_async(req)`.
3. EngineCore enqueues the request; the busy loop eventually ingests it and calls `scheduler.add_request(request)`.

At this point the request is just another scheduler request:

- `Request.kind == POC`
- `Request.poc_params != None`
- `Request.priority == POC_REQUEST_PRIORITY` (default 100)

### B) `Scheduler.schedule()` chooses a batch

The scheduler has “RUNNING” and “WAITING” queues.

Core outcomes of `schedule()`:

- Which requests are included this step (`scheduled_new_reqs`)
- How many tokens per request (`num_scheduled_tokens`)
- Which KV blocks to use / append for each request (`block_ids`)

#### PoC token budgeting

PoC is special-cased:

- PoC schedules based on:

  `num_new_tokens = request.num_tokens - request.num_computed_tokens`

  (not `num_tokens_with_spec`, not decode placeholders)

- It is capped by the step token budget.

PoC still competes with chat for token budget:

- With `policy="priority"`, a smaller `priority` value is “higher priority”.
- Preemption logic considers larger priority as lower priority (worse).
- With chat at priority 0 and PoC at 100, **PoC yields**.

#### PoC KV-less scheduling

When the scheduler decides to schedule a PoC request, it does **not** call KV allocation.

Instead it uses:

- `new_blocks = kv_cache_manager.empty_kv_cache_blocks`
- The resulting `NewRequestData.block_ids` are empty lists for each KV group.

This is intentionally aligned with the worker block table implementation:

- Empty block ids → KV-less row (`-1` fill) → PAD slot mapping.

### C) Worker builds inputs

The worker receives `SchedulerOutput` and constructs model inputs for the batch.

For PoC requests in that batch:

- The request carries `poc_params` (`seq_len`, `k_dim`, `nonce`, etc.).
- In preprocess, PoC embeddings are generated and written into the `inputs_embeds` buffer.
- `is_token_ids` is updated so the model uses embeddings.

KV-less rows produce PAD slot mappings for all PoC tokens.

### D) Worker runs the model

- The model forward runs as part of the normal step.
- Sampling/decoding is not meaningful for PoC; the PoC path extracts its own result from hidden states.

### E) Scheduler consumes output

- PoC requests are treated as finished immediately.
- Scheduler emits `EngineCoreOutput(..., poc_result=...)`.

### F) AsyncLLM resolves futures

`AsyncLLM.output_handler`:

- Matches `request_id` to `_poc_waiters`.
- Calls `future.set_result(poc_result)`.

The API awaits this future and then constructs HTTP response payload(s).

---

## What exactly is “a PoC result”?

In this integration, the engine returns a dict-like payload (as `poc_result`) shaped like:

```json
{
  "nonces": [123],
  "vectors_b64": ["<base64-encoded-f16-vector>"]
}
```

- `nonces`: one nonce per request (current API submits one nonce per engine request).
- `vectors_b64`: base64 strings of float16 vectors (dimension = `k_dim`).

The details of embedding generation and vector extraction live in:

- `vllm/poc/engine/gpu.py`

---

## Continuous generation loop (server-side)

There are two “continuous” concepts:

1) **Continuous API loop**: a long-running asyncio task inside the HTTP server.
2) **Scheduler loop**: EngineCore’s infinite scheduling/forward loop.

The API loop simply keeps submitting more PoC requests to the engine.

### Where it lives

- `vllm/poc/server/compute.py::generation_loop()`
  - Iterates nonces
  - Calls `engine_client.poc_compute(...)`
  - On timeout or engine busy errors: backs off and retries

### Queue mode vs wait mode

- `POST /api/v1/pow/generate` with `wait=false` (default):
  - Enqueues a job into `GenerateQueue` (`vllm/poc/server/queue.py`).
  - The queue worker consumes jobs and submits PoC requests.

- With `wait=true`:
  - Performs compute immediately in request/response.

### Callback sending

If a callback URL is provided:

- `vllm/poc/server/callbacks.py` manages sending results with retry/backoff.

---

## Integration map: files, symbols, and rationale

This section lists the key integration points and what they do.

### Request typing & stability

- `vllm/v1/engine/__init__.py`
  - Defines `EngineCoreRequestKind.POC` and extends request/output structs.

- `vllm/v1/request.py`
  - `Request.kind` and `Request.poc_params`
  - `Request.is_poc` property

Rationale: PoC must be schedulable as a normal request kind.

### Async engine API

- `vllm/v1/engine/async_llm.py`
  - `AsyncLLM.poc_compute()` and `AsyncLLM.poc_request()`
  - `_poc_waiters` and output handler interception

- `vllm/poc/engine/bridge.py::poc_compute_impl()`
  - Builds `EngineCoreRequest(kind=POC, poc_params=...)`
  - Adds request to engine core
  - Aborts on timeout/cancel

Rationale: PoC should integrate with the same AsyncLLM abstraction as chat.

### Scheduler: PoC coexistence & KV-less behavior

- `vllm/v1/core/sched/output.py`
  - `NewRequestData.poc_params`
  - `SchedulerOutput.poc_req_ids` (informational marker; KV-less scheduling uses empty block ids)

- `vllm/v1/core/sched/scheduler.py`
  - `schedule()` special-cases PoC:
    - prefill-only token accounting
    - **skip KV allocation** → empty `block_ids`
  - `update_from_output()` finishes PoC immediately and emits `poc_result`

Rationale:

- Share the exact same token budget and batching logic.
- Ensure PoC never consumes KV cache capacity.
- Keep the “finish immediately” semantics so PoC doesn’t enter decode.

### Worker: PoC pre/post forward hooks

- `vllm/v1/worker/gpu_model_runner.py`
  - Calls PoC hooks when the batch contains PoC requests.

- `vllm/poc/engine/gpu.py`
  - `fill_poc_inputs_embeds(...)` — embedding generation
  - `extract_poc_results(...)` — result extraction
  - `build_poc_prompt_embeddings(...)` — prompt embedding math
  - `compute_poc_result(...)` — distance computation

Rationale: keep model-runner changes minimal; PoC logic lives in `vllm/poc/engine/*`.

### API/server wiring

- `vllm/poc/server/routes.py`
  - `/api/v1/pow/*` endpoints

- `vllm/entrypoints/openai/api_server.py`
  - Includes `poc_router`.

Rationale: PoC is an additional API surface on the same server.

---

## Practical examples

### A) Using the engine client (Python)

```python
# Pseudocode-ish: depends on your server wiring.
result = await engine_client.poc_compute(
    request_id="uuid-123",
    block_hash="0xabc...",
    public_key="pubkey...",
    block_height=12345,
    nonce=777,
    seq_len=256,
    k_dim=12,
    timeout=5.0,
    priority=100,
)
# result == {"nonces": [777], "vectors_b64": ["..."]}
```

### B) HTTP: start continuous generation

`POST /api/v1/pow/init/generate`

```json
{
  "block_hash": "0xabc",
  "block_height": 123,
  "public_key": "pk",
  "node_id": 0,
  "node_count": 1,
  "group_id": 0,
  "n_groups": 1,
  "params": {"seq_len": 256, "k_dim": 12},
  "url": "https://callback.example.com/poc"
}
```

### C) HTTP: enqueue nonces (queue mode)

`POST /api/v1/pow/generate`

```json
{
  "block_hash": "0xabc",
  "block_height": 123,
  "public_key": "pk",
  "node_id": 0,
  "node_count": 1,
  "nonces": [1,2,3,4],
  "params": {"seq_len": 256, "k_dim": 12},
  "wait": false
}
```

---

## Debugging and correctness checklist

### Confirm PoC is KV-less

- Scheduler-side expectation:
  - A scheduled PoC request should have **empty** `block_ids` in `NewRequestData`.

There is a CUDA-only test that asserts this behavior:

- `tests/poc/test_coexist.py::TestPoCSchedulerNativeCoexistence::test_poc_does_not_allocate_kv_blocks`

### Common failure modes

- **Timeouts under chat load**:
  - API’s `compute_artifact()` intentionally treats timeouts as “engine busy” and retries with backoff.

- **Duplicate `request_id`**:
  - `poc_compute_impl()` rejects duplicate IDs in `_poc_waiters`.

- **Mixing PoC and multimodal encoder batches**:
  - `GPUModelRunner` rejects batches that contain both.

- **Speculative decoding incompatibility**:
  - PoC is prefill-only; do not combine with spec decode expectations.

---

## Notes on scheduling semantics

- Priority scheduling uses numeric `priority`.
- Larger values are treated as **lower priority** (more likely to be preempted).
- Default PoC uses `POC_REQUEST_PRIORITY = 100` to yield to chat priority `0`.

---

## Maintenance notes

If you change PoC semantics, keep these invariants:

1. PoC must remain **first-class** (request kind).
2. PoC must be **prefill-only** and finish in one step.
3. PoC must remain **KV-less** so it cannot exhaust or pollute KV cache.
4. AsyncLLM output handler must keep resolving `_poc_waiters` and exclude PoC outputs from normal token streaming.
