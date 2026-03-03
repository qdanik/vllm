# PoC (Proof of Compute) Module

Proof-of-Compute module for vLLM with parallel GPU execution.

## Quick Start

```bash
# Run tests
pytest tests/poc -v

# CPU-only tests
pytest tests/poc -v -m "not cuda"

# Build Docker image
docker build -f Dockerfile.quick -t vllm:0.15.1-test .

# Run PoC V2 e2e script
docker run --rm --gpus all \
  -v ${HF_HOME:-/data/shared}:/root/.cache/huggingface \
  -v $(pwd)/vllm/poc/e2e/e2e_poc.py:/e2e_poc.py \
  --entrypoint python3 \
  vllm:0.15.1-test \
  /e2e_poc.py
```

See [QUICKSTART.md](QUICKSTART.md) for detailed guide.

## Architecture — Three-Layer Design

```
   ┌─────────────────────────────────────────────┐
   │  consensus/   DO NOT MODIFY — bit-exact math │
   │  (crypto, encoding, transforms, hooks)       │
   └──────────────────┬──────────────────────────┘
                      │ imports
   ┌──────────────────▼──────────────────────────┐
   │  engine/     vLLM v1 scheduler integration   │
   │  (plugin, gpu, scheduler, dedup, bridge)     │
   └──────────────────┬──────────────────────────┘
                      │ imports
   ┌──────────────────▼──────────────────────────┐
   │  server/     HTTP API + runtime              │
   │  (routes, compute, queue, callbacks, models) │
   └─────────────────────────────────────────────┘
```

**Import rule:** `consensus/` → `engine/` → `server/`. Never the reverse.

## Structure

```
vllm/poc/
├── __init__.py         # Package root, exports poc_router
├── _log.py             # Logger with [PoCV2] prefix
├── constants.py        # Shared PoC defaults and priorities
├── env.py              # Environment variables
│
├── consensus/          # Consensus-critical — DO NOT MODIFY
│   ├── crypto.py       # Deterministic CSPRNG (uniform, normal, hash)
│   ├── encoding.py     # Token & vector encoding/decoding
│   ├── transforms.py   # Householder & Haar rotations
│   └── hooks.py        # Per-layer forward-pass hooks
│
├── engine/             # vLLM v1 scheduler-native integration
│   ├── params.py       # PoCSchedulerParams (msgspec Struct)
│   ├── plugin.py       # PoCRunnerPlugin — GPU runner lifecycle
│   ├── gpu.py          # GPU embeddings + result extraction
│   ├── dedup.py        # Dedup registry + identity key
│   ├── scheduler.py    # Scheduler PoC lifecycle helpers
│   ├── bridge.py       # AsyncLLM ↔ EngineCore bridge
│   └── output.py       # PoC output routing & orphan handling
│
├── server/             # HTTP API layer
│   ├── models.py       # All shared types (config, enums, dataclasses)
│   ├── schemas.py      # Pydantic response schemas
│   ├── routes.py       # FastAPI /api/v1/pow/* endpoints
│   ├── compute.py      # Artifact computation + generation loop
│   ├── state.py        # App state helpers
│   ├── queue.py        # Generate queue with TTL
│   ├── callbacks.py    # Async callback sender
│   └── validation.py   # Statistical artifact validation
│
└── e2e/                # Profiling & integration tests
    ├── e2e_poc_chat.py # PoC + chat coexistence test
    ├── e2e_poc_http.py # HTTP PoC + inference workflow
    └── e2e_poc_tiny.py # Minimal PoC validation
```

## Scheduler-Native Architecture

PoC requests are first-class scheduler requests like chat:
- **Request type**: `EngineCoreRequest(kind=EngineCoreRequestKind.POC, poc_params=...)`
- **Priority-based scheduling**: `POC_REQUEST_PRIORITY=100` (yields to chat at 0)
- **Per-nonce model**: One nonce = one scheduler request
- **Prefill-only**: Sequence marked finished after single forward pass
- **KV exclusion**: Empty block tables prevent KV cache writes

### Key Components

**Consensus** ([`consensus/`](consensus/)):
- `crypto.py`: Deterministic CSPRNG — `uniform()`, `normal()`, `poc_hash()`
- `encoding.py`: Token/vector encode/decode for bit-exact results
- `transforms.py`: `apply_householder()`, `apply_haar_rotation()`, `generate_inputs()`
- `hooks.py`: `LayerHouseholderHook`, `poc_forward_context()` — layer-level transforms

**Engine** ([`engine/`](engine/)):
- `plugin.py`: `PoCRunnerPlugin` — self-contained runner plugin
  - `begin_step()`: One-time batch analysis per step (sets `has_poc`)
  - `update_token_mask()`: Per-token PoC mask on GPU
  - `fill_embeds()`: Generate & place PoC embeddings
  - `forward_context()`: Context manager — hooks (all-PoC) or in-graph (mixed)
  - `extract_results()`: Compute PoC outputs from hidden states
- `gpu.py`: GPU data-plane helpers
  - `fill_poc_inputs_embeds()`: Generate embeddings on GPU
  - `extract_poc_results()`: Extract distance from hidden states
  - `build_poc_prompt_embeddings()`: Generate prompt embeddings
  - `compute_poc_result()`: Compute distance from hidden states
- `bridge.py`: Async engine API — `poc_compute_impl()`
- `dedup.py`: `PoCDedupRegistry` + `poc_identity_key()`
- `scheduler.py`: PoC tagging and output finalization for scheduler
- `params.py`: `PoCSchedulerParams` (msgspec Struct)

**Server** ([`server/`](server/)):
- `routes.py`: FastAPI router `/api/v1/pow/*`
- `compute.py`: `compute_artifact()`, `compute_artifacts_chunk()`, `generation_loop()`
- `models.py`: All types — `PoCConfig`, `PoCState`, enums, `Artifact`, etc.
- `schemas.py`: Pydantic response schemas
- `queue.py`: `GenerateQueue` with TTL and worker tasks
- `callbacks.py`: `CallbackSender` with retry/backoff
- `validation.py`: Statistical fraud detection

## Configuration

Environment variables (see [`env.py`](env.py)):

```bash
# Batch sizing
POC_BATCH_SIZE_DEFAULT=32

# Callbacks
POC_CALLBACK_INTERVAL_SEC=5
POC_CALLBACK_MAX_RETRIES=10

# Generate queue
POC_GENERATE_CHUNK_TIMEOUT_SEC=60
POC_GENERATE_RESULT_TTL_SEC=300
POC_MAX_QUEUED_NONCES=100000
```

## Testing

### Test Structure

- **test_coexist.py**: PoC + chat coexistence (V1 scheduler)
- **test_data.py**: Data structures, configs, schemas
- **test_gpu_random.py**: Deterministic GPU RNG (CUDA only)
- **test_layer_hooks.py**: Layer transformation hooks
- **test_poc_first_class_request.py**: First-class request integration
- **test_poc_hardening.py**: Output routing + dedup hardening
- **test_routes.py**: API endpoints, queuing, callbacks
- **test_callbacks.py**: Async callback delivery
- **test_validation_core.py**: Core validation metrics

### Running Tests

```bash
# All tests
pytest tests/poc -v

# Skip CUDA-only tests
pytest tests/poc -v -m "not cuda"

# Single test file
pytest tests/poc/test_routes.py -v

# With coverage
pytest tests/poc --cov=vllm.poc --cov-report=html
```

## Code Guide

### Extending GPU Operations

1. Add helper function to [`engine/gpu.py`](engine/gpu.py)
2. Wire it through [`engine/plugin.py`](engine/plugin.py)
3. Add tests in `tests/poc/` (see `test_poc_first_class_request.py`)

### Adding New API Endpoints

1. Add request/response types to [`server/models.py`](server/models.py) or [`server/schemas.py`](server/schemas.py)
2. Add route handler to [`server/routes.py`](server/routes.py)
3. Add tests in [`tests/poc/test_routes.py`](../../tests/poc/test_routes.py)

Example:
```python
from vllm.poc.server.schemas import MyResponseSchema
from vllm.poc.server.models import MyRequestModel

@router.post("/my-endpoint", response_model=MyResponseSchema)
async def my_endpoint(request: MyRequestModel):
    return MyResponseSchema(...)
```

### Adding New Transformations

1. Implement transformation in [`consensus/transforms.py`](consensus/transforms.py)
2. Add GPU tests in [`tests/poc/test_gpu_random.py`](../../tests/poc/test_gpu_random.py)
3. Integrate in [`consensus/hooks.py`](consensus/hooks.py)

### Logging

```python
from vllm.poc._log import init_poc_logger

logger = init_poc_logger(__name__)
logger.info("Message")  # Output: [PoCV2] Message
```

## Performance

### Profiling

```bash
# Basic profiling
python -m vllm.poc.e2e.e2e_poc

# With custom batch size (OOM caution)
POC_BATCH_SIZE_DEFAULT=64 \
  python -m vllm.poc.e2e.e2e_poc
```

### Optimization Tips

1. **First run warmup**: DeepGEMM takes 6-10 min to compile kernels (one-time)
2. **Batch size**: Adjust `POC_BATCH_SIZE_DEFAULT` for optimal throughput

## Backend Selection (Dockerfile.quick)

```bash
# FP16 MoE (DeepGEMM, default)
ENV VLLM_USE_DEEP_GEMM=1
ENV VLLM_MOE_USE_DEEP_GEMM=1
ENV VLLM_USE_FLASHINFER_MOE_FP8=0

# FP16 MoE (FlashInfer CUTLASS)
ENV VLLM_USE_FLASHINFER_MOE_FP8=1
ENV VLLM_USE_FLASHINFER_MOE_FP16=0
ENV VLLM_USE_DEEP_GEMM=0
ENV VLLM_MOE_USE_DEEP_GEMM=0

# FP16 MoE (FlashInfer TRTLLM)
ENV VLLM_USE_FLASHINFER_MOE_FP8=1
ENV VLLM_USE_FLASHINFER_MOE_FP16=1
ENV VLLM_USE_DEEP_GEMM=0
ENV VLLM_MOE_USE_DEEP_GEMM=0

# Triton (fallback)
ENV VLLM_USE_DEEP_GEMM=0
ENV VLLM_MOE_USE_DEEP_GEMM=0
ENV VLLM_USE_FLASHINFER_MOE_FP8=0
```

See [`Dockerfile.quick`](../../Dockerfile.quick) for full backend guide.

## Contributing

1. Follow the three-layer architecture (`consensus/` → `engine/` → `server/`)
2. Never modify `consensus/` without cryptographic review
3. Add tests for new features
4. Run linter: `ruff check vllm/poc tests/poc`
5. Run tests: `python -m pytest -q tests/poc -v`
6. Update documentation
