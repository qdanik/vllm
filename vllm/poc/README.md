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

See [GLOSSARY.md](GLOSSARY.md) for naming, acronyms, and module map.

## Structure

```
vllm/poc/
├── constants.py        # Shared PoC defaults and priorities
├── env.py              # Environment variables
├── api/                # FastAPI routes + request handling
├── core/               # Consensus-critical components
│   ├── crypto.py       # Deterministic CSPRNG
│   ├── encoding.py     # Token & vector encoding
│   ├── layer_hooks.py  # Per-layer transformation hooks
│   ├── transforms.py   # Householder & Haar transformations
│   └── validation.py   # Statistical artifact validation
├── e2e/                # Profiling & testing tools
│   ├── e2e_poc.py       # CPU PoC profiling
│   ├── e2e_poc_chat.py  # PoC + chat coexistence test
│   └── e2e_poc_http.py  # HTTP PoC + inference workflow
├── protocol/           # Runtime types, schemas, queue, callbacks
│   ├── config.py       # PoC configuration
│   ├── api_schemas.py  # Pydantic request/response schemas (canonical)
│   ├── runtime_types.py # Dataclasses for runtime payloads (canonical)
│   ├── status_enums.py # Status enums (canonical)
│   ├── callbacks.py    # Async callback sender
│   ├── queue.py        # Generate queue with TTL
│   └── state.py        # Generation state tracking
├── utils/              # Helper utilities
│   └── poc_logger.py   # Logger with [PoC] prefix
└── v1/                 # V1 scheduler-native integration
  ├── scheduler_params.py            # Scheduler-native params (canonical)
  ├── scheduler_integration.py       # Scheduler PoC lifecycle helpers
  ├── runner_plugin.py               # PoCRunnerPlugin – self-contained runner plugin
  ├── gpu_model_runner_integration.py # Embedding & result data-plane helpers
  ├── gpu_artifacts.py               # GPU artifact compute (canonical)
  ├── async_engine_integration.py    # AsyncLLM helper (canonical)
  ├── dedup_registry.py              # Engine-core PoC dedup/abort registry
  ├── identity_key.py                # SHA-256 identity key computation
  ├── engine_output_filtering.py     # PoC output routing & orphan handling
  └── ...                            # Shims: async_engine.py, gpu_runner.py, etc
```

## Architecture

### Scheduler-Native Architecture

PoC requests are first-class scheduler requests like chat:
- **Request type**: `EngineCoreRequest(kind=EngineCoreRequestKind.POC, poc_params=...)`
- **Priority-based scheduling**: `POC_REQUEST_PRIORITY=100` (yields to chat at 0)
- **Per-nonce model**: One nonce = one scheduler request
- **Prefill-only**: Sequence marked finished after single forward pass
- **KV exclusion**: Empty block tables prevent KV cache writes

### Key Components

**V1 Integration** ([`v1/`](v1/)):
- `scheduler_integration.py`: PoC tagging and PoC output finalization for scheduler
- `runner_plugin.py`: `PoCRunnerPlugin` — self-contained runner plugin
  - `begin_step()`: One-time batch analysis per step (sets `has_poc`)
  - `update_token_mask()`: Per-token PoC mask on GPU
  - `fill_embeds()`: Generate & place PoC embeddings
  - `forward_context()`: Context manager — hooks (all-PoC) or in-graph (mixed)
  - `extract_results()`: Compute PoC outputs from hidden states
- `gpu_model_runner_integration.py`: Data-plane helpers
  - `fill_poc_inputs_embeds()`: Generate embeddings on GPU
  - `extract_poc_results()`: Extract distance from hidden states
- `async_engine_integration.py`: Async engine API
  - `poc_compute_impl()`: Submit nonce & await result
- `gpu_artifacts.py`: GPU computation functions
  - `build_poc_prompt_embeddings()`: Generate prompt embeddings
  - `compute_poc_result()`: Compute distance from hidden states
- `scheduler_params.py`: Scheduler-native params (`PoCSchedulerParams`)

## Configuration

Environment variables (see [`vllm/poc/env.py`](env.py)):

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

### Test Coverage

- ✅ **V1 integration** - scheduler-native request flow + GPU runner hooks
- ✅ **Runtime components** - callbacks, queue, routes
- ✅ **Core validation** - Statistical metrics
- ✅ **Protocol** - Schemas, runtime types, status enums
- ✅ **E2E** - CPU profiling, PoC+chat coexistence

Run coverage report:
```bash
pytest tests/poc --cov=vllm.poc --cov-report=html --cov-report=term-missing
```

## Code Guide

### Extending V1 Integration

To add new GPU operations:
1. Add helper function to [`v1/gpu_artifacts.py`](v1/gpu_artifacts.py)
2. Call it from [`v1/gpu_model_runner_integration.py`](v1/gpu_model_runner_integration.py)
3. Add/extend tests in `tests/poc/` (see `test_poc_first_class_request.py`)

### Adding New E2E Scripts

1. Create script in [`e2e/`](e2e/)
2. Define test profile in `e2e_*.py`
3. Add test case to `test_e2e.py`
4. Add tests in [`tests/poc/test_routes.py`](../../tests/poc/test_routes.py)

Example:
```python
from vllm.poc.protocol.api_schemas import MyRequestSchema, MyResponseSchema

@router.post("/my-endpoint", response_model=MyResponseSchema)
async def my_endpoint(request: MyRequestSchema):
    # Implementation
    return MyResponseSchema(...)
```

### Adding New Transformations

1. Implement transformation in [`core/transforms.py`](core/transforms.py)
2. Add GPU tests in [`tests/poc/test_gpu_random.py`](../../tests/poc/test_gpu_random.py)
3. Integrate in [`core/layer_hooks.py`](core/layer_hooks.py)

### Logging

Use PoC-specific logger with `[PoCV2]` prefix:

```python
from vllm.poc.utils.poc_logger import init_poc_logger

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

1. Follow existing code structure
2. Add tests for new features
3. Run linter: `ruff check vllm/poc tests/poc`
4. Run tests: `python -m pytest -q tests/poc -v`
5. Update documentation
