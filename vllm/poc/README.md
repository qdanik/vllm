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

# Run PoC V2 emulator
docker run --rm --gpus all \
  -v ${HF_HOME:-/data/shared}:/root/.cache/huggingface \
  -v $(pwd)/vllm/poc/emulators/emulate_poc.py:/emulate_poc.py \
  --entrypoint python3 \
  vllm:0.15.1-test \
  /emulate_poc.py
```

See [QUICKSTART.md](QUICKSTART.md) for detailed guide.

## Structure

```
vllm/poc/
├── core/               # Crypto primitives & transformations
│   ├── crypto.py      # Deterministic random generation
│   ├── transforms.py  # Householder reflections, Haar rotations
│   └── validation.py  # Statistical artifact validation
├── emulators/         # Profiling & emulation
│   └── emulate_poc.py # Local PoC profiling
├── inference/         # vLLM inference integration
│   ├── layer_hooks.py # Per-layer transformations
│   └── model_runner.py # GPU forward pass for PoC
├── protocol/          # API schemas & types
│   ├── config.py      # PoC configuration
│   ├── constants.py   # Protocol constants
│   ├── enums.py       # Status enums & callback paths
│   ├── schemas.py     # Pydantic schemas for API
│   └── types.py       # Base data types
├── runtime/           # HTTP API & request handling
│   ├── callbacks.py   # Async callback sender
│   ├── queue.py       # Generate queue with TTL
│   ├── routes.py      # FastAPI endpoints
│   ├── state.py       # Generation state
│   └── validation_utils.py # Validation utilities
├── utils/             # Helper utilities
│   ├── env.py         # Environment variables
│   └── poc_logger.py  # Logger with [PoCV2] prefix
└── v1/                # vLLM V1 integration
    ├── async_worker.py # Async PoC worker
    ├── constants.py    # Request priorities
    └── request.py      # PoCRequest class
```

## Architecture

### Parallel Execution Model

PoC requests execute in parallel with normal chat inference:
- **Separate queue**: `Scheduler.poc_waiting` (priority-based)
- **Non-blocking**: PoC runs alongside chat batch via separate CUDA streams
- **Fire-and-forget**: Results collected asynchronously

### Key Components

**Core Engine** ([`vllm/v1/core/sched/scheduler.py`](../../v1/core/sched/scheduler.py)):
- `schedule()`: Schedules PoC in parallel with chat
- `_poc_running`: Tracks active PoC request
- `poc_waiting`: Priority queue for pending PoC requests

**Worker** ([`vllm/poc/inference/model_runner.py`](inference/model_runner.py)):
- `execute_poc_forward()`: Runs PoC batch on GPU
- Layer hooks: Apply Householder/Haar transformations

**API** ([`vllm/poc/runtime/routes.py`](runtime/routes.py)):
- POST `/api/v1/pow/init/generate`: Start generation round
- POST `/api/v1/pow/generate`: Generate artifacts (queue or sync)
- GET `/api/v1/pow/generate/{task_key}`: Poll queued result
- GET `/api/v1/pow/status`: Get generation status
- POST `/api/v1/pow/stop`: Stop generation

## Configuration

Environment variables (see [`vllm/poc/utils/env.py`](utils/env.py)):

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

- **tests/poc/test_coexist.py**: Parallel execution & scheduling
- **tests/poc/test_data.py**: Data structures & schemas
- **tests/poc/test_gpu_random.py**: Deterministic GPU RNG (CUDA only)
- **tests/poc/test_layer_hooks.py**: Layer transformation hooks
- **tests/poc/test_poc.py**: End-to-end PoC protocol
- **tests/poc/test_routes.py**: API endpoints
- **tests/poc/test_scheduler_poc.py**: Scheduler integration (CUDA only)

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

### Test Results (Latest)

- ✅ **61 passed** - All functional tests
- ⏭️ **55 skipped** - GPU tests on CPU environment
- ✅ **Linting**: ruff clean
- ✅ **Type checking**: All imports valid

## Code Guide

### Adding New Endpoints

1. Define request/response schemas in [`protocol/schemas.py`](protocol/schemas.py)
2. Add route handler in [`runtime/routes.py`](runtime/routes.py)
3. Update OpenAPI docs
4. Add tests in [`tests/poc/test_routes.py`](../../tests/poc/test_routes.py)

Example:
```python
from vllm.poc.protocol.schemas import MyRequestSchema, MyResponseSchema

@router.post("/my-endpoint", response_model=MyResponseSchema)
async def my_endpoint(request: MyRequestSchema):
    # Implementation
    return MyResponseSchema(...)
```

### Adding New Transformations

1. Implement transformation in [`core/transforms.py`](core/transforms.py)
2. Add GPU tests in [`tests/poc/test_gpu_random.py`](../../tests/poc/test_gpu_random.py)
3. Integrate in [`inference/layer_hooks.py`](inference/layer_hooks.py)

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
python -m vllm.poc.emulators.emulate_poc

# With custom batch size (OOM caution)
POC_BATCH_SIZE_DEFAULT=64 \
    python -m vllm.poc.emulators.emulate_poc
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
