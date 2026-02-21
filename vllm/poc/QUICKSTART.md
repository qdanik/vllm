# PoC Quick Start

## 🚀 Docker Setup (Recommended)

### 1. Build Image

```bash
# From vllm repository root
docker build -f Dockerfile.quick -t vllm:0.15.1-test .
```

Build time: ~2-5 minutes (with Docker cache)

### 2. Run PoC V2 Emulator

```bash
# Basic run
docker run --rm --gpus all \
  -v ${HF_HOME:-/data/shared}:/root/.cache/huggingface \
  -v $(pwd)/vllm/poc/emulators/emulate_poc.py:/emulate_poc.py \
  --entrypoint python3 \
  vllm:0.15.1-test \
  /emulate_poc.py


**⚠️ First Run**: DeepGEMM takes 6-10 min to compile kernels (one-time warmup)

### 3. Run Tests

```bash
# All PoC tests
docker run --gpus all --rm vllm:0.15.1-test \
    pytest tests/poc -v

# CPU-only tests
docker run --rm vllm:0.15.1-test \
    pytest tests/poc -v -m "not cuda"
```

## 🔧 Local Development

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements/common.txt
pip install -r requirements/test.txt  # For testing
```

### Run Tests

```bash
# All tests
pytest tests/poc -v

# Skip CUDA tests
pytest tests/poc -v -m "not cuda"

# Single test
pytest tests/poc/test_routes.py::TestPoCGenerate -v

# With coverage
pytest tests/poc --cov=vllm.poc --cov-report=html
```

### Run Emulator Locally

```bash
# Basic
python -m vllm.poc.emulators.emulate_poc
```

### Code Quality

```bash
# Lint
ruff check vllm/poc tests/poc

# Auto-fix
ruff check --fix vllm/poc tests/poc

# Compile check
python -m compileall -q vllm/poc tests/poc
```

## 📝 Environment Variables

Key variables (see [README.md](README.md#configuration) for full list):

```bash
# Batch sizing
POC_BATCH_SIZE_DEFAULT=32           # Batch size, prod sends 32 by default

# Callbacks
POC_CALLBACK_INTERVAL_SEC=5         # Callback interval
POC_CALLBACK_MAX_RETRIES=10         # Max retry attempts

# Queue
POC_GENERATE_CHUNK_TIMEOUT_SEC=60   # Request timeout
POC_MAX_QUEUED_NONCES=100000        # Max queued nonces
```

## 🐛 Troubleshooting

### DeepGEMM Warmup Too Long

First run compiles CUDA kernels (6-10 min). Subsequent runs are fast.

**Solution**: Wait for warmup or switch to FlashInfer backend:
```bash
# In Dockerfile.quick
ENV VLLM_USE_DEEP_GEMM=0
ENV VLLM_USE_FLASHINFER_MOE_FP16=1
```

### CUDA Out of Memory

Reduce batch size:
```bash
POC_BATCH_SIZE_DEFAULT=16 python -m vllm.poc.emulators.emulate_poc
```

### Tests Fail on CPU

GPU tests auto-skip on CPU. If failures persist:
```bash
pytest tests/poc -v -m "not cuda"  # Skip all CUDA tests
```

### Import Errors

Ensure vLLM is installed:
```bash
pip install -e .
```

## 📚 Next Steps

- See [README.md](README.md) for architecture details
- Check [tests/poc/](../../tests/poc/) for usage examples
- Read [Dockerfile.quick](../../Dockerfile.quick) for backend selection

## 🔗 Quick Links

- API Docs: [`runtime/routes.py`](runtime/routes.py)
- Protocol Schemas: [`protocol/schemas.py`](protocol/schemas.py)
- Core Logic: [`core/`](core/)
- Test Suite: [`../../tests/poc/`](../../tests/poc/)
