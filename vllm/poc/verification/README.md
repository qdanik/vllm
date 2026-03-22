# Verification run guide

A short guide for running `run.py` verification (`vllm/poc/verification`).

## 1) Run with Docker

### Build image

```bash
docker build -f Dockerfile.quick -t gonka/vllm:0.15.1 .
```

### Run: FP8 (1000 prompts)

```bash
docker run --rm --gpus all \
  -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  -v ${HF_HOME:-/data/shared}:/root/.cache/huggingface \
  -v /data/vllm/vllm/poc/verification:/verification \
  -v /data/vllm/vllm/results:/results \
  --entrypoint python3 \
  gonka/vllm:0.15.1 \
  /verification/run.py \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --max-model-len 240000 \
  --configs-dir /verification/configs \
  --results-dir /results \
  --logprobs-mode processed_logprobs \
  --limit-prompts 1000 \
  --skip-poc
```

### Run: INT4 category (250 prompts)

```bash
docker run --rm --gpus all \
  -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  -v ${HF_HOME:-/data/shared}:/root/.cache/huggingface \
  -v /data/vllm/vllm/poc/verification:/verification \
  -v /data/vllm/vllm/results:/results \
  --entrypoint python3 \
  gonka/vllm:0.15.1 \
  /verification/run.py \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --max-model-len 240000 \
  --configs-dir /verification/configs \
  --results-dir /results \
  --category inference_int4 \
  --logprobs-mode processed_logprobs \
  --limit-prompts 250 \
  --skip-poc
```

## 2) Run with Python (without Docker)

Below are equivalent runs if dependencies are already installed locally.

Before running with Python, set the required environment variable:

```bash
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
```

### Run: FP8 (1000 prompts)

From the repository root:

```bash
python3 vllm/poc/verification/run.py \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --max-model-len 240000 \
  --configs-dir vllm/poc/verification/configs \
  --results-dir vllm/poc/verification/results \
  --logprobs-mode processed_logprobs \
  --limit-prompts 1000 \
  --skip-poc
```

### Run: INT4 category (250 prompts)

```bash
python3 vllm/poc/verification/run.py \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --max-model-len 240000 \
  --configs-dir vllm/poc/verification/configs \
  --results-dir vllm/poc/verification/results \
  --category inference_int4 \
  --logprobs-mode processed_logprobs \
  --limit-prompts 250 \
  --skip-poc
```

## Where to find results

Artifacts are written to `--results-dir`:

- `inference_*/config.json`
- `inference_*/inference_config.json`
- `inference_*/inference_result.json`
- `inference_*/inference_results.jsonl`
- `inference_*/validation_config.json`
- `session_*/summary.json`
