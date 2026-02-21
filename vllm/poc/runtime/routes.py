"""PoC API routes for vLLM v0.15.1 server with non-blocking async parallel inference.

Key changes from v0.9.1:
- PoC is submitted to the v1 scheduler as a first-class request (no KV cache).
- Scheduler emits PoC alongside normal batch in SchedulerOutput.
- Worker.execute_model() fires PoC in a background thread + dedicated CUDA stream,
  then immediately executes normal chat batch and returns chat results.
- PoC results arrive in a future scheduler-loop iteration via CUDA event polling.

Benefits:
- Chat returns independently regardless of PoC duration
- No preemption: PoC and chat are fully independent
- OOM safety: PoC OOM → FinishReason.ERROR, main loop unaffected
"""

import asyncio
import contextlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from vllm.logger import init_logger
from vllm.poc.protocol.config import PoCConfig, PoCState
from vllm.poc.protocol.constants import (
    DEFAULT_DIST_THRESHOLD,
    DEFAULT_FRAUD_THRESHOLD,
    DEFAULT_K_DIM,
    DEFAULT_P_MISMATCH,
)
from vllm.poc.protocol.enums import GenerateResultStatus
from vllm.poc.protocol.schemas import (
    ArtifactSchema,
    GenerateCompletedResponseSchema,
    GenerateQueuedResponseSchema,
    GenerateResponseSchema,
    GenerateValidatedCompletedResponseSchema,
    GetGenerateResultResponseSchema,
    InitGenerateResponseSchema,
    PoCConfigSchema,
    PoCGenerationStatsSchema,
    StatusResponseSchema,
    StopResponseSchema,
)
from vllm.poc.protocol.types import Artifact, ArtifactBatchMeta
from vllm.poc.runtime.callbacks import CallbackSender
from vllm.poc.runtime.queue import GenerateJob, clear_queue, get_queue
from vllm.poc.runtime.state import PoCAppTasks, PoCGenerationStats
from vllm.poc.runtime.validation_utils import (
    build_artifacts_obj,
    build_encoding,
    validate_artifacts,
)
from vllm.poc.utils import env
from vllm.poc.v1.constants import POC_REQUEST_PRIORITY

logger = init_logger(__name__)

router = APIRouter(prefix="/api/v1/pow", tags=["PoC"])

# Typed runtime state (avoid anonymous dict[str, Any]).
_poc_tasks_typed: dict[int, PoCAppTasks] = {}


# =============================================================================
# Batch Size Calculation
# =============================================================================


def calculate_optimal_batch_size(
    engine_client, seq_len: int, safety_factor: float = 0.7
) -> int:
    """Calculate optimal batch size based on available GPU memory.

    This function is safe to use in routes.py as it doesn't affect PoC hash computation.
    It only determines how many nonces to process in parallel.

    Args:
        engine_client: vLLM engine client with GPU memory info
        seq_len: Sequence length for PoC computation
        safety_factor: Reserve this fraction of free memory (default 0.7 = 30% reserved)

    Returns:
        Optimal batch size (capped at reasonable limits)
    """
    try:
        # Get GPU memory info from engine
        model_config = engine_client.vllm_config.model_config
        cache_config = engine_client.vllm_config.cache_config

        hidden_size = model_config.get_hidden_size()
        num_layers = model_config.get_num_layers(
            engine_client.vllm_config.parallel_config
        )

        # Estimate memory per sample (in bytes).
        # PoC does not use the KV cache, but attention still needs Q/K/V and
        # intermediate activations. This heuristic intentionally overestimates
        # to stay on the safe side.
        # - Input embeddings: seq_len * hidden_size * 2 (fp16)
        # - Attention/MLP activations (rough proxy):
        #   ~2 * num_layers * seq_len * hidden_size * 2
        # - Output: seq_len * hidden_size * 2 (fp16)
        bytes_per_token = 2  # fp16
        mem_per_sample = (
            seq_len * hidden_size * bytes_per_token  # input
            + 2
            * num_layers
            * seq_len
            * hidden_size
            * bytes_per_token  # activations (proxy)
            + seq_len * hidden_size * bytes_per_token  # output
        )

        # Get available GPU memory
        gpu_memory_utilization = cache_config.gpu_memory_utilization

        # Estimate free memory (rough calculation)
        # Typical model weights for Qwen3-235B-A22B: ~90-100GB on H200, ~70GB on H100
        # We'll use a conservative estimate based on total memory
        if hasattr(cache_config, "num_gpu_blocks") and cache_config.num_gpu_blocks:
            # If cache is initialized, use that info
            block_size = cache_config.block_size
            # Rough estimate: each block uses ~512KB
            cache_memory = cache_config.num_gpu_blocks * block_size * 512
            free_memory = cache_memory * gpu_memory_utilization * safety_factor
        else:
            # Fallback: assume 140GB total for H200, 80GB for H100
            # Use env var or conservative default
            total_memory = env.POC_GPU_MEMORY_GB * 1024**3
            free_memory = total_memory * gpu_memory_utilization * safety_factor

        # Calculate batch size
        batch_size = int(free_memory / mem_per_sample)

        # Apply reasonable limits
        batch_size = max(32, min(batch_size, 512))  # Min 32, max 512

        logger.info(
            "Calculated optimal batch_size=%d "
            "(seq_len=%d, hidden_size=%d, mem_per_sample=%.1fMB, free_memory=%.1fGB)",
            batch_size,
            seq_len,
            hidden_size,
            mem_per_sample / 1024**2,
            free_memory / 1024**3,
        )

        return batch_size

    except Exception as e:
        logger.warning(
            "Failed to calculate optimal batch size: %s, using default=%s",
            e,
            env.POC_BATCH_SIZE_DEFAULT,
        )
        return env.POC_BATCH_SIZE_DEFAULT


# =============================================================================
# Request/Response Models
# =============================================================================


class PoCParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    seq_len: int
    k_dim: int = DEFAULT_K_DIM


class PoCInitGenerateRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    group_id: int = 0
    n_groups: int = 1
    batch_size: int = None  # Will use env.POC_BATCH_SIZE_DEFAULT if None
    params: PoCParamsModel
    url: str | None = None


@dataclass
class NonceIterator:
    """Iterator for nonces with multi-node and multi-group support."""

    node_id: int
    n_nodes: int
    group_id: int
    n_groups: int
    _current_x: int = 0

    def __iter__(self):
        return self

    def __next__(self) -> int:
        offset = self.node_id + self.group_id * self.n_nodes
        step = self.n_groups * self.n_nodes
        value = offset + self._current_x * step
        self._current_x += 1
        return value

    def take(self, n: int) -> list[int]:
        """Take the next n nonces."""
        return [next(self) for _ in range(n)]


class ValidationModel(BaseModel):
    artifacts: list[ArtifactSchema]


class StatTestModel(BaseModel):
    dist_threshold: float = DEFAULT_DIST_THRESHOLD
    p_mismatch: float = DEFAULT_P_MISMATCH
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD


class PoCGenerateRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    nonces: list[int]
    params: PoCParamsModel
    batch_size: int = None  # Will use env.POC_BATCH_SIZE_DEFAULT if None
    wait: bool = False
    url: str | None = None
    validation: ValidationModel | None = None
    stat_test: StatTestModel | None = None


# =============================================================================
# Helpers
# =============================================================================


async def get_engine_client(request: Request):
    engine_client = getattr(request.app.state, "engine_client", None)
    if engine_client is None:
        raise HTTPException(status_code=503, detail="Engine not available")
    return engine_client


def check_params_match(request: Request, params: PoCParamsModel):
    """Check params match deployed config. Raises 409 on mismatch."""
    serving_models = getattr(request.app.state, "openai_serving_models", None)
    if serving_models and hasattr(serving_models, "base_model_paths"):
        base_paths = serving_models.base_model_paths
        if base_paths:
            model_path = base_paths[0].model_path
            served_names = [p.name for p in base_paths]
            valid_models = {model_path} | set(served_names)
            if params.model not in valid_models:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "params mismatch",
                        "requested": {
                            "model": params.model,
                            "seq_len": params.seq_len,
                            "k_dim": params.k_dim,
                        },
                        "deployed": {
                            "model": list(valid_models),
                            "seq_len": None,
                            "k_dim": None,
                        },
                    },
                )

    deployed = getattr(request.app.state, "poc_deployed", None)
    if deployed:
        mismatches = []
        if deployed.get("model") and params.model != deployed["model"]:
            mismatches.append("model")
        if deployed.get("seq_len") and params.seq_len != deployed["seq_len"]:
            mismatches.append("seq_len")
        if deployed.get("k_dim") and params.k_dim != deployed["k_dim"]:
            mismatches.append("k_dim")

        if mismatches:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "params mismatch",
                    "fields": mismatches,
                    "requested": {
                        "model": params.model,
                        "seq_len": params.seq_len,
                        "k_dim": params.k_dim,
                    },
                    "deployed": deployed,
                },
            )


def _is_generation_active(app_id: int) -> bool:
    tasks = _poc_tasks_typed.get(app_id)
    if tasks is None:
        return False
    gen_task = tasks.gen_task
    return gen_task is not None and not gen_task.done()


def _get_api_status(app_id: int) -> StatusResponseSchema:
    tasks = _poc_tasks_typed.get(app_id)

    if tasks is None or not _is_generation_active(app_id):
        return StatusResponseSchema(status=PoCState.IDLE, config=None, stats=None)

    config = tasks.config
    stats = tasks.stats
    start_time = stats.start_time
    total_processed = stats.total_processed
    elapsed = time.time() - start_time if start_time > 0 else 0
    nonces_per_second = total_processed / elapsed if elapsed > 0 else 0

    return StatusResponseSchema(
        status=PoCState.GENERATING,
        config=PoCConfigSchema.from_config(config),
        stats=PoCGenerationStatsSchema(
            total_processed=total_processed,
            nonces_per_second=nonces_per_second,
        ),
    )


async def _cancel_poc_tasks(app_id: int):
    tasks = _poc_tasks_typed.pop(app_id, None)
    if tasks is not None:
        tasks.stop_event.set()
        if tasks.callback_task is not None:
            tasks.callback_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tasks.callback_task
        if tasks.gen_task is not None:
            tasks.gen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tasks.gen_task
        if tasks.callback_sender is not None:
            tasks.callback_sender.clear()


# =============================================================================
# PoC RPC helper
# =============================================================================


async def run_poc_request(
    engine_client,
    nonces: list[int],
    block_hash: str,
    public_key: str,
    seq_len: int,
    k_dim: int,
    timeout_sec: float = None,
) -> list[Artifact]:
    """Run PoC forward via first-class scheduler request and return artifacts."""

    if timeout_sec is None:
        timeout_sec = env.POC_GENERATE_CHUNK_TIMEOUT_SEC

    # The internal engine scheduler requires a unique request_id.
    request_id = str(uuid.uuid4())

    try:
        result = await engine_client.poc_request(
            request_id=request_id,
            block_hash=block_hash,
            public_key=public_key,
            nonces=nonces,
            seq_len=seq_len,
            k_dim=k_dim,
            timeout=timeout_sec,
            # PoC should have higher priority than normal generation.
            priority=POC_REQUEST_PRIORITY,
        )
    except TimeoutError:
        # Preserve timeout semantics for generation-loop backoff.
        logger.warning(
            "PoC request timed out (request_id=%s, block_hash=%s, nonces=%s)",
            request_id,
            block_hash,
            nonces,
        )
        return []
    except Exception:
        # On timeout or error, return empty artifacts instead of raising
        logger.exception(
            "PoC request failed (request_id=%s, block_hash=%s, nonces=%s). Exception: %s",
            request_id,
            block_hash,
            nonces,
            exc_info=True,
        )
        return []

    # if result is empty should return empty artifacts
    if not result:
        return []

    vectors_b64 = result["vectors_b64"]  # Pre-encoded base64 strings
    return build_artifacts_obj(result["nonces"], vectors_b64)


async def _compute_artifacts_chunk(
    engine_client,
    nonces: list[int],
    block_hash: str,
    public_key: str,
    seq_len: int,
    k_dim: int,
    timeout_sec: float = None,
    check_cancelled: Callable[[], bool] | None = None,
) -> list[Artifact]:
    """Compute artifacts for a chunk."""
    if check_cancelled and check_cancelled():
        raise RuntimeError("Cancelled")

    result = await run_poc_request(
        engine_client,
        nonces,
        block_hash,
        public_key,
        seq_len,
        k_dim,
        timeout_sec,
    )
    return result


# =============================================================================
# Generation Loop
# =============================================================================


async def _generation_loop(
    engine_client,
    stop_event: asyncio.Event,
    callback_sender: CallbackSender | None,
    config: PoCConfig,
    stats: PoCGenerationStats,
):
    nonce_iter = NonceIterator(
        node_id=config.node_id,
        n_nodes=config.node_count,
        group_id=config.group_id,
        n_groups=config.n_groups,
    )
    batch_size = config.batch_size

    start_time = time.time()
    stats.start_time = start_time
    stats.total_processed = 0
    last_report_time = start_time

    logger.info(
        "PoC generation started (node %s/%s, group %s/%s)",
        config.node_id,
        config.node_count,
        config.group_id,
        config.n_groups,
    )
    timeout_count = 0

    try:
        while not stop_event.is_set():
            nonces = nonce_iter.take(batch_size)

            try:
                artifacts = await run_poc_request(
                    engine_client,
                    nonces,
                    config.block_hash,
                    config.public_key,
                    config.seq_len,
                    config.k_dim,
                    timeout_sec=env.POC_RPC_TIMEOUT_MS / 1000.0,
                )
                timeout_count = 0
            except TimeoutError:
                timeout_count += 1
                if timeout_count == 1 or timeout_count % 10 == 0:
                    logger.warning(
                        "PoC timed out (#%d), engine busy",
                        timeout_count,
                    )
                await asyncio.sleep(env.POC_CHAT_BUSY_BACKOFF_SEC * 2)
                continue

            if artifacts and callback_sender:
                callback_sender.add_artifacts(
                    artifacts,
                    ArtifactBatchMeta(
                        public_key=config.public_key,
                        block_hash=config.block_hash,
                        block_height=config.block_height,
                        node_id=config.node_id,
                    ),
                )

            stats.total_processed += len(nonces)

            current_time = time.time()
            if current_time - last_report_time >= 5.0:
                elapsed_min = (current_time - start_time) / 60
                rate = stats.total_processed / elapsed_min if elapsed_min > 0 else 0
                logger.info(
                    "Generated: %d nonces (%.0f/min)",
                    stats.total_processed,
                    rate,
                )
                last_report_time = current_time

    except asyncio.CancelledError:
        elapsed_min = (time.time() - start_time) / 60
        logger.info(
            "PoC stopped: %d nonces in %.2fmin",
            stats.total_processed,
            elapsed_min,
        )
    except Exception as e:
        logger.exception("PoC generation crashed: %s", e)
        raise


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("/init/generate")
async def init_generate(
    request: Request, body: PoCInitGenerateRequest
) -> InitGenerateResponseSchema:
    logger.info(
        "PoC /init/generate: block_hash=%s, block_height=%s, public_key=%s, "
        "node_id=%s, node_count=%s, group_id=%s, n_groups=%s, batch_size=%s, "
        "params=%s, url=%s",
        body.block_hash,
        body.block_height,
        body.public_key,
        body.node_id,
        body.node_count,
        body.group_id,
        body.n_groups,
        body.batch_size,
        body.params,
        body.url,
    )
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)

    app_id = id(request.app)

    if _is_generation_active(app_id):
        raise HTTPException(status_code=409, detail="Already generating")

    await _cancel_poc_tasks(app_id)

    # Auto-calculate batch_size if using default or None
    batch_size = body.batch_size or env.POC_BATCH_SIZE_DEFAULT
    if env.POC_AUTO_BATCH_SIZE_DEFAULT:
        batch_size = calculate_optimal_batch_size(engine_client, body.params.seq_len)
        logger.info(
            "Auto-calculated batch_size: %d (requested: %s)",
            batch_size,
            body.batch_size,
        )

    config = PoCConfig(
        block_hash=body.block_hash,
        block_height=body.block_height,
        public_key=body.public_key,
        node_id=body.node_id,
        node_count=body.node_count,
        batch_size=batch_size,
        seq_len=body.params.seq_len,
        k_dim=body.params.k_dim,
        callback_url=body.url,
        group_id=body.group_id,
        n_groups=body.n_groups,
    )

    stats = PoCGenerationStats()
    stop_event = asyncio.Event()

    callback_sender = None
    callback_task = None
    if body.url:
        callback_sender = CallbackSender(body.url, stop_event, body.params.k_dim)
        callback_task = asyncio.create_task(callback_sender.run())

    gen_task = asyncio.create_task(
        _generation_loop(engine_client, stop_event, callback_sender, config, stats)
    )

    _poc_tasks_typed[app_id] = PoCAppTasks(
        gen_task=gen_task,
        callback_task=callback_task,
        callback_sender=callback_sender,
        stop_event=stop_event,
        config=config,
        stats=stats,
    )

    return InitGenerateResponseSchema()


@router.post("/generate")
async def generate(
    request: Request,
    body: PoCGenerateRequest,
) -> GenerateResponseSchema:
    logger.info(
        "PoC /generate: block_hash=%s, block_height=%s, public_key=%s, node_id=%s, "
        "node_count=%s, nonces=%s, params=%s, batch_size=%s, wait=%s, url=%s, "
        "validation=%s, stat_test=%s",
        body.block_hash,
        body.block_height,
        body.public_key,
        body.node_id,
        body.node_count,
        body.nonces,
        body.params,
        body.batch_size,
        body.wait,
        body.url,
        body.validation,
        body.stat_test,
    )
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)

    app_id = id(request.app)

    if body.validation:
        validation_nonces = set(a.nonce for a in body.validation.artifacts)
        if validation_nonces != set(body.nonces):
            raise HTTPException(
                status_code=400,
                detail="validation.artifacts nonces must match nonces field",
            )

    validation_map = (
        {a.nonce: a.vector_b64 for a in body.validation.artifacts}
        if body.validation
        else None
    )
    stat_test = body.stat_test or StatTestModel()

    # Auto-calculate batch_size if using default or None
    batch_size = body.batch_size or env.POC_BATCH_SIZE_DEFAULT
    if batch_size == env.POC_BATCH_SIZE_DEFAULT:
        batch_size = calculate_optimal_batch_size(engine_client, body.params.seq_len)
        logger.info(
            "Auto-calculated batch_size: %d (requested: %s)",
            batch_size,
            body.batch_size,
        )

    if not body.wait:
        queue = get_queue()
        queue.set_generation_active_check(_is_generation_active)

        if queue.queued_nonces + len(body.nonces) > env.POC_MAX_QUEUED_NONCES:
            detail = (
                f"Queue full: {queue.queued_nonces} nonces queued, "
                f"limit is {env.POC_MAX_QUEUED_NONCES}"
            )
            raise HTTPException(
                status_code=429,
                detail=detail,
            )

        job = GenerateJob(
            request_id=str(uuid.uuid4()),
            engine_client=engine_client,
            app_id=app_id,
            block_hash=body.block_hash,
            block_height=body.block_height,
            public_key=body.public_key,
            node_id=body.node_id,
            node_count=body.node_count,
            nonces=body.nonces,
            seq_len=body.params.seq_len,
            k_dim=body.params.k_dim,
            batch_size=batch_size,
            validation_artifacts=validation_map,
            stat_test_dist_threshold=stat_test.dist_threshold,
            stat_test_p_mismatch=stat_test.p_mismatch,
            stat_test_fraud_threshold=stat_test.fraud_threshold,
            callback_url=body.url,
        )

        request_id = await queue.enqueue(job)
        if request_id is None:
            detail = (
                f"Queue full: {queue.queued_nonces} nonces queued, "
                f"limit is {env.POC_MAX_QUEUED_NONCES}"
            )
            raise HTTPException(
                status_code=429,
                detail=detail,
            )

        await queue.ensure_worker_running(engine_client, app_id)

        return GenerateQueuedResponseSchema(
            request_id=request_id,
            queued_count=len(body.nonces),
        )

    while _is_generation_active(app_id):
        await asyncio.sleep(0.1)

    total_nonces = len(body.nonces)
    n_chunks = (total_nonces + batch_size - 1) // batch_size
    logger.info(
        "PoC /generate: %d nonces, batch_size=%d, chunks=%d",
        total_nonces,
        batch_size,
        n_chunks,
    )

    start_time = time.time()
    computed_artifacts: list[Artifact] = []

    for i in range(0, total_nonces, batch_size):
        chunk = body.nonces[i : i + batch_size]
        chunk_idx = i // batch_size

        def check_cancelled():
            return False

        while _is_generation_active(app_id):
            await asyncio.sleep(0.1)

        try:
            artifacts = await _compute_artifacts_chunk(
                engine_client,
                chunk,
                body.block_hash,
                body.public_key,
                body.params.seq_len,
                body.params.k_dim,
                env.POC_GENERATE_CHUNK_TIMEOUT_SEC,
                check_cancelled,
            )
            computed_artifacts.extend(artifacts)
            logger.debug(
                "PoC /generate: chunk %d/%d done (%d nonces)",
                chunk_idx + 1,
                n_chunks,
                len(chunk),
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

    elapsed = time.time() - start_time
    rate = total_nonces / elapsed if elapsed > 0 else 0
    logger.info(
        "PoC /generate completed: %d nonces in %.2fs (%.0f/s)",
        total_nonces,
        elapsed,
        rate,
    )

    if not body.validation:
        return GenerateCompletedResponseSchema(
            request_id=str(uuid.uuid4()),
            artifacts=computed_artifacts,
            encoding=build_encoding(body.params.k_dim),
        )
    validation = validate_artifacts(
        computed_artifacts,
        validation_map or {},
        dist_threshold=stat_test.dist_threshold,
        p_mismatch=stat_test.p_mismatch,
        fraud_threshold=stat_test.fraud_threshold,
    )
    return GenerateValidatedCompletedResponseSchema(
        request_id=str(uuid.uuid4()),
        n_total=validation.n_total,
        n_mismatch=validation.n_mismatch,
        mismatch_nonces=validation.mismatch_nonces,
        p_value=validation.p_value,
        fraud_detected=validation.fraud_detected,
    )


@router.get("/generate/{request_id}")
async def get_generate_result(
    request: Request,
    request_id: str,
) -> GetGenerateResultResponseSchema:
    queue = get_queue()
    record = queue.get_result(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")

    payload = record.result if record.status == GenerateResultStatus.COMPLETED else None
    error = record.error if record.status == GenerateResultStatus.FAILED else None
    return GetGenerateResultResponseSchema(
        status=record.status,
        request_id=request_id,
        payload=payload,
        error=error,
    )


@router.get("/status")
async def get_status(request: Request) -> StatusResponseSchema:
    return _get_api_status(id(request.app))


@router.post("/stop")
async def stop_round(request: Request) -> StopResponseSchema:
    app_id = id(request.app)

    await _cancel_poc_tasks(app_id)
    await clear_queue()

    return StopResponseSchema()
