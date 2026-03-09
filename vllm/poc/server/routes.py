"""PoC API routes for vLLM server (scheduler-native).

PoC is submitted as a first-class v1 scheduler request kind and is mixed-batched
with chat under the same token budget.
"""

import asyncio

from fastapi import APIRouter, HTTPException, Request

import vllm.poc.env as env
from vllm.poc._log import init_poc_logger
from vllm.poc.server.callbacks import CallbackSender
from vllm.poc.server.compute import (
    compute_artifacts_pipelined,
    generation_loop,
)
from vllm.poc.server.models import (
    Artifact,
    GenerateResultStatus,
    PoCAppTasks,
    PoCConfig,
    PoCGenerateRequest,
    PoCGenerationStats,
    PoCInitGenerateRequest,
    StatTestModel,
)
from vllm.poc.server.queue import GenerateJob, clear_queue, get_queue
from vllm.poc.server.schemas import (
    GenerateCompletedResponseSchema,
    GenerateQueuedResponseSchema,
    GenerateResponseSchema,
    GenerateValidatedCompletedResponseSchema,
    GetGenerateResultResponseSchema,
    InitGenerateResponseSchema,
    StatusResponseSchema,
    StopResponseSchema,
)
from vllm.poc.server.state import (
    _poc_tasks_typed,
    _sprint_tasks_typed,
    cancel_poc_tasks,
    cancel_sprint_tasks,
    check_params_match,
    generate_request_id,
    get_api_sprint_status,
    get_api_status,
    get_engine_client,
    is_generation_active,
    is_sprint_active,
)
from vllm.poc.server.validation import build_encoding, validate_artifacts

logger = init_poc_logger(__name__)

router = APIRouter(prefix="/api/v1/pow", tags=["PoC"])


@router.post("/sprint")
async def sprint_generate(
    request: Request, body: PoCInitGenerateRequest
) -> InitGenerateResponseSchema:
    """Continuous PoC generation loop with callback results, bypassing the scheduler queue."""
    logger.info(
        "/sprint: block_hash=%s, block_height=%s, public_key=%s, "
        "node_id=%s, node_count=%s, group_id=%s, n_groups=%s, "
        "batch_size=%s, params=%s, url=%s",
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

    if is_generation_active(app_id):
        raise HTTPException(status_code=409, detail="Already generating")

    await cancel_poc_tasks(app_id)

    config = PoCConfig(
        block_hash=body.block_hash,
        block_height=body.block_height,
        public_key=body.public_key,
        node_id=body.node_id,
        node_count=body.node_count,
        seq_len=body.params.seq_len,
        k_dim=body.params.k_dim,
        callback_url=body.url,
        group_id=body.group_id,
        n_groups=body.n_groups,
        batch_size=body.batch_size,
    )

    stats = PoCGenerationStats()
    stop_event = asyncio.Event()

    callback_sender = None
    callback_task = None
    if body.url:
        callback_sender = CallbackSender(
            callback_url=body.url,
            stop_event=stop_event,
            k_dim=body.params.k_dim,
        )
        callback_task = asyncio.create_task(callback_sender.run())

    gen_task = asyncio.create_task(
        generation_loop(
            engine_client=engine_client,
            stop_event=stop_event,
            callback_sender=callback_sender,
            config=config,
            stats=stats,
        )
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


@router.post("/init/generate")
async def init_generate(
    request: Request,
    body: PoCInitGenerateRequest,
) -> InitGenerateResponseSchema:
    """v0.9.1 flow"""
    logger.info(
        "/init/generate: block_hash=%s, block_height=%s, public_key=%s, "
        "node_id=%s, node_count=%s, group_id=%s, n_groups=%s, "
        "batch_size=%s, params=%s, url=%s",
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

    if is_sprint_active(app_id):
        raise HTTPException(status_code=409, detail="Sprint already running")

    await cancel_sprint_tasks(app_id)

    config = PoCConfig(
        block_hash=body.block_hash,
        block_height=body.block_height,
        public_key=body.public_key,
        node_id=body.node_id,
        node_count=body.node_count,
        seq_len=body.params.seq_len,
        k_dim=body.params.k_dim,
        callback_url=body.url,
        group_id=body.group_id,
        n_groups=body.n_groups,
        batch_size=body.batch_size,
    )

    stats = PoCGenerationStats()
    stop_event = asyncio.Event()

    callback_sender = None
    callback_task = None
    if body.url:
        callback_sender = CallbackSender(
            callback_url=body.url,
            stop_event=stop_event,
            k_dim=body.params.k_dim,
        )
        callback_task = asyncio.create_task(callback_sender.run())

    from vllm.poc.server.sprint import sprint_generation_loop

    gen_task = asyncio.create_task(
        sprint_generation_loop(
            engine_client=engine_client,
            stop_event=stop_event,
            callback_sender=callback_sender,
            config=config,
            stats=stats,
        )
    )

    _sprint_tasks_typed[app_id] = PoCAppTasks(
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
        "/generate: block_hash=%s, block_height=%s, public_key=%s, node_id=%s, "
        "node_count=%s, nonces=%s, params=%s, wait=%s, url=%s, "
        "validation=%s, stat_test=%s",
        body.block_hash,
        body.block_height,
        body.public_key,
        body.node_id,
        body.node_count,
        body.nonces,
        body.params,
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

    if not body.wait:
        queue = get_queue()
        queue.set_generation_active_check(is_generation_active)

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
            request_id=generate_request_id(),
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

    while is_generation_active(app_id):
        await asyncio.sleep(0.1)

    total_nonces = len(body.nonces)
    logger.info("/generate: %d nonces", total_nonces)

    computed_artifacts: list[Artifact] = []

    while is_generation_active(app_id):
        await asyncio.sleep(0.1)

    try:
        computed_artifacts = await compute_artifacts_pipelined(
            engine_client=engine_client,
            nonces=body.nonces,
            block_hash=body.block_hash,
            block_height=body.block_height,
            public_key=body.public_key,
            seq_len=body.params.seq_len,
            k_dim=body.params.k_dim,
            timeout_sec=env.POC_GENERATE_CHUNK_TIMEOUT_SEC,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    if not body.validation:
        return GenerateCompletedResponseSchema(
            request_id=generate_request_id(),
            artifacts=computed_artifacts,
            encoding=build_encoding(body.params.k_dim),
        )

    validation = validate_artifacts(
        computed_artifacts=computed_artifacts,
        expected_map=validation_map or {},
        dist_threshold=stat_test.dist_threshold,
        p_mismatch=stat_test.p_mismatch,
        fraud_threshold=stat_test.fraud_threshold,
        k_dim=body.params.k_dim,
    )
    return GenerateValidatedCompletedResponseSchema(
        request_id=generate_request_id(),
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


@router.get("/sprint/status")
async def get_sprint_status(request: Request) -> StatusResponseSchema:
    return get_api_status(id(request.app))


@router.get("/status")
async def get_status(request: Request) -> StatusResponseSchema:
    return get_api_sprint_status(id(request.app))


@router.post("/stop")
async def stop_round(request: Request) -> StopResponseSchema:
    app_id = id(request.app)

    await cancel_sprint_tasks(app_id)
    await cancel_poc_tasks(app_id)
    await clear_queue()

    return StopResponseSchema()
