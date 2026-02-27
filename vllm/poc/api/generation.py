import asyncio
import time

import vllm.poc.env as env
from vllm.poc.api.compute import compute_artifact
from vllm.poc.api.models import NonceIterator
from vllm.poc.constants import POC_CHAT_BUSY_BACKOFF_SEC
from vllm.poc.protocol.callbacks import CallbackSender
from vllm.poc.protocol.config import PoCConfig
from vllm.poc.protocol.state import PoCGenerationStats
from vllm.poc.protocol.types import ArtifactBatchMeta
from vllm.poc.utils.poc_logger import init_poc_logger

logger = init_poc_logger(__name__)


async def generation_loop(
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

    start_time = time.time()
    stats.start_time = start_time
    stats.total_processed = 0
    last_report_time = start_time
    last_report_total = 0

    logger.info(
        "Generation started (node %s/%s, group %s/%s)",
        config.node_id,
        config.node_count,
        config.group_id,
        config.n_groups,
    )
    timeout_count = 0
    error_count = 0
    pending_nonces: list[int] | None = None

    try:
        while not stop_event.is_set():
            nonces = (
                pending_nonces if pending_nonces is not None else nonce_iter.take(1)
            )

            try:
                artifacts = await compute_artifact(
                    engine_client,
                    nonces,
                    config.block_hash,
                    config.block_height,
                    config.public_key,
                    config.seq_len,
                    config.k_dim,
                    timeout_sec=env.POC_RPC_TIMEOUT_MS / 1000.0,
                )
                timeout_count = 0
                error_count = 0
            except TimeoutError:
                timeout_count += 1
                if timeout_count == 1 or timeout_count % 10 == 0:
                    logger.warning(
                        "Generation timed out (#%d), engine busy",
                        timeout_count,
                    )
                pending_nonces = nonces
                await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC * 2)
                continue
            except Exception:
                error_count += 1
                if error_count == 1 or error_count % 10 == 0:
                    logger.warning(
                        "Generation request failed (#%d), backing off",
                        error_count,
                    )
                pending_nonces = nonces
                await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC * 2)
                continue

            pending_nonces = None

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

            stats.total_processed += len(artifacts)

            current_time = time.time()
            if current_time - last_report_time >= 5.0:
                window_sec = current_time - last_report_time
                window_delta = stats.total_processed - last_report_total
                rate = (window_delta / (window_sec / 60.0)) if window_sec > 0 else 0
                logger.info(
                    "Generated: %d nonces (%.0f/min)",
                    stats.total_processed,
                    rate,
                )
                last_report_time = current_time
                last_report_total = stats.total_processed

    except asyncio.CancelledError:
        elapsed_min = (time.time() - start_time) / 60
        logger.info(
            "Generation stopped: %d nonces in %.2fmin",
            stats.total_processed,
            elapsed_min,
        )
    except Exception as e:
        logger.exception("Generation crashed: %s", e)
        raise
