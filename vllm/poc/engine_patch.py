"""PoC Engine Patch for vLLM 0.15.1 V1 Engine.

This module patches the V1 AsyncLLM class to add poc_request support,
enabling PoC (Proof of Compute) artifact generation.

PoC Priority:
    PoC has priority over inference. When PoC generation is active, the
    chat and completion API endpoints reject new requests with 503.

    IMPORTANT: PoC's execute_poc_forward reuses KV cache blocks starting
    from block 0 (both as scratch for inputs_embeds and as the attention
    slot mapping).  If any inference request still has KV blocks allocated,
    PoC will overwrite them and permanently corrupt the model output.

    Therefore poc_request aborts all in-flight inference requests before
    issuing collective_rpc.  The API-level 503 gating prevents new
    requests from arriving while PoC is active.

Usage:
    Import this module early in the application startup to apply the patch.
"""
import asyncio
import contextlib
import os
from typing import Dict, Any, Optional, TYPE_CHECKING
from vllm.logger import init_logger

logger = init_logger(__name__)

_patched = False

# -------------------------------------------------------------------------
# Phase-0 fingerprint capture flags (architecture.md Section 5.6 / 5.7).
#
# Default OFF: when VLLM_POC_FINGERPRINT_CAPTURE is unset/0 the PoC path is
# byte-for-byte unchanged — capture_fingerprint stays False, no hook is
# installed, no extra compute runs, and no files are written. These are read
# from os.environ locally (mirroring the POC_* envs in routes.py) rather than
# added to vllm/envs.py, since they are Phase-0 calibration-only and never
# touch consensus.
# -------------------------------------------------------------------------
_FINGERPRINT_DIR_DEFAULT = "poc_fingerprint_dumps"


def _fingerprint_capture_enabled() -> bool:
    """True iff VLLM_POC_FINGERPRINT_CAPTURE is set to a truthy value."""
    raw = os.environ.get("VLLM_POC_FINGERPRINT_CAPTURE", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _fingerprint_dir() -> str:
    """Output directory for fingerprint dumps (VLLM_POC_FINGERPRINT_DIR)."""
    return os.environ.get("VLLM_POC_FINGERPRINT_DIR", _FINGERPRINT_DIR_DEFAULT)


async def poc_request(
    self,
    action: str,
    payload: dict,
    timeout_ms: int = 60000,
    capture_fingerprint: "bool | None" = None,
) -> dict:
    """Send a PoC (Proof of Compute) request to the engine.
    
    Only supports 'generate_artifacts' action. All PoC state (generation
    loop, nonce counter, stats) is managed in the API layer.
    
    Before issuing the GPU work this method aborts all in-flight inference
    requests.  This is required because execute_poc_forward writes into
    KV-cache blocks starting from block 0; if any request still holds
    those blocks the KV data is corrupted and the model produces garbage
    for the rest of its lifetime.
    
    The API-level 503 gating (chat and completion api_router.py) prevents
    new requests from arriving while PoC is active.
    
    Args:
        action: The PoC action to perform (only 'generate_artifacts' supported)
        payload: Dict containing nonces, block_hash, public_key, seq_len, k_dim
        timeout_ms: Timeout in milliseconds for the RPC call
        
    Returns:
        Dict with 'artifacts' list and optionally 'skipped' boolean
        
    Raises:
        TimeoutError: If engine doesn't respond within timeout
    """
    if action != "generate_artifacts":
        raise ValueError(f"Unknown PoC action: {action}")
    
    # Import PoC modules here to avoid circular imports
    from vllm.poc.poc_model_runner import execute_poc_forward
    from vllm.poc.data import encode_vector
    
    nonces = payload.get("nonces", [])
    block_hash = payload.get("block_hash", "")
    public_key = payload.get("public_key", "")
    seq_len = payload.get("seq_len", 256)
    k_dim = payload.get("k_dim", 12)
    poc_stronger_rng = payload.get("poc_stronger_rng", False)

    # Phase-0 fingerprint capture: default OFF. The caller may force it via the
    # capture_fingerprint argument; otherwise it follows the env flag. When
    # False the worker path is unchanged (no hook, no extra compute).
    if capture_fingerprint is None:
        capture_fingerprint = _fingerprint_capture_enabled()

    if not nonces:
        return {"artifacts": []}
    
    # Abort all in-flight inference before touching the GPU.
    # execute_poc_forward reuses KV-cache blocks from block 0, so any
    # request that still holds allocated blocks would get its KV data
    # destroyed, permanently corrupting model output.
    # The API-level 503 gating already blocks new requests, so only the
    # first batch will typically find anything to abort.
    output_processor = getattr(self, 'output_processor', None)
    if output_processor is not None and output_processor.has_unfinished_requests():
        request_ids = list(output_processor.request_states.keys())
        if request_ids:
            logger.info("PoC aborting %d in-flight inference request(s)",
                        len(request_ids))
            await self.abort(request_ids, internal=True)
            await asyncio.sleep(0.05)
    
    # Get model config for hidden_size
    # V1 engine stores config differently
    try:
        vllm_config = self.vllm_config
        hidden_size = vllm_config.model_config.get_hidden_size()
    except AttributeError:
        # Fallback - try to get from model config
        try:
            hidden_size = self.model_config.get_hidden_size()
        except Exception:
            # Default for Qwen models
            hidden_size = 8192
            logger.warning(f"Could not get hidden_size from config, using default: {hidden_size}")
    
    try:
        # Use collective_rpc to execute PoC forward on all workers
        timeout_sec = timeout_ms / 1000.0
        results = await self.collective_rpc(
            execute_poc_forward,
            timeout=timeout_sec,
            args=(
                block_hash,
                public_key,
                nonces,
                seq_len,
                hidden_size,
                k_dim,
                poc_stronger_rng,
                capture_fingerprint,
            ),
        )

        # Only the last PP rank returns a result
        result = next((r for r in results if r is not None), None)

        if result is None:
            return {"artifacts": [], "skipped": True}

        # Phase-0: serialise the captured fingerprints to a calibration dump on
        # the driver rank. Best-effort: a dump failure must never break PoC.
        if capture_fingerprint:
            _dump_fingerprints(
                self,
                result.get("fingerprints"),
                block_hash=block_hash,
                public_key=public_key,
                seq_len=seq_len,
            )

        # Convert result to artifact format
        vectors = result.get("vectors")  # FP16 numpy array
        result_nonces = result.get("nonces", nonces)

        artifacts = []
        for i, nonce in enumerate(result_nonces):
            vector_b64 = encode_vector(vectors[i])
            artifacts.append({"nonce": nonce, "vector_b64": vector_b64})

        return {"artifacts": artifacts}
        
    except asyncio.TimeoutError:
        logger.warning(f"PoC request timed out after {timeout_ms}ms")
        raise TimeoutError(f"PoC request timed out after {timeout_ms}ms")
    except Exception as e:
        logger.error(f"PoC request failed: {e}")
        return {"artifacts": [], "skipped": True}


def _detect_meta(engine) -> dict:
    """Derive the dump meta sidecar fields from the live engine config.

    All lookups are wrapped in try/except so a missing attribute on an unusual
    deployment degrades to a sentinel string instead of breaking PoC. This runs
    only when fingerprint capture is enabled.

    Detected fields:
        model_id: served model name (model_config.served_model_name) falling
            back to the model path.
        gpu:      torch.cuda.get_device_name(0) — the driver-rank GPU.
        backend:  attention backend from VLLM_ATTENTION_BACKEND, plus a
            ``+deepgemm`` suffix when VLLM_MOE_USE_DEEP_GEMM is on (the two
            knobs that most change cross-GPU numerics — architecture.md 5.6).
        tp:       parallel_config.tensor_parallel_size.
    """
    model_id = "unknown-model"
    gpu = "unknown-gpu"
    backend = "unknown-backend"
    tp = 1

    # Each lookup is defensively suppressed so a missing attribute on an unusual
    # deployment degrades to a sentinel rather than breaking PoC generation.
    with contextlib.suppress(Exception):
        model_config = engine.vllm_config.model_config
        served = getattr(model_config, "served_model_name", None)
        if isinstance(served, (list, tuple)) and served:
            model_id = str(served[0])
        elif isinstance(served, str) and served:
            model_id = served
        else:
            model_id = str(getattr(model_config, "model", model_id))

    with contextlib.suppress(Exception):
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)

    with contextlib.suppress(Exception):
        attention_backend = os.environ.get("VLLM_ATTENTION_BACKEND") or "auto"
        moe_deepgemm = os.environ.get("VLLM_MOE_USE_DEEP_GEMM", "1")
        suffix = "+deepgemm" if moe_deepgemm not in ("0", "", "false") else ""
        backend = f"{attention_backend}{suffix}"

    with contextlib.suppress(Exception):
        tp = int(engine.vllm_config.parallel_config.tensor_parallel_size)

    return {"model_id": model_id, "gpu": gpu, "backend": backend, "tp": tp}


def _dump_fingerprints(engine, fingerprints, block_hash, public_key, seq_len):
    """Write one PoC batch's captured fingerprints to a calibration dump.

    Best-effort and fully isolated: any failure here is logged and swallowed so
    fingerprint capture can never break PoC generation. The seed recorded in the
    sidecar is ``block_hash_public_key`` (the same string the seeded site
    selection derives from), so a validator can regenerate the identical sites.
    """
    if not fingerprints:
        return
    try:
        from vllm.poc.fingerprint.dump_writer import write_capture_dump
        from vllm.poc.poc_model_runner import (
            DEFAULT_FINGERPRINT_TOP_K,
        )

        meta = _detect_meta(engine)
        written = write_capture_dump(
            fingerprints,
            _fingerprint_dir(),
            model_id=meta["model_id"],
            seed=f"{block_hash}_{public_key}",
            gpu=meta["gpu"],
            backend=meta["backend"],
            tp=meta["tp"],
            seq_len=seq_len,
            block_hash=block_hash,
            signal_set=["routing", "logit"],
            top_k=DEFAULT_FINGERPRINT_TOP_K,
        )
        if written is not None:
            dump_path, _meta_path = written
            logger.info("PoC fingerprint dump appended: %s", dump_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("PoC fingerprint dump failed (ignored): %s", exc)


def apply_patch():
    """Apply the PoC patch to vLLM V1 AsyncLLM class."""
    global _patched
    
    if _patched:
        logger.debug("PoC engine patch already applied")
        return
    
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        
        # Add poc_request method to AsyncLLM
        AsyncLLM.poc_request = poc_request
        
        _patched = True
        logger.info("PoC engine patch applied successfully to AsyncLLM (V1)")
        
    except ImportError as e:
        logger.warning(f"Could not import V1 AsyncLLM, trying V0: {e}")
        
        try:
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            
            # For V0 engine, the implementation is slightly different
            AsyncLLMEngine.poc_request = poc_request
            
            _patched = True
            logger.info("PoC engine patch applied successfully to AsyncLLMEngine (V0)")
            
        except ImportError as e2:
            logger.error(f"Could not import any LLM engine: {e2}")
            raise


# Auto-apply patch when module is imported
apply_patch()
