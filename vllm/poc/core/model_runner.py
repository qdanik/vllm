"""PoC model runner — simplified forward pass for vLLM v0.15.1.

Key properties:
- Uses FlashAttentionMetadata(direct_qkv=True) so attention calls
  flash_attn_varlen_func with raw Q/K/V tensors (no KV cache).
  This mirrors the v0.9.1 prefill path to improve bit-level reproducibility.
- PoC and normal inference are isolated via a ContextVar gate (poc_forward_context).

OOM safety (implemented by caller / worker thread):
- PoC thread catches torch.cuda.OutOfMemoryError
- Error stored and translated to FinishReason.ERROR
- Scheduler loop stays alive

⚠️ CONSENSUS / SAFETY NOTES ⚠️
- This file is performance-sensitive.
- Determinism across heterogeneous hardware is not guaranteed by CUDA alone.
  We minimize sources of nondeterminism and provide an optional per-nonce cache.
"""

from __future__ import annotations

import base64
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from vllm.attention.layer import Attention
from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.poc.constants import DEFAULT_K_DIM
from vllm.poc.core.layer_hooks import LayerHouseholderHook, poc_forward_context
from vllm.poc.core.transforms import apply_haar_rotation, generate_inputs, random_pick_indices
from vllm.poc.utils.poc_logger import init_poc_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_poc_logger(__name__)

# Key: (block_hash, public_key, nonce, seq_len, hidden_size, k_dim)
# Value: fp16 numpy row (shape [k_dim])
_VECTOR_CACHE_MAX = 100_000
_VectorCacheKey = Tuple[str, str, int, int, int, int]
_vector_cache: "OrderedDict[_VectorCacheKey, Any]" = OrderedDict()


def _cache_key(
    *,
    block_hash: str,
    public_key: str,
    nonce: int,
    seq_len: int,
    hidden_size: int,
    k_dim: int,
) -> _VectorCacheKey:
    return (block_hash, public_key, int(nonce), int(seq_len), int(hidden_size), int(k_dim))


def _cache_get(key: _VectorCacheKey):
    vec = _vector_cache.get(key)
    if vec is None:
        return None
    _vector_cache.move_to_end(key)
    return vec


def _cache_put(key: _VectorCacheKey, vec) -> None:
    _vector_cache[key] = vec
    _vector_cache.move_to_end(key)
    while len(_vector_cache) > _VECTOR_CACHE_MAX:
        _vector_cache.popitem(last=False)


def _create_poc_attn_context(worker, *, batch_size: int, seq_len: int, device: torch.device):
    """Create attention metadata for PoC direct Q/K/V forward.

    Returns:
        (attn_metadata_dict, slot_mapping_dict)
        - attn_metadata_dict: dict[layer_name -> FlashAttentionMetadata]
        - slot_mapping_dict: empty dict (skip KV cache writes)
    """
    vllm_config = worker.vllm_config
    forward_ctx = vllm_config.compilation_config.static_forward_context

    attn_layers: Dict[str, Attention] = {
        name: layer for name, layer in forward_ctx.items() if isinstance(layer, Attention)
    }
    if not attn_layers:
        layer_types = [type(v).__name__ for v in forward_ctx.values()][:10]
        raise RuntimeError(
            "No Attention layers found in static_forward_context. "
            f"Layer types: {layer_types}"
        )

    num_tokens = int(batch_size) * int(seq_len)

    query_start_loc = torch.arange(
        0,
        num_tokens + 1,
        int(seq_len),
        dtype=torch.int32,
        device=device,
    )

    seq_lens = torch.full(
        (int(batch_size),),
        int(seq_len),
        dtype=torch.int32,
        device=device,
    )

    attn_md = FlashAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=int(seq_len),
        query_start_loc=query_start_loc,
        max_seq_len=int(seq_len),
        seq_lens=seq_lens,
        block_table=torch.empty(0, dtype=torch.int32, device=device),
        slot_mapping=torch.empty(0, dtype=torch.int64, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
        direct_qkv=True,
    )

    return {name: attn_md for name in attn_layers}, {}


def _ensure_layer_hooks(worker, *, block_hash: str, hidden_size: int) -> None:
    """Install/refresh per-round layer hooks for this worker."""
    model = worker.model_runner.model
    device = worker.device

    hook: Optional[LayerHouseholderHook] = getattr(worker, "_poc_layer_hooks", None)

    # If already attached for this round, nothing to do.
    if hook is not None and getattr(hook, "block_hash", None) == block_hash and hook.num_layers > 0:
        return

    # Detach old hooks if any.
    if hook is not None:
        with torch.no_grad():
            hook.detach()

    new_hook = LayerHouseholderHook(model, block_hash, device, int(hidden_size))
    # Prefer the modern API if available.
    if hasattr(new_hook, "attach"):
        new_hook.attach()
    else:
        # Backward-compat for older hook implementation.
        new_hook._setup(model, block_hash, device, int(hidden_size))

    worker._poc_layer_hooks = new_hook


def _normalize_rows_f32(x: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    x = x.float()
    denom = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True).add(eps)
    return x / denom


def _encode_f16_row_to_b64(row_f16_numpy) -> str:
    return base64.b64encode(row_f16_numpy.tobytes()).decode("ascii")


@torch.inference_mode()
def execute_poc_forward(
    worker,
    block_hash: str,
    public_key: str,
    nonces: list[int],
    seq_len: int,
    hidden_size: int,
    k_dim: int = DEFAULT_K_DIM,
) -> Optional[dict[str, Any]]:
    """Execute PoC forward pass.

    Returns:
        - dict for last PP rank: {"nonces": [...], "vectors_b64": [...]}.
        - None for non-last PP ranks.
    """
    t0 = time.time()
    device = worker.device
    dtype = worker.vllm_config.model_config.dtype
    model = worker.model_runner.model
    vllm_config = worker.vllm_config

    tp_group = get_tp_group()
    rank = tp_group.rank_in_group

    try:
        batch_size = len(nonces)
        if batch_size <= 0:
            raise ValueError("nonces must be non-empty")
        if int(seq_len) <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        if int(hidden_size) <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
        if int(k_dim) <= 0 or int(k_dim) > int(hidden_size):
            raise ValueError(f"k_dim must be in [1, hidden_size], got k_dim={k_dim}, hidden_size={hidden_size}")

        pp_group = get_pp_group()

        inputs_embeds = None
        intermediate_tensors = None

        if pp_group.is_first_rank:
            inputs_embeds = generate_inputs(
                block_hash,
                public_key,
                nonces,
                dim=int(hidden_size),
                seq_len=int(seq_len),
                device=device,
                dtype=dtype,
            )
        else:
            intermediate_tensors = IntermediateTensors(
                pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
            )

        # Positions for prefill.
        positions = torch.arange(int(seq_len), device=device, dtype=torch.int64)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        # Install per-round hooks.
        _ensure_layer_hooks(worker, block_hash=block_hash, hidden_size=int(hidden_size))

        # Build attention metadata for direct Q/K/V.
        attn_md, slot_md = _create_poc_attn_context(
            worker,
            batch_size=batch_size,
            seq_len=int(seq_len),
            device=device,
        )

        # Workspace manager guard: avoid crashing compilation workspace.
        ws_manager = None
        was_locked = False
        try:
            ws_manager = current_workspace_manager()
            was_locked = ws_manager.is_locked()
            if was_locked:
                ws_manager._locked = False
        except AssertionError:
            # Workspace may be unavailable in some contexts.
            pass

        try:
            with (
                set_forward_context(
                    attn_md,
                    vllm_config,
                    slot_mapping=slot_md,
                    skip_compiled=True,
                ),
                poc_forward_context(),
            ):
                hidden_states = model(
                    input_ids=None,
                    positions=positions.flatten(),
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds.view(-1, int(hidden_size))
                    if inputs_embeds is not None
                    else None,
                )
        finally:
            if was_locked and ws_manager is not None:
                ws_manager._locked = True

        # Pipeline parallel: pass along intermediate tensors.
        if not pp_group.is_last_rank:
            if isinstance(hidden_states, IntermediateTensors):
                pp_group.send_tensor_dict(
                    hidden_states.tensors,
                    all_gather_group=get_tp_group(),
                )
            return None

        # Last rank: compute PoC vectors.
        hs = hidden_states.view(batch_size, int(seq_len), -1)
        last_hidden = hs[:, -1, :]
        last_hidden = _normalize_rows_f32(last_hidden)

        indices = random_pick_indices(
            block_hash,
            public_key,
            nonces,
            int(hidden_size),
            int(k_dim),
            device,
        )

        xk = torch.gather(last_hidden, dim=1, index=indices)
        yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)
        yk = _normalize_rows_f32(yk)

        # Convert to fp16 CPU for encoding.
        vectors_f16 = yk.to(dtype=torch.float16).cpu().numpy()

        vectors_b64: list[str] = []
        for i, nonce in enumerate(nonces):
            key = _cache_key(
                block_hash=block_hash,
                public_key=public_key,
                nonce=int(nonce),
                seq_len=int(seq_len),
                hidden_size=int(hidden_size),
                k_dim=int(k_dim),
            )

            cached = _cache_get(key)
            if cached is None:
                cached = vectors_f16[i]
                _cache_put(key, cached)

            vectors_b64.append(_encode_f16_row_to_b64(cached))

        return {"nonces": nonces, "vectors_b64": vectors_b64}

    except Exception as e:
        logger.exception(
            "[rank=%d] execute_poc_forward FAILED after %.2fs: %s",
            rank,
            time.time() - t0,
            e,
        )
        raise
