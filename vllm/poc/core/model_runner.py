"""PoC model runner — v2 (vector-cache optimization).

Notes:
- Vector cache stores the *already base64-encoded* FP16 bytes per nonce.
  On cache hits we avoid re-encoding (base64 + tobytes) on the CPU.
- Cache is a memoization of the exact output string for an identity key.
- On miss we compute exactly the same bytes as before, then store/return.
- Cache size remains bounded.
- Cached entries are treated as immutable strings.
"""

from __future__ import annotations

import base64
import time
from collections import OrderedDict
from typing import Any

import torch

from vllm.attention.layer import Attention
from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.poc.constants import DEFAULT_K_DIM
from vllm.poc.core.layer_hooks import LayerHouseholderHook, poc_forward_context
from vllm.poc.core.transforms import (
    apply_haar_rotation,
    generate_inputs,
    random_pick_indices,
)
from vllm.poc.utils.poc_logger import init_poc_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_poc_logger(__name__)


_VECTOR_CACHE_MAX = 100_000
# (block_hash, public_key, nonce, seq_len, hidden_size, k_dim)
_VectorCacheKey = tuple[str, str, int, int, int, int] 
_vector_cache_b64: OrderedDict[_VectorCacheKey, str] = OrderedDict()


def _cache_key(
    *,
    block_hash: str,
    public_key: str,
    nonce: int,
    seq_len: int,
    hidden_size: int,
    k_dim: int,
) -> _VectorCacheKey:
    return (
        block_hash,
        public_key,
        nonce,
        seq_len,
        hidden_size,
        k_dim,
    )


def _cache_get_b64(key: _VectorCacheKey) -> str | None:
    val = _vector_cache_b64.get(key)
    if val is None:
        return None
    _vector_cache_b64.move_to_end(key)
    return val


def _cache_put_b64(key: _VectorCacheKey, b64: str) -> None:
    _vector_cache_b64[key] = b64
    _vector_cache_b64.move_to_end(key)
    while len(_vector_cache_b64) > _VECTOR_CACHE_MAX:
        _vector_cache_b64.popitem(last=False)


_AttnKey = tuple[int, int, str]  # (batch_size, seq_len, device_str)
_ATTN_CACHE_MAX = 32
_attn_md_cache: OrderedDict[_AttnKey, FlashAttentionMetadata] = OrderedDict()

_POS_CACHE_MAX = 32
_pos_cache: OrderedDict[_AttnKey, torch.Tensor] = OrderedDict()


def _device_key(device: torch.device) -> str:
    return str(device)


def _get_positions(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    key: _AttnKey = (batch_size, seq_len, _device_key(device))
    positions = _pos_cache.get(key)
    if positions is not None:
        _pos_cache.move_to_end(key)
        return positions

    positions = torch.arange(seq_len, device=device, dtype=torch.int64)
    positions = positions.repeat(batch_size)

    _pos_cache[key] = positions
    _pos_cache.move_to_end(key)
    while len(_pos_cache) > _POS_CACHE_MAX:
        _pos_cache.popitem(last=False)

    return positions


def _get_attn_layer_names(worker) -> list[str]:
    names: list[str] | None = getattr(worker, "_poc_attn_layer_names", None)
    if names is not None:
        return names

    vllm_config = worker.vllm_config
    forward_ctx = vllm_config.compilation_config.static_forward_context

    names = [
        name for name, layer in forward_ctx.items() if isinstance(layer, Attention)
    ]
    if not names:
        layer_types = [type(v).__name__ for v in forward_ctx.values()][:10]
        raise RuntimeError(
            "No Attention layers found in static_forward_context. "
            f"Layer types: {layer_types}"
        )

    worker._poc_attn_layer_names = names
    return names


def _get_attn_md(worker, *, batch_size: int, seq_len: int, device: torch.device):
    key: _AttnKey = (batch_size, seq_len, _device_key(device))

    attn_md = _attn_md_cache.get(key)
    if attn_md is not None:
        _attn_md_cache.move_to_end(key)
    else:
        num_tokens = batch_size * seq_len

        query_start_loc = torch.arange(
            0, num_tokens + 1, seq_len, dtype=torch.int32, device=device
        )
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

        attn_md = FlashAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=seq_len,
            query_start_loc=query_start_loc,
            max_seq_len=seq_len,
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

        _attn_md_cache[key] = attn_md
        _attn_md_cache.move_to_end(key)
        while len(_attn_md_cache) > _ATTN_CACHE_MAX:
            _attn_md_cache.popitem(last=False)

    names = _get_attn_layer_names(worker)
    return {name: attn_md for name in names}, {}


def _ensure_layer_hooks(worker, *, block_hash: str, hidden_size: int) -> None:
    model = worker.model_runner.model
    device = worker.device

    hook: LayerHouseholderHook | None = getattr(worker, "_poc_layer_hooks", None)
    if (
        hook is not None
        and getattr(hook, "block_hash", None) == block_hash
        and hook.num_layers > 0
    ):
        return

    if hook is not None:
        hook.detach()

    new_hook = LayerHouseholderHook(model, block_hash, device, hidden_size)
    if hasattr(new_hook, "attach"):
        new_hook.attach()
    else:
        new_hook._setup(model, block_hash, device, hidden_size)

    worker._poc_layer_hooks = new_hook


def _normalize_rows_f32(x: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    x = x.float()
    denom = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True).add(eps)
    return x / denom


def _encode_np_f16_row_to_b64(row_f16_numpy) -> str:
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
) -> dict[str, Any] | None:
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

        pp_group = get_pp_group()

        inputs_embeds = None
        intermediate_tensors = None

        if pp_group.is_first_rank:
            inputs_embeds = generate_inputs(
                block_hash,
                public_key,
                nonces,
                dim=hidden_size,
                seq_len=seq_len,
                device=device,
                dtype=dtype,
            )
        else:
            intermediate_tensors = IntermediateTensors(
                pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
            )

        positions = _get_positions(batch_size, seq_len, device)

        _ensure_layer_hooks(worker, block_hash=block_hash, hidden_size=hidden_size)

        attn_md, slot_md = _get_attn_md(
            worker,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )

        ws_manager = None
        was_locked = False
        try:
            ws_manager = current_workspace_manager()
            was_locked = ws_manager.is_locked()
            if was_locked:
                ws_manager._locked = False
        except AssertionError:
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
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds.view(-1, hidden_size)
                    if inputs_embeds is not None
                    else None,
                )
        finally:
            if was_locked and ws_manager is not None:
                ws_manager._locked = True

        if not pp_group.is_last_rank:
            if isinstance(hidden_states, IntermediateTensors):
                pp_group.send_tensor_dict(
                    hidden_states.tensors,
                    all_gather_group=get_tp_group(),
                )
            return None

        hs_view = hidden_states.view(batch_size, seq_len, -1)
        last_hidden = _normalize_rows_f32(hs_view[:, -1, :])

        indices = random_pick_indices(
            block_hash,
            public_key,
            nonces,
            hidden_size,
            k_dim,
            device,
        )

        xk = torch.gather(last_hidden, dim=1, index=indices)
        yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)
        yk = _normalize_rows_f32(yk)

        vectors_f16 = yk.to(dtype=torch.float16).cpu().numpy()

        vectors_b64: list[str] = [""] * batch_size

        for i, nonce in enumerate(nonces):
            key = _cache_key(
                block_hash=block_hash,
                public_key=public_key,
                nonce=nonce,
                seq_len=seq_len,
                hidden_size=hidden_size,
                k_dim=k_dim,
            )

            b64 = _cache_get_b64(key)
            if b64 is None:
                b64 = _encode_np_f16_row_to_b64(vectors_f16[i])
                _cache_put_b64(key, b64)

            vectors_b64[i] = b64

        return {"nonces": nonces, "vectors_b64": vectors_b64}

    except Exception as e:
        logger.exception(
            "[rank=%d] execute_poc_forward FAILED after %.2fs: %s",
            rank,
            time.time() - t0,
            e,
        )
        raise
