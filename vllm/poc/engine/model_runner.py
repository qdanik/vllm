"""Direct PoC worker forward path.

This path bypasses scheduler-native PoC requests and executes a batched PoC
forward directly on workers via ``collective_rpc``.
"""

from __future__ import annotations

import base64
import time
from collections import OrderedDict
from typing import Any

import torch

from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention.attention import Attention
from vllm.poc._log import init_poc_logger
from vllm.poc.consensus.hooks import LayerHouseholderHook, poc_forward_context
from vllm.poc.consensus.transforms import (
    apply_haar_rotation,
    generate_inputs,
    random_pick_indices,
)
from vllm.poc.constants import DEFAULT_K_DIM
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata
from vllm.v1.attention.backends.triton_attn import (
    MIN_LAUNCH_GRID_SIZE_2D,
    NUM_PAR_SOFTMAX_SEGMENTS,
    TritonAttentionMetadata,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_poc_logger(__name__)

_VECTOR_CACHE_MAX = 100000
_vector_cache: OrderedDict[tuple[str, str, int, int, int, int], Any] = OrderedDict()


def _create_flash_attention_metadata(
    *,
    num_tokens: int,
    seq_len: int,
    batch_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> FlashAttentionMetadata:
    return FlashAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=seq_len,
        query_start_loc=query_start_loc,
        max_seq_len=seq_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
        direct_qkv=True,
    )


def _create_triton_attention_metadata(
    *,
    layer: Attention,
    num_tokens: int,
    seq_len: int,
    batch_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    device: torch.device,
) -> TritonAttentionMetadata:
    num_kv_heads = max(getattr(layer, "num_kv_heads", 1), 1)
    num_heads = getattr(layer, "num_heads", num_kv_heads)
    head_size = getattr(layer, "head_size", None)
    if head_size is None:
        raise RuntimeError("PoC Triton attention metadata requires layer.head_size")

    seq_threshold_3d = max(1, MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads)
    headdim_padded = next_power_of_2(head_size)

    return TritonAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=seq_len,
        query_start_loc=query_start_loc,
        max_seq_len=seq_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        seq_threshold_3D=seq_threshold_3d,
        num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
        softmax_segm_output=torch.empty(
            (
                seq_threshold_3d,
                num_heads,
                NUM_PAR_SOFTMAX_SEGMENTS,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        ),
        softmax_segm_max=torch.empty(
            (seq_threshold_3d, num_heads, NUM_PAR_SOFTMAX_SEGMENTS),
            dtype=torch.float32,
            device=device,
        ),
        softmax_segm_expsum=torch.empty(
            (seq_threshold_3d, num_heads, NUM_PAR_SOFTMAX_SEGMENTS),
            dtype=torch.float32,
            device=device,
        ),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )


def _create_rocm_attention_metadata(
    *,
    num_tokens: int,
    seq_len: int,
    batch_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> RocmAttentionMetadata:
    return RocmAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=seq_len,
        query_start_loc=query_start_loc,
        max_seq_len=seq_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )


def _create_layer_attention_metadata(
    *,
    layer: Attention,
    num_tokens: int,
    seq_len: int,
    batch_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    device: torch.device,
) -> FlashAttentionMetadata | TritonAttentionMetadata | RocmAttentionMetadata:
    impl_module = getattr(type(getattr(layer, "impl", None)), "__module__", "")
    backend_name = None
    attn_backend = getattr(layer, "attn_backend", None)
    if attn_backend is not None:
        backend_name = attn_backend.get_name()

    if impl_module == "vllm.v1.attention.backends.rocm_aiter_fa":
        raise RuntimeError(
            "PoC direct forward does not yet support ROCm AITER Flash attention metadata"
        )
    if impl_module == "vllm.v1.attention.backends.rocm_aiter_unified_attn":
        raise RuntimeError(
            "PoC direct forward does not yet support ROCm AITER unified attention metadata"
        )

    if impl_module == "vllm.v1.attention.backends.triton_attn" or backend_name == "TRITON_ATTN":
        return _create_triton_attention_metadata(
            layer=layer,
            num_tokens=num_tokens,
            seq_len=seq_len,
            batch_size=batch_size,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            device=device,
        )

    if impl_module == "vllm.v1.attention.backends.rocm_attn" or backend_name == "ROCM_ATTN":
        return _create_rocm_attention_metadata(
            num_tokens=num_tokens,
            seq_len=seq_len,
            batch_size=batch_size,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )

    if (
        impl_module in {
            "vllm.v1.attention.backends.flash_attn",
            "vllm.v1.attention.backends.flash_attn_diffkv",
        }
        or backend_name == "FLASH_ATTN"
    ):
        return _create_flash_attention_metadata(
            num_tokens=num_tokens,
            seq_len=seq_len,
            batch_size=batch_size,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )

    raise RuntimeError(
        f"Unsupported PoC attention backend {backend_name or impl_module!r}"
    )


def _cache_get(
    block_hash: str,
    public_key: str,
    nonce: int,
    seq_len: int,
    hidden_size: int,
    k_dim: int,
):
    key = (block_hash, public_key, nonce, seq_len, hidden_size, k_dim)
    if key in _vector_cache:
        _vector_cache.move_to_end(key)
        return _vector_cache[key]
    return None


def _cache_put(
    block_hash: str,
    public_key: str,
    nonce: int,
    seq_len: int,
    hidden_size: int,
    k_dim: int,
    vector,
) -> None:
    key = (block_hash, public_key, nonce, seq_len, hidden_size, k_dim)
    _vector_cache[key] = vector
    _vector_cache.move_to_end(key)
    while len(_vector_cache) > _VECTOR_CACHE_MAX:
        _vector_cache.popitem(last=False)


def _create_poc_attn_context(
    worker,
    batch_size: int,
    seq_len: int,
    device: torch.device,
):
    """Create attention metadata for a prefill-only PoC forward."""
    forward_ctx = worker.vllm_config.compilation_config.static_forward_context
    attn_layers = {
        name: layer
        for name, layer in forward_ctx.items()
        if isinstance(layer, Attention)
    }

    if not attn_layers:
        raise RuntimeError("No Attention layers found in static_forward_context")

    num_tokens = batch_size * seq_len
    query_start_loc = torch.arange(
        0,
        num_tokens + 1,
        seq_len,
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    # Represent KV-less PoC attention explicitly with negative sentinels rather
    # than zero-sized tensors. Newer v1 attention paths assume block tables are
    # rank-2 and may reshape slot mappings; using -1 keeps them on the direct
    # Q/K/V path without ambiguous empty reshapes.
    block_table = torch.full((batch_size, 1), -1, dtype=torch.int32, device=device)
    slot_mapping = torch.full((num_tokens,), -1, dtype=torch.int64, device=device)

    return (
        {
            name: _create_layer_attention_metadata(
                layer=layer,
                num_tokens=num_tokens,
                seq_len=seq_len,
                batch_size=batch_size,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                block_table=block_table,
                slot_mapping=slot_mapping,
                device=device,
            )
            for name, layer in attn_layers.items()
        },
        {name: slot_mapping for name in attn_layers},
    )


def _ensure_layer_hooks(worker, block_hash: str, hidden_size: int) -> None:
    hook = getattr(worker, "_poc_layer_hooks", None)
    if hook is not None and hook.block_hash == block_hash:
        return
    if hook is not None:
        hook.detach()

    hook = LayerHouseholderHook(
        model=worker.model_runner.model,
        block_hash=block_hash,
        device=worker.device,
        hidden_size=hidden_size,
    )
    hook.attach()
    worker._poc_layer_hooks = hook


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
    """Execute a direct PoC forward pass on one worker."""
    t_start = time.time()
    device = worker.device
    dtype = worker.vllm_config.model_config.dtype
    model = worker.model_runner.model
    pp_group = get_pp_group()
    tp_group = get_tp_group()
    rank = tp_group.rank_in_group

    try:
        batch_size = len(nonces)
        intermediate_tensors = None
        inputs_embeds = None

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

        positions = torch.arange(seq_len, device=device, dtype=torch.int64)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        _ensure_layer_hooks(worker, block_hash, hidden_size)
        attn_metadata_dict, slot_mapping_dict = _create_poc_attn_context(
            worker,
            batch_size,
            seq_len,
            device,
        )

        was_locked = False
        ws_manager = None
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
                    attn_metadata_dict,
                    worker.vllm_config,
                    slot_mapping=slot_mapping_dict,
                    skip_compiled=True,
                ),
                poc_forward_context(),
            ):
                hidden_states = model(
                    input_ids=None,
                    positions=positions.flatten(),
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=(
                        inputs_embeds.view(-1, hidden_size)
                        if inputs_embeds is not None
                        else None
                    ),
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

        hidden_states = hidden_states.view(batch_size, seq_len, -1)
        last_hidden = hidden_states[:, -1, :].float()
        last_hidden.div_(last_hidden.norm(dim=-1, keepdim=True).add_(1e-8))

        indices = random_pick_indices(
            block_hash,
            public_key,
            nonces,
            hidden_size,
            k_dim,
            device,
        )
        xk = torch.gather(last_hidden, 1, indices)
        yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)
        yk.div_(yk.norm(dim=-1, keepdim=True).add_(1e-8))

        vectors_f16 = yk.half().cpu().numpy()
        vectors_b64: list[str] = []
        for index, nonce in enumerate(nonces):
            cached_vec = _cache_get(
                block_hash,
                public_key,
                nonce,
                seq_len,
                hidden_size,
                k_dim,
            )
            if cached_vec is None:
                cached_vec = vectors_f16[index]
                _cache_put(
                    block_hash,
                    public_key,
                    nonce,
                    seq_len,
                    hidden_size,
                    k_dim,
                    cached_vec,
                )
            vectors_b64.append(base64.b64encode(cached_vec.tobytes()).decode("ascii"))

        return {
            "nonces": nonces,
            "vectors_b64": vectors_b64,
        }
    except Exception as error:
        logger.exception(
            "[PoC][rank=%d] execute_poc_forward failed after %.2fs: %s",
            rank,
            time.time() - t_start,
            error,
        )
        raise