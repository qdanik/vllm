"""PoC model runner - simplified forward pass for vLLM v0.15.1

Uses direct_qkv=True in FlashAttentionMetadata so that attention calls
flash_attn_varlen_func with raw Q/K/V tensors (no KV cache), matching
v0.9.1 prefill path for bit-exact reproducibility.

PoC and normal inference are fully independent — neither blocks the other.

OOM Safety:
    - PoC thread catches torch.cuda.OutOfMemoryError
    - Error stored in AsyncPoCWorker, converted to FinishReason.ERROR
    - Main scheduler loop never crashes from PoC failures
"""

import base64
import time
from typing import Any

import torch

from vllm.attention.layer import Attention
from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.poc.core.transforms import (
    apply_haar_rotation,
    generate_inputs,
    random_pick_indices,
)
from vllm.poc.inference.layer_hooks import LayerHouseholderHook, poc_forward_context
from vllm.poc.protocol.constants import DEFAULT_K_DIM
from vllm.poc.utils.poc_logger import init_poc_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_poc_logger(__name__)


def _create_poc_attn_context(worker, batch_size, seq_len, device):
    """Create attention metadata for PoC direct Q/K/V forward.

    Uses direct_qkv=True so FlashAttention calls flash_attn_varlen_func
    with raw Q/K/V tensors (no KV cache), matching v0.9.1 prefill path
    for bit-exact reproducibility.

    Returns:
        (attn_metadata_dict, slot_mapping_dict) — attn_metadata_dict is
        dict[str, FlashAttentionMetadata], slot_mapping_dict is empty
        (no KV cache writes needed).
    """
    vllm_config = worker.vllm_config
    forward_ctx = vllm_config.compilation_config.static_forward_context
    attn_layers = {
        name: layer for name, layer in forward_ctx.items() if isinstance(layer, Attention)
    }

    if not attn_layers:
        raise RuntimeError(
            f"No Attention layers found in static_forward_context. "
            f"Layer types: {[type(v).__name__ for v in forward_ctx.values()][:10]}"
        )

    num_tokens = batch_size * seq_len

    query_start_loc = torch.arange(0, num_tokens + 1, seq_len, dtype=torch.int32, device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    attn_metadata = FlashAttentionMetadata(
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

    attn_metadata_dict = {name: attn_metadata for name in attn_layers}
    slot_mapping_dict = {}  # Empty — skip KV cache writes

    return attn_metadata_dict, slot_mapping_dict


def _ensure_layer_hooks(worker, block_hash: str, hidden_size: int) -> None:
    """Ensure layer hooks are installed on the worker for the given block_hash.

    Caches hooks on worker._poc_layer_hooks. If block_hash changes, detaches
    old hooks and installs new ones (per-round transform changes).
    """
    model = worker.model_runner.model
    device = worker.device

    existing_hook = getattr(worker, "_poc_layer_hooks", None)

    if existing_hook is not None:
        if existing_hook.block_hash == block_hash:
            return
        existing_hook.detach()

    hook = LayerHouseholderHook(model, block_hash, device, hidden_size)
    hook._setup(model, block_hash, device, hidden_size)
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
    """Execute PoC forward pass in a background thread + dedicated CUDA stream.

    Uses direct_qkv=True for bit-exact match with v0.9.1 prefill.

    Returns:
        Dict with nonces and vectors (FP16 numpy arrays for encoding).
        Returns None for non-last PP ranks.
    """
    t_start = time.time()
    device = worker.device
    dtype = worker.vllm_config.model_config.dtype
    model = worker.model_runner.model
    worker_vllm_config = worker.vllm_config

    tp_group = get_tp_group()
    rank = tp_group.rank_in_group

    try:
        # The scheduler/engine dispatch passes identical arguments to all TP
        # workers, so no barriers or broadcast needed.
        batch_size = len(nonces)

        # Generate embeddings on first PP rank
        intermediate_tensors = None
        inputs_embeds = None

        pp_group = get_pp_group()

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

        # Create positions tensor - optimized to avoid expand overhead
        # Instead of unsqueeze(0).expand().flatten(), directly create flattened tensor
        positions = torch.arange(batch_size * seq_len, device=device, dtype=torch.int32)

        # Ensure layer hooks are installed for this block_hash (lazy + cached)
        _ensure_layer_hooks(worker, block_hash, hidden_size)

        # Create real attention metadata
        attn_metadata_dict, slot_mapping_dict = _create_poc_attn_context(
            worker, batch_size, seq_len, device
        )

        # Unlock workspace to allow growth for MoE operations
        # (workspace may have been locked after initial warmup)
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
            # Forward pass
            with (
                set_forward_context(
                    attn_metadata_dict,
                    worker_vllm_config,
                    slot_mapping=slot_mapping_dict,
                    skip_compiled=True,
                ),
                poc_forward_context(),
            ):
                hidden_states = model(
                    input_ids=None,
                    positions=positions.flatten(),
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds.view(-1, hidden_size)
                    if inputs_embeds is not None
                    else None,
                )

        finally:
            # Re-lock workspace if it was locked before
            if was_locked and ws_manager is not None:
                ws_manager._locked = True

        # PP: send to next rank if not last
        if not pp_group.is_last_rank:
            if isinstance(hidden_states, IntermediateTensors):
                pp_group.send_tensor_dict(hidden_states.tensors, all_gather_group=get_tp_group())
            return None

        # Extract last token hidden state and compute in FP32
        hidden_states = hidden_states.view(batch_size, seq_len, -1)
        last_hidden = hidden_states[:, -1, :].float()

        # Normalize to unit sphere (in-place division)
        last_hidden.div_(last_hidden.norm(dim=-1, keepdim=True).add_(1e-8))

        # Per-nonce k-dim pick + Haar rotation
        indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, k_dim, device)

        xk = torch.gather(last_hidden, 1, indices)
        yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)

        # Normalize output vectors (in-place)
        yk.div_(yk.norm(dim=-1, keepdim=True).add_(1e-8))

        # Encode vectors as base64 FP16 strings (avoids numpy over msgpack)
        vectors_f16 = yk.half().cpu().numpy()
        vectors_b64 = [
            base64.b64encode(vectors_f16[i].tobytes()).decode("ascii") for i in range(batch_size)
        ]

        return {
            "nonces": nonces,
            "vectors_b64": vectors_b64,
        }

    except Exception as e:
        logger.exception(
            "[rank=%d] execute_poc_forward FAILED after %.2fs: %s",
            rank,
            time.time() - t_start,
            e,
        )
        raise
