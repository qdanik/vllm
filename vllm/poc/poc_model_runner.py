"""PoC model runner - simplified forward pass.

This mimics vLLM's /chat/completion TP synchronization:
- TP rank0 (driver) broadcasts metadata to all TP workers
- Non-driver TP workers block until they receive the broadcast
- All TP ranks then enter model forward together (NCCL collectives align)
"""
import torch
import torch.distributed as dist
from typing import List, Optional, Dict, Any

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors

from .gpu_random import (
    generate_inputs,
    random_pick_indices,
    apply_haar_rotation,
)
from .layer_hooks import LayerHouseholderHook, poc_forward_context

# Default k_dim (can be overridden per-request)
DEFAULT_K_DIM = 12


def _ensure_layer_hooks(worker, block_hash: str, hidden_size: int) -> None:
    """Ensure layer hooks are installed on the worker for the given block_hash.
    
    Caches hooks on worker._poc_layer_hooks. If block_hash changes, detaches
    old hooks and installs new ones (per-round transform changes).
    """
    model = worker.model_runner.model
    device = worker.device
    
    existing_hook = getattr(worker, '_poc_layer_hooks', None)
    
    if existing_hook is not None:
        if existing_hook.block_hash == block_hash:
            return
        existing_hook.detach()
    
    hook = LayerHouseholderHook(model, block_hash, device, hidden_size)
    hook._setup(model, block_hash, device, hidden_size)
    worker._poc_layer_hooks = hook


def _create_prefill_attn_metadata(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    attn_backend,
):
    """Create prefill attention metadata for the given backend.
    
    Uses PAD_SLOT_ID for all slots to skip KV cache writes.
    """
    num_tokens = batch_size * seq_len
    seq_lens = [seq_len] * batch_size
    
    seq_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    seq_start_loc[1:] = torch.cumsum(
        torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0
    )
    
    backend_name = attn_backend.get_name()
    
    if backend_name == "FLASHINFER":
        from vllm.v1.attention.backends.flashinfer import FlashInferMetadata
        return FlashInferMetadata(
            num_prefills=batch_size,
            num_prefill_tokens=num_tokens,
            num_decode_tokens=0,
            slot_mapping=torch.full((num_tokens,), PAD_SLOT_ID, dtype=torch.long, device=device),
            max_prefill_seq_len=seq_len,
            seq_start_loc=seq_start_loc,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            use_cuda_graph=False,
            is_profile_run=True,
        )
    else:
        # Default to FlashAttention
        # NOTE: vLLM v1 uses the "FLASH_ATTN" backend name.
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
        return FlashAttentionMetadata(
            num_prefills=batch_size,
            num_prefill_tokens=num_tokens,
            num_decode_tokens=0,
            slot_mapping=torch.full((num_tokens,), PAD_SLOT_ID, dtype=torch.long, device=device),
            seq_lens=seq_lens,
            seq_lens_tensor=torch.tensor(seq_lens, dtype=torch.int, device=device),
            max_prefill_seq_len=seq_len,
            max_decode_seq_len=0,
            query_start_loc=seq_start_loc.clone(),
            seq_start_loc=seq_start_loc,
            context_lens_tensor=torch.zeros(batch_size, dtype=torch.int, device=device),
            block_tables=torch.empty((batch_size, 0), dtype=torch.int, device=device),
            use_cuda_graph=False,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
        )


def _get_poc_attn_context(
    worker,
    batch_size: int,
    seq_len: int,
    device: torch.device,
):
    """Build (attn_metadata, slot_mappings) for a PoC prefill forward.

    vLLM v1 expects ForwardContext.attn_metadata to be a dict mapping
    attention-layer names to backend-specific metadata, and ForwardContext
    slot_mapping to be a dict mapping attention-layer names to slot mappings.

    Prefer using the model runner's built-in dummy metadata path (stable across
    vLLM internal refactors) and only fall back to a best-effort manual build.
    """
    model_runner = worker.model_runner
    num_tokens = batch_size * seq_len

    if hasattr(model_runner, "prepare_dummy_attn_metadata") and hasattr(
        model_runner, "input_buffers"
    ):
        from vllm.v1.worker.gpu.input_batch import InputBatch

        input_batch = InputBatch.make_dummy(
            num_reqs=batch_size,
            num_tokens=num_tokens,
            input_buffers=model_runner.input_buffers,
            device=device,
        )
        # Ensure deterministic IDs in case any internal paths reference them.
        input_batch.req_ids = [f"poc_req_{i}" for i in range(batch_size)]
        model_runner.prepare_dummy_attn_metadata(input_batch)
        return input_batch.attn_metadata, input_batch.slot_mappings

    # ---------------------------------------------------------------------
    # Fallback: v1 GPUModelRunner path (stable): use metadata builders.
    # ---------------------------------------------------------------------
    if hasattr(model_runner, "attn_groups") and hasattr(model_runner, "kv_cache_config"):
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec

        num_tokens = batch_size * seq_len

        # Query/seq metadata.
        seq_lens = torch.full(
            (batch_size,),
            seq_len,
            dtype=torch.int32,
            device=device,
        )
        query_start_loc_cpu = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            step=seq_len,
            dtype=torch.int32,
            device="cpu",
        )
        query_start_loc = query_start_loc_cpu.to(device, non_blocking=True)

        attn_metadata: dict[str, Any] = {}
        slot_mappings_by_layer: dict[str, torch.Tensor] = {}

        # Slot mappings: -1 skips KV cache writes.
        slot_mapping = torch.full(
            (num_tokens,),
            PAD_SLOT_ID,
            dtype=torch.int64,
            device=device,
        )

        kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups

        for gid, kv_cache_group in enumerate(kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                block_table_tensor = torch.zeros(
                    (batch_size, 1), dtype=torch.int32, device=device
                )
            else:
                # Use the model runner's own block table tensor shape for safety,
                # but clone so we don't mutate global scheduling state.
                try:
                    blk_table = model_runner.input_batch.block_table[gid]
                    block_table_tensor = blk_table.get_device_tensor(batch_size).clone()
                    block_table_tensor.fill_(0)
                except Exception:
                    block_table_tensor = torch.zeros(
                        (batch_size, 1), dtype=torch.int32, device=device
                    )

            cm = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens,
                num_reqs=batch_size,
                num_actual_tokens=num_tokens,
                max_query_len=seq_len,
                max_seq_len=seq_len,
                block_table_tensor=block_table_tensor,
                slot_mapping=slot_mapping,
                causal=True,
            )

            # Build per-attention-group metadata and map to layers.
            for group in model_runner.attn_groups[gid]:
                backend = group.backend
                builder_cls = backend.get_builder_cls()
                builder = builder_cls(
                    group.kv_cache_spec,
                    group.layer_names,
                    worker.vllm_config,
                    device,
                )
                group_md = builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=cm,
                    fast_build=False,
                )
                for layer_name in group.layer_names:
                    attn_metadata[layer_name] = group_md
                    slot_mappings_by_layer[layer_name] = slot_mapping

            # Some models reference kv_cache_group.layer_names directly.
            for layer_name in kv_cache_group.layer_names:
                slot_mappings_by_layer.setdefault(layer_name, slot_mapping)

        return attn_metadata, slot_mappings_by_layer

    raise RuntimeError(
        "PoC: unable to build attention context; unsupported worker.model_runner implementation"
    )


@torch.inference_mode()
def execute_poc_forward(
    worker,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    hidden_size: int,
    k_dim: int = DEFAULT_K_DIM,
) -> Optional[Dict[str, Any]]:
    """Execute PoC forward pass on a worker.
    
    Mimics /chat/completion TP synchronization:
    - TP rank0 broadcasts PoC metadata
    - Non-driver ranks block until broadcast received
    - All ranks enter forward together (NCCL ops align)
    
    Returns:
        Dict with nonces and vectors (FP16 numpy arrays for encoding).
        Returns None for non-last PP ranks.
    """
    device = worker.device
    dtype = worker.model_runner.model_config.dtype
    model = worker.model_runner.model
    worker_vllm_config = worker.vllm_config
    
    tp_group = get_tp_group()
    is_tp_driver = tp_group.rank_in_group == 0
    
    # =========================================================================
    # TP SYNC: Rendezvous + CPU-only gate (no NCCL)
    # =========================================================================
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
        
        if is_tp_driver:
            broadcast_tensor_dict({
                "poc_go": True,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "nonces": nonces,
                "k_dim": k_dim,
            }, src=0)
        else:
            broadcast_data = broadcast_tensor_dict(src=0)
            seq_len = int(broadcast_data["seq_len"])
            hidden_size = int(broadcast_data["hidden_size"])
            nonces = list(broadcast_data["nonces"])
            k_dim = int(broadcast_data["k_dim"])
    
    batch_size = len(nonces)
    pp_group = get_pp_group()

    # Keep PoC forwards within compile/cudagraph range.
    # This is aligned with vLLM scheduler's max_num_batched_tokens logic.
    max_num_batched_tokens = worker_vllm_config.scheduler_config.max_num_batched_tokens
    if max_num_batched_tokens is None or max_num_batched_tokens <= 0:
        max_num_batched_tokens = batch_size * seq_len
    max_batch_per_chunk = max(1, max_num_batched_tokens // seq_len)

    # =========================================================================
    # TP SYNC: Pre-forward rendezvous
    # =========================================================================
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)

    # Ensure layer hooks are installed for this block_hash (lazy + cached)
    _ensure_layer_hooks(worker, block_hash, hidden_size)

    last_hidden_chunks: list[torch.Tensor] = []

    for start_idx in range(0, batch_size, max_batch_per_chunk):
        end_idx = min(start_idx + max_batch_per_chunk, batch_size)
        chunk_nonces = nonces[start_idx:end_idx]
        chunk_batch = len(chunk_nonces)

        # Generate embeddings on first PP rank, receive intermediate tensors on others.
        if pp_group.is_first_rank:
            inputs_embeds = generate_inputs(
                block_hash,
                public_key,
                chunk_nonces,
                dim=hidden_size,
                seq_len=seq_len,
                device=device,
                dtype=dtype,
            )
            intermediate_tensors = None
        else:
            inputs_embeds = None
            intermediate_tensors = IntermediateTensors(
                pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
            )

        # Create per-chunk attention metadata and positions.
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(chunk_batch, -1)
        attn_metadata, slot_mappings = _get_poc_attn_context(
            worker,
            chunk_batch,
            seq_len,
            device,
        )

        dummy_input_ids = torch.zeros(
            chunk_batch * seq_len,
            dtype=torch.long,
            device=device,
        )

        # Forward pass with PoC context (activates layer hook transformations).
        with set_forward_context(attn_metadata, worker_vllm_config, slot_mapping=slot_mappings):
            with poc_forward_context():
                hidden_states = model(
                    input_ids=dummy_input_ids,
                    positions=positions.flatten(),
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=(
                        inputs_embeds.view(-1, hidden_size)
                        if inputs_embeds is not None
                        else None
                    ),
                )

        # PP: send to next rank if not last.
        if not pp_group.is_last_rank:
            if isinstance(hidden_states, IntermediateTensors):
                pp_group.send_tensor_dict(
                    hidden_states.tensors, all_gather_group=get_tp_group()
                )
            continue

        # Last PP rank: collect last-token hidden per chunk in FP32.
        hidden_states = hidden_states.view(chunk_batch, seq_len, -1)
        chunk_last_hidden = hidden_states[:, -1, :].float()
        last_hidden_chunks.append(chunk_last_hidden)

    if not pp_group.is_last_rank:
        return None

    # Concatenate chunk outputs for final post-processing.
    last_hidden = torch.cat(last_hidden_chunks, dim=0)
    
    # Normalize to unit sphere
    last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Per-nonce k-dim pick + Haar rotation (via Householder chain, no cuSOLVER)
    indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, k_dim, device)
    xk = torch.gather(last_hidden, 1, indices)
    yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)
    
    # Normalize output vectors
    yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Convert to FP16 for artifact encoding (compute was in FP32)
    vectors_f16 = yk.half().cpu().numpy()
    
    return {
        "nonces": nonces,
        "vectors": vectors_f16,  # FP16 numpy array, shape [batch_size, k_dim]
    }
