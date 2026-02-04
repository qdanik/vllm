"""PoC model runner - simplified forward pass.

This mimics vLLM's /chat/completion TP synchronization:
- TP rank0 (driver) broadcasts metadata to all TP workers
- Non-driver TP workers block until they receive the broadcast
- All TP ranks then enter model forward together (NCCL collectives align)
"""
import os
import torch
import torch.distributed as dist
from typing import List, Optional, Dict, Any

from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.attention.utils.fa_utils import get_flash_attn_version
from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from .gpu_random import (
    generate_inputs,
    generate_target,
    random_pick_indices,
    apply_haar_rotation,
)
from .layer_hooks import LayerHouseholderHook, poc_forward_context
from .triton_kernels import (
    fused_gather_haar_rotation,
    USE_TRITON_KERNELS,
)

logger = init_logger(__name__)

# Log FA3 status at import time
_fa3_logged = False

# Default k_dim (can be overridden per-request)
DEFAULT_K_DIM = 12

# Use optimized Triton kernels for Haar rotation
USE_FUSED_HAAR = os.environ.get("POC_USE_FUSED_HAAR", "1") == "1"

# Enable timing profiling (set POC_PROFILE=1 to enable)
POC_PROFILE = os.environ.get("POC_PROFILE", "0") == "1"

# Enable detailed profiling (set POC_PROFILE_DETAILED=1 for more breakdown)
POC_PROFILE_DETAILED = os.environ.get("POC_PROFILE_DETAILED", "0") == "1"

# Timing accumulators (for profiling)
_profile_counts = {"forward": 0, "post": 0, "input_gen": 0, "model": 0}
_profile_times = {"forward": 0.0, "post": 0.0, "input_gen": 0.0, "model": 0.0}


def _log_fa3_status():
    """Log Flash Attention version status once at startup."""
    global _fa3_logged
    if _fa3_logged:
        return
    _fa3_logged = True
    
    # Log NCCL version
    try:
        import torch.distributed as _dist
        if hasattr(_dist, 'is_nccl_available') and _dist.is_nccl_available():
            try:
                # Try to get NCCL version from torch
                nccl_version = torch.cuda.nccl.version()
                logger.info(f"PoC: NCCL version {nccl_version[0]}.{nccl_version[1]}.{nccl_version[2]}")
            except Exception:
                logger.info("PoC: NCCL available (version unknown)")
    except Exception:
        pass
    
    try:
        fa_version = get_flash_attn_version()
        if fa_version == 3:
            logger.info("PoC: Using Flash Attention 3 (H100 optimized) ✓")
        elif fa_version == 2:
            logger.warning("PoC: Using Flash Attention 2 (FA3 not available)")
        else:
            logger.warning(f"PoC: Unknown Flash Attention version: {fa_version}")
    except Exception as e:
        logger.warning(f"PoC: Could not determine Flash Attention version: {e}")
    
    if USE_FUSED_HAAR and USE_TRITON_KERNELS:
        logger.info("PoC: Using fused Triton kernels for Haar rotation ✓")
    else:
        logger.info("PoC: Using standard Python implementation for Haar rotation")


# =============================================================================
# Caching for frequently allocated tensors
# =============================================================================

# Cache for attention metadata (keyed by (batch_size, seq_len, backend_name))
_attn_metadata_cache: dict = {}

# Cache for position tensors (keyed by (batch_size, seq_len, device))
_positions_cache: dict = {}


def _get_cached_positions(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Get or create cached position tensor."""
    cache_key = (batch_size, seq_len, str(device))
    if cache_key not in _positions_cache:
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        _positions_cache[cache_key] = positions.contiguous()
    return _positions_cache[cache_key]


def _get_cached_attn_metadata(batch_size: int, seq_len: int, device: torch.device, attn_backend):
    """Get or create cached attention metadata."""
    backend_name = attn_backend.get_name()
    cache_key = (batch_size, seq_len, str(device), backend_name)
    
    if cache_key not in _attn_metadata_cache:
        _attn_metadata_cache[cache_key] = _create_prefill_attn_metadata(
            batch_size, seq_len, device, attn_backend
        )
    return _attn_metadata_cache[cache_key]


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
    
    if backend_name == "XFORMERS":
        from vllm.attention.backends.xformers import XFormersMetadata
        return XFormersMetadata(
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
    elif backend_name == "FLASHINFER":
        from vllm.attention.backends.flashinfer import FlashInferMetadata
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
        from vllm.attention.backends.flash_attn import FlashAttentionMetadata
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
    # TP SYNC: broadcast_tensor_dict includes implicit sync
    # Removed explicit barrier - broadcast is sufficient for synchronization
    # =========================================================================
    if tp_group.world_size > 1:
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
    
    # Generate embeddings on first PP rank, receive intermediate tensors on others
    intermediate_tensors = None
    inputs_embeds = None
    
    pp_group = get_pp_group()
    
    # Detailed profiling: input generation
    if POC_PROFILE_DETAILED:
        import time as _time
        torch.cuda.synchronize()
        _t_input_start = _time.perf_counter()
    
    if pp_group.is_first_rank:
        inputs_embeds = generate_inputs(
            block_hash, public_key, nonces,
            dim=hidden_size, seq_len=seq_len,
            device=device, dtype=dtype,
        )
    else:
        intermediate_tensors = IntermediateTensors(
            pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
        )
    
    if POC_PROFILE_DETAILED:
        torch.cuda.synchronize()
        _profile_times["input_gen"] += _time.perf_counter() - _t_input_start
        _profile_counts["input_gen"] += 1
    
    # Create attention metadata and positions (CACHED)
    positions = _get_cached_positions(batch_size, seq_len, device)
    attn_backend = worker.model_runner.attn_backend
    attn_metadata = _get_cached_attn_metadata(batch_size, seq_len, device, attn_backend)
    
    # NOTE: Second barrier removed - broadcast_tensor_dict already synchronizes TP ranks
    # Only sync when profiling is enabled
    if POC_PROFILE:
        torch.cuda.synchronize()
    
    # Ensure layer hooks are installed for this block_hash (lazy + cached)
    _ensure_layer_hooks(worker, block_hash, hidden_size)
    
    # Start timing forward pass
    if POC_PROFILE:
        import time
        _t0 = time.perf_counter()
    
    # Detailed: time just the model forward
    if POC_PROFILE_DETAILED:
        import time as _time
        torch.cuda.synchronize()
        _t_model_start = _time.perf_counter()
    
    # Forward pass with PoC context (activates layer hook transformations)
    with set_forward_context(attn_metadata, worker_vllm_config):
        with poc_forward_context():
            hidden_states = model(
                input_ids=None,
                positions=positions.flatten(),
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds.view(-1, hidden_size) if inputs_embeds is not None else None,
            )
    
    if POC_PROFILE_DETAILED:
        torch.cuda.synchronize()
        _profile_times["model"] += _time.perf_counter() - _t_model_start
        _profile_counts["model"] += 1
    
    if POC_PROFILE:
        torch.cuda.synchronize()
        _profile_times["forward"] += time.perf_counter() - _t0
        _profile_counts["forward"] += 1
    
    # PP: send to next rank if not last
    if not pp_group.is_last_rank:
        if isinstance(hidden_states, IntermediateTensors):
            pp_group.send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group()
            )
        return None
    
    # Log FA3 status once
    _log_fa3_status()
    
    # Start timing post-processing
    if POC_PROFILE:
        import time
        _t1 = time.perf_counter()
    
    # Extract last token hidden state and compute in FP32
    # Use contiguous() to ensure memory layout is optimal for subsequent ops
    last_hidden = hidden_states.view(batch_size, seq_len, -1)[:, -1, :].float()
    
    # Normalize to unit sphere (in-place division)
    last_hidden.div_(last_hidden.norm(dim=-1, keepdim=True).add_(1e-8))
    
    # Per-nonce k-dim pick + Haar rotation (via Householder chain, no cuSOLVER)
    indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, k_dim, device)
    
    if USE_FUSED_HAAR:
        # Optimized: fused gather + Haar rotation using Triton
        yk = fused_gather_haar_rotation(
            last_hidden, indices, block_hash, public_key, nonces, device
        )
    else:
        # Original: separate gather + Python loop
        xk = torch.gather(last_hidden, 1, indices)
        yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)
    
    # Normalize output vectors (in-place)
    yk.div_(yk.norm(dim=-1, keepdim=True).add_(1e-8))
    
    if POC_PROFILE:
        torch.cuda.synchronize()
        _profile_times["post"] += time.perf_counter() - _t1
        _profile_counts["post"] += 1
        # Log every 10 batches
        if _profile_counts["forward"] % 10 == 0:
            avg_fwd = _profile_times["forward"] / max(_profile_counts["forward"], 1) * 1000
            avg_post = _profile_times["post"] / max(_profile_counts["post"], 1) * 1000
            msg = f"PoC Profile: forward={avg_fwd:.1f}ms, post={avg_post:.1f}ms (n={_profile_counts['forward']})"
            
            # Add detailed breakdown if enabled
            if POC_PROFILE_DETAILED and _profile_counts["model"] > 0:
                avg_input = _profile_times["input_gen"] / max(_profile_counts["input_gen"], 1) * 1000
                avg_model = _profile_times["model"] / max(_profile_counts["model"], 1) * 1000
                msg += f" [input={avg_input:.1f}ms, model={avg_model:.1f}ms]"
            
            logger.info(msg)
    
    # Convert to FP16 for artifact encoding
    # Use non_blocking transfer to overlap with next batch preparation
    vectors_f16 = yk.half().cpu().numpy()
    
    return {
        "nonces": nonces,
        "vectors": vectors_f16,  # FP16 numpy array, shape [batch_size, k_dim]
    }
