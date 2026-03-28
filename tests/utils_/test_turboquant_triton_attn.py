from types import SimpleNamespace

import torch

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
)


def test_triton_turboquant_cudagraph_disabled() -> None:
    kv_cache_spec = SimpleNamespace(dtype="turboquant")

    support = TritonAttentionMetadataBuilder.get_cudagraph_support(
        vllm_config=SimpleNamespace(),
        kv_cache_spec=kv_cache_spec,
    )

    assert support is AttentionCGSupport.NEVER


def test_triton_non_turboquant_cudagraph_unchanged() -> None:
    kv_cache_spec = SimpleNamespace(dtype="float16")

    support = TritonAttentionMetadataBuilder.get_cudagraph_support(
        vllm_config=SimpleNamespace(),
        kv_cache_spec=kv_cache_spec,
    )

    assert support is AttentionCGSupport.ALWAYS


def test_decode_turboquant_cache_uses_requested_output_dtype(monkeypatch) -> None:
    impl = TritonAttentionImpl(
        num_heads=1,
        head_size=4,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="turboquant",
    )

    decode_calls: list[torch.dtype] = []

    def fake_decode(indices, norms, pi, codebook, output_dtype=torch.bfloat16):
        decode_calls.append(output_dtype)
        shape = (indices.shape[0], indices.shape[1], indices.shape[2])
        return torch.zeros(shape, dtype=output_dtype, device=indices.device)

    import vllm.v1.attention.ops.triton_turboquant as tq_ops

    monkeypatch.setattr(tq_ops, "turboquant_decode", fake_decode)

    state = SimpleNamespace(
        head_size=4,
        normal_size=4,
        config=SimpleNamespace(bit_width=4),
        Pi=torch.empty(4, 4),
        codebook=torch.empty(16),
        normal_idx=None,
        outlier_idx=None,
    )
    layer = SimpleNamespace(_tq_k_state=state, _tq_v_state=state)

    # slot_bytes = packed_bytes(2) + norm_bytes(2)
    key_cache = torch.zeros((1, 1, 1, 4), dtype=torch.uint8)
    value_cache = torch.zeros((1, 1, 1, 4), dtype=torch.uint8)
    block_table = torch.zeros((1, 1), dtype=torch.int32)

    decoded_k, decoded_v, new_block_table = impl._decode_turboquant_cache(
        key_cache,
        value_cache,
        layer,
        block_table,
        output_dtype=torch.float16,
    )

    assert decode_calls == [torch.float16, torch.float16]
    assert decoded_k.dtype == torch.float16
    assert decoded_v.dtype == torch.float16
    assert torch.equal(new_block_table, torch.zeros_like(block_table))