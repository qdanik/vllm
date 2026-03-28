from types import SimpleNamespace

import torch

from vllm.model_executor.layers.attention.attention import Attention
from vllm.poc.engine.model_runner import _create_poc_attn_context
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata


class _FakeBackend:
    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"


class _FakeTritonBackend:
    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"


def test_create_poc_attn_context_uses_negative_kv_sentinels() -> None:
    attn_layer = Attention.__new__(Attention)
    attn_layer.attn_backend = _FakeBackend
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={
                    "model.layers.0.self_attn.attn": attn_layer,
                    "not_attn": object(),
                }
            )
        )
    )

    attn_metadata_dict, slot_mapping_dict = _create_poc_attn_context(
        worker=worker,
        batch_size=2,
        seq_len=4,
        device=torch.device("cpu"),
    )

    assert list(attn_metadata_dict) == ["model.layers.0.self_attn.attn"]
    metadata = attn_metadata_dict["model.layers.0.self_attn.attn"]
    slot_mapping = slot_mapping_dict["model.layers.0.self_attn.attn"]

    assert isinstance(metadata, FlashAttentionMetadata)
    assert metadata.direct_qkv is True
    assert tuple(metadata.block_table.shape) == (2, 1)
    assert tuple(metadata.slot_mapping.shape) == (8,)
    assert torch.equal(
        metadata.block_table,
        torch.full((2, 1), -1, dtype=torch.int32),
    )
    assert torch.equal(slot_mapping, torch.full((8,), -1, dtype=torch.int64))
    assert torch.equal(metadata.slot_mapping, slot_mapping)


def test_create_poc_attn_context_uses_triton_metadata_for_triton_layers() -> None:
    attn_layer = Attention.__new__(Attention)
    attn_layer.attn_backend = _FakeTritonBackend
    attn_layer.num_heads = 8
    attn_layer.num_kv_heads = 4
    attn_layer.head_size = 96
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={
                    "model.layers.0.self_attn.attn": attn_layer,
                }
            )
        )
    )

    attn_metadata_dict, slot_mapping_dict = _create_poc_attn_context(
        worker=worker,
        batch_size=2,
        seq_len=4,
        device=torch.device("cpu"),
    )

    metadata = attn_metadata_dict["model.layers.0.self_attn.attn"]
    slot_mapping = slot_mapping_dict["model.layers.0.self_attn.attn"]

    assert isinstance(metadata, TritonAttentionMetadata)
    assert metadata.direct_qkv is True
    assert tuple(metadata.block_table.shape) == (2, 1)
    assert tuple(metadata.slot_mapping.shape) == (8,)
    assert metadata.seq_threshold_3D == 32
    assert metadata.num_par_softmax_segments == 16
    assert tuple(metadata.softmax_segm_output.shape) == (32, 8, 16, 128)
    assert tuple(metadata.softmax_segm_max.shape) == (32, 8, 16)
    assert tuple(metadata.softmax_segm_expsum.shape) == (32, 8, 16)
    assert torch.equal(slot_mapping, torch.full((8,), -1, dtype=torch.int64))
    assert torch.equal(metadata.slot_mapping, slot_mapping)