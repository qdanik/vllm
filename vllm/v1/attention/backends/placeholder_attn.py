# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec


@dataclass
class PlaceholderAttentionMetadata(AttentionMetadata):
    """A minimal metadata object for testing/plugin wiring."""


class PlaceholderAttentionMetadataBuilder(
    AttentionMetadataBuilder[PlaceholderAttentionMetadata]
):

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> PlaceholderAttentionMetadata:
        return PlaceholderAttentionMetadata()


class PlaceholderAttentionImpl(AttentionImpl[PlaceholderAttentionMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = "decoder",
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: PlaceholderAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise RuntimeError(
            "PlaceholderAttentionBackend is for testing/plugin wiring only."
        )


class PlaceholderAttentionBackend(AttentionBackend):
    """A no-op attention backend used by plugin tests."""

    @staticmethod
    def get_name() -> str:
        return "PLACEHOLDER"

    @staticmethod
    def get_impl_cls() -> type[PlaceholderAttentionImpl]:
        return PlaceholderAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[PlaceholderAttentionMetadataBuilder]:
        return PlaceholderAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Match the conventional (num_blocks, 2, block_size, num_kv_heads, head_size)
        # layout used by most CUDA backends.
        return (num_blocks, 2, block_size, num_kv_heads, head_size)
