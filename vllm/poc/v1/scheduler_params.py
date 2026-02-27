# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import msgspec

from vllm.poc.constants import DEFAULT_K_DIM


class PoCSchedulerParams(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Scheduler-native parameters for a single PoC nonce request."""

    block_hash: str
    public_key: str
    block_height: int
    nonce: int
    seq_len: int
    k_dim: int = DEFAULT_K_DIM
    r_target: float = 1.0  # Difficulty target for proof-of-work
    return_vectors: bool = False  # Whether to return embeddings

    @property
    def sequence_length(self) -> int:
        return int(self.seq_len)

    @property
    def projection_dimension(self) -> int:
        return int(self.k_dim)
