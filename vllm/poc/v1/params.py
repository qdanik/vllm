# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import msgspec


class PoCParams(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """PoC parameters carried through the normal vLLM scheduler path.

    One nonce == one scheduler request.
    """

    block_hash: str
    public_key: str
    block_height: int
    nonce: int
    seq_len: int
    k_dim: int = 12
    r_target: float = 1.0  # Difficulty target for proof-of-work
    return_vectors: bool = False  # Whether to return embeddings
