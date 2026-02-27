# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib

from vllm.poc.v1.scheduler_params import PoCSchedulerParams


def _canonical_bytes(params: PoCSchedulerParams) -> bytes:
    """Serialize identity fields into a canonical byte sequence.

    Avoids repr() because repr formatting is not a stable serialization format.
    Uses explicit UTF-8 encoding and ASCII separators.
    """
    parts = (
        str(params.block_hash).encode("utf-8"),
        str(params.public_key).encode("utf-8"),
        str(int(params.block_height)).encode("ascii"),
        str(int(params.nonce)).encode("ascii"),
        str(int(params.seq_len)).encode("ascii"),
        str(int(params.k_dim)).encode("ascii"),
    )
    return b"|".join(parts)


def poc_identity_key(params: PoCSchedulerParams) -> str:
    """Return a stable identity key for PoC requests.

    This function guarantees:
    - Deterministic output for identical inputs.
    - No dependence on Python tuple repr.
    - No hidden locale or formatting effects.
    """
    canonical = _canonical_bytes(params)
    digest = hashlib.sha256(canonical).hexdigest()
    return f"poc:{digest}"
