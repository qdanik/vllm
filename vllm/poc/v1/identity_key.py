# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib

from vllm.poc.v1.scheduler_params import PoCSchedulerParams


def poc_identity_key(params: PoCSchedulerParams) -> str:
    """Stable string key for the canonical PoC identity tuple."""

    identity_tuple = (
        params.block_hash,
        params.public_key,
        int(params.block_height),
        int(params.nonce),
        int(params.seq_len),
        int(params.k_dim),
    )
    hash_hex = hashlib.sha256(repr(identity_tuple).encode("utf-8")).hexdigest()
    return f"poc:{hash_hex}"
