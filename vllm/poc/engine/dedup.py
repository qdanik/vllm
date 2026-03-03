# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-core dedup registry and identity key computation for PoC.

Combines the dedup registry (prevents duplicate in-flight PoC requests)
with the SHA-256 identity key derivation into one module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from vllm.poc.engine.params import PoCSchedulerParams

# ── Identity key ────────────────────────────────────────────────────────────


def _canonical_bytes(params: PoCSchedulerParams) -> bytes:
    """Serialize identity fields into a canonical byte sequence."""
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

    Deterministic, no dependence on Python tuple repr or locale.
    """
    canonical = _canonical_bytes(params)
    digest = hashlib.sha256(canonical).hexdigest()
    return f"poc:{digest}"


# ── Dedup registry ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PoCAcceptResult:
    accepted: bool
    canonical_request_id: str


class PoCDedupRegistry:
    """Engine-side minimal dedup/abort registry for PoC.

    If a duplicate identity is submitted while the canonical request is still
    in flight, the duplicate must be rejected quickly.
    """

    def __init__(self) -> None:
        # identity_key -> canonical_request_id (in-flight only)
        self._in_flight: dict[str, str] = {}
        # canonical_request_id -> identity_key (in-flight only)
        self._canonical_to_identity: dict[str, str] = {}
        # request_ids that were aborted (best-effort suppression)
        self._aborted: set[str] = set()

    def on_accept(
        self,
        *,
        identity_key: str,
        request_id: str,
    ) -> PoCAcceptResult:
        canonical = self._in_flight.get(identity_key)
        if canonical is None:
            self._in_flight[identity_key] = request_id
            self._canonical_to_identity[request_id] = identity_key
            return PoCAcceptResult(accepted=True, canonical_request_id=request_id)

        # Duplicate identity while in-flight: reject.
        return PoCAcceptResult(accepted=False, canonical_request_id=canonical)

    def on_abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        # Clean up in-flight entry so the identity_key can be reused.
        identity_key = self._canonical_to_identity.pop(request_id, None)
        if (
            identity_key is not None
            and self._in_flight.get(identity_key) == request_id
        ):
            self._in_flight.pop(identity_key, None)

    def on_executed_and_emitted(
        self,
        *,
        request_id: str,
    ) -> list[str]:
        identity_key = self._canonical_to_identity.pop(request_id, None)
        if identity_key is not None and self._in_flight.get(identity_key) == request_id:
            self._in_flight.pop(identity_key, None)

        self._aborted.discard(request_id)
        return []

    def is_aborted(self, request_id: str) -> bool:
        return request_id in self._aborted
