# SPDX-License-Identifier: Apache-2.0
"""Seeded selection of capture sites for the PoC discrete fingerprint.

The *which positions / which layers do we capture* decision must be fixed by
the chain seed, not by the prover, so a validator regenerates the identical
site set and the prover cannot cherry-pick easy decisions (architecture.md
Sections 5.1 and 5.5).

These functions reuse the same seed derivation
(:func:`vllm.poc.gpu_random._seed_from_string`) and the same murmur3-topk
selection idea as :func:`vllm.poc.gpu_random.random_pick_indices`, but on the
CPU in pure Python so they are cheap and trivially unit-testable: pick ``k``
indices out of ``n`` by hashing each candidate index with a per-nonce seed and
taking the ``k`` smallest hashes (deterministic, anti-cherry-pick).
"""
from vllm.poc.gpu_random import _murmur3_32_scalar, _seed_from_string


def _seed_for(block_hash: str, public_key: str, nonce: int, salt: str) -> int:
    """Derive a 32-bit selection seed for one nonce and one selection axis."""
    return _seed_from_string(f"{block_hash}_{public_key}_nonce_{nonce}_{salt}")


def _pick_smallest_hash_indices(seed: int, population: int, count: int) -> list[int]:
    """Pick ``count`` of ``population`` indices by smallest murmur3 hash.

    Mirrors the ``topk(-scores)`` selection in
    :func:`vllm.poc.gpu_random.random_pick_indices`: every candidate index is
    hashed with ``seed`` and the ``count`` indices with the smallest hash win.
    Ties (equal hashes) break by smaller index, which keeps the result fully
    deterministic across machines. Sorted ascending for a canonical order.
    """
    if count <= 0:
        return []
    take = min(count, population)
    scored = [
        (_murmur3_32_scalar(index, seed), index) for index in range(population)
    ]
    scored.sort(key=lambda hash_and_index: (hash_and_index[0], hash_and_index[1]))
    chosen = [index for _hash, index in scored[:take]]
    chosen.sort()
    return chosen


def pick_routing_sites(
    block_hash: str,
    public_key: str,
    nonce: int,
    num_moe_layers: int,
    seq_len: int,
    n_layers_sample: int,
    n_positions_sample: int,
) -> list[tuple[int, int]]:
    """Pick the ``(layer_idx, position_idx)`` routing sites for one nonce.

    Layers and positions are chosen independently, each by its own seeded
    murmur3-topk pick, then combined as a full grid. The result is the canonical
    routing site set the capture path keeps and the validator regenerates.

    Returns:
        Sorted list of ``(layer_idx, position_idx)`` pairs.
    """
    layer_seed = _seed_for(block_hash, public_key, nonce, "route_layers")
    position_seed = _seed_for(block_hash, public_key, nonce, "route_positions")
    layers = _pick_smallest_hash_indices(layer_seed, num_moe_layers, n_layers_sample)
    positions = _pick_smallest_hash_indices(
        position_seed, seq_len, n_positions_sample
    )
    sites = [
        (layer_idx, position_idx)
        for layer_idx in layers
        for position_idx in positions
    ]
    sites.sort()
    return sites


def pick_logit_positions(
    block_hash: str,
    public_key: str,
    nonce: int,
    seq_len: int,
    n_positions: int,
) -> list[int]:
    """Pick the seeded positions at which to capture LM-head logit decisions.

    Logit capture costs one LM-head matmul per position (architecture.md
    Section 5.4), so this set is deliberately small. Returns a sorted list of
    position indices.
    """
    position_seed = _seed_for(block_hash, public_key, nonce, "logit_positions")
    return _pick_smallest_hash_indices(position_seed, seq_len, n_positions)
