# SPDX-License-Identifier: Apache-2.0
"""Tests for seeded capture-site selection (anti-cherry-pick determinism)."""
from vllm.poc.fingerprint.site_selection import (
    _murmur3_32_scalar,
    _pick_smallest_hash_indices,
    pick_logit_positions,
    pick_routing_sites,
)
from vllm.poc.gpu_random import _murmur3_32_scalar as gpu_scalar


def test_scalar_murmur_matches_gpu_random_export():
    # site_selection re-exports the same scalar murmur from gpu_random.
    assert _murmur3_32_scalar is gpu_scalar


def test_scalar_murmur_matches_tensor_murmur():
    import torch

    from vllm.poc.gpu_random import _murmur3_32

    seed = 123456
    keys = torch.arange(50, dtype=torch.int32)
    tensor_hashes = _murmur3_32(keys, seed).tolist()
    scalar_hashes = [_murmur3_32_scalar(int(key), seed) for key in range(50)]
    assert scalar_hashes == tensor_hashes


def test_pick_smallest_hash_indices_is_deterministic_and_sorted():
    first = _pick_smallest_hash_indices(seed=42, population=100, count=5)
    second = _pick_smallest_hash_indices(seed=42, population=100, count=5)
    assert first == second
    assert first == sorted(first)
    assert len(first) == 5
    assert all(0 <= index < 100 for index in first)


def test_pick_clamps_count_to_population():
    chosen = _pick_smallest_hash_indices(seed=1, population=3, count=10)
    assert chosen == [0, 1, 2]


def test_pick_zero_count_is_empty():
    assert _pick_smallest_hash_indices(seed=1, population=10, count=0) == []


def test_pick_routing_sites_grid_and_determinism():
    args = dict(
        block_hash="blockhashABC",
        public_key="pubkeyXYZ",
        nonce=7,
        num_moe_layers=48,
        seq_len=16,
        n_layers_sample=3,
        n_positions_sample=4,
    )
    sites_first = pick_routing_sites(**args)
    sites_second = pick_routing_sites(**args)
    assert sites_first == sites_second
    # Full grid of sampled layers x sampled positions.
    assert len(sites_first) == 3 * 4
    assert sites_first == sorted(sites_first)
    layers = {layer_idx for layer_idx, _pos in sites_first}
    positions = {pos for _layer, pos in sites_first}
    assert len(layers) == 3
    assert len(positions) == 4
    assert all(0 <= layer_idx < 48 for layer_idx in layers)
    assert all(0 <= pos < 16 for pos in positions)


def test_pick_routing_sites_differs_by_nonce():
    base = dict(
        block_hash="blockhashABC",
        public_key="pubkeyXYZ",
        num_moe_layers=48,
        seq_len=16,
        n_layers_sample=3,
        n_positions_sample=4,
    )
    nonce_a = pick_routing_sites(nonce=1, **base)
    nonce_b = pick_routing_sites(nonce=2, **base)
    assert nonce_a != nonce_b


def test_pick_logit_positions_deterministic_and_bounded():
    args = dict(
        block_hash="blockhashABC",
        public_key="pubkeyXYZ",
        nonce=3,
        seq_len=16,
        n_positions=4,
    )
    first = pick_logit_positions(**args)
    second = pick_logit_positions(**args)
    assert first == second
    assert len(first) == 4
    assert first == sorted(first)
    assert all(0 <= pos < 16 for pos in first)


def test_routing_and_logit_position_picks_use_independent_seeds():
    # The three selection axes use different salts, so their derived seeds must
    # differ (guards against a shared-seed bug that would correlate them).
    from vllm.poc.fingerprint.site_selection import _seed_for

    block_hash, public_key, nonce = "blockhashABC", "pubkeyXYZ", 3
    layer_seed = _seed_for(block_hash, public_key, nonce, "route_layers")
    route_position_seed = _seed_for(block_hash, public_key, nonce, "route_positions")
    logit_position_seed = _seed_for(block_hash, public_key, nonce, "logit_positions")
    assert len({layer_seed, route_position_seed, logit_position_seed}) == 3
