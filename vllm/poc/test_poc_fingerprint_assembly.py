# SPDX-License-Identifier: Apache-2.0
"""Tests for the runner-side fingerprint assembly (seeded-site filtering).

These exercise the pure glue in ``poc_model_runner`` that turns drained routing
decisions + captured logit decisions into per-nonce fingerprints, applying the
seeded routing-site filter (anti-cherry-pick) and emitting ALL surviving
decisions (no margin pre-filter).
"""
from vllm.poc.fingerprint.schema import LOGIT_KIND, ROUTING_KIND, CapturedDecision
from vllm.poc.fingerprint.site_selection import pick_routing_sites
from vllm.poc.poc_model_runner import (
    DEFAULT_N_ROUTING_LAYERS_SAMPLE,
    DEFAULT_N_ROUTING_POSITIONS_SAMPLE,
    _build_fingerprints,
    _serialize_fingerprints,
)


def _routing_decision(layer_idx, position_idx, top_expert=0, margin=1.0):
    return CapturedDecision(
        kind=ROUTING_KIND,
        site_id=[layer_idx, position_idx],
        topk_ids=[top_expert],
        margin=margin,
        scores_topk=[5.0, 4.0],
    )


def test_build_fingerprints_keeps_only_seeded_routing_sites():
    block_hash, public_key = "bh", "pk"
    nonces = [100]
    seq_len = 8
    num_moe_layers = 20

    seeded = set(
        pick_routing_sites(
            block_hash,
            public_key,
            100,
            num_moe_layers=num_moe_layers,
            seq_len=seq_len,
            n_layers_sample=DEFAULT_N_ROUTING_LAYERS_SAMPLE,
            n_positions_sample=DEFAULT_N_ROUTING_POSITIONS_SAMPLE,
        )
    )
    assert seeded, "expected a non-empty seeded site set"
    seeded_site = next(iter(seeded))
    off_site = (999, 999)  # definitely not in the seeded set

    routing_decisions = [
        (0, seeded_site[1], _routing_decision(seeded_site[0], seeded_site[1])),
        (0, off_site[1], _routing_decision(off_site[0], off_site[1])),
    ]

    fingerprints = _build_fingerprints(
        block_hash,
        public_key,
        nonces,
        seq_len,
        num_moe_layers,
        routing_decisions,
        logit_decisions_per_nonce={},
    )

    kept = fingerprints[100].decisions
    assert len(kept) == 1
    assert tuple(kept[0].site_id) == seeded_site


def test_build_fingerprints_merges_logit_decisions():
    block_hash, public_key = "bh", "pk"
    nonces = [100, 101]
    seq_len = 8
    num_moe_layers = 20

    logit_decision = CapturedDecision(
        kind=LOGIT_KIND,
        site_id=[3],
        topk_ids=[42],
        margin=2.0,
        scores_topk=[9.0, 7.0],
    )
    fingerprints = _build_fingerprints(
        block_hash,
        public_key,
        nonces,
        seq_len,
        num_moe_layers,
        routing_decisions=[],
        logit_decisions_per_nonce={1: [logit_decision]},
    )
    assert fingerprints[100].decisions == []
    assert len(fingerprints[101].decisions) == 1
    assert fingerprints[101].decisions[0].kind == LOGIT_KIND


def test_build_fingerprints_emits_low_margin_decisions():
    # No margin pre-filter: a zero-margin seeded decision must still be emitted.
    block_hash, public_key = "bh", "pk"
    nonces = [100]
    seq_len = 8
    num_moe_layers = 20
    seeded = next(
        iter(
            pick_routing_sites(
                block_hash,
                public_key,
                100,
                num_moe_layers=num_moe_layers,
                seq_len=seq_len,
                n_layers_sample=DEFAULT_N_ROUTING_LAYERS_SAMPLE,
                n_positions_sample=DEFAULT_N_ROUTING_POSITIONS_SAMPLE,
            )
        )
    )
    routing_decisions = [
        (0, seeded[1], _routing_decision(seeded[0], seeded[1], margin=0.0)),
    ]
    fingerprints = _build_fingerprints(
        block_hash,
        public_key,
        nonces,
        seq_len,
        num_moe_layers,
        routing_decisions,
        logit_decisions_per_nonce={},
    )
    assert len(fingerprints[100].decisions) == 1
    assert fingerprints[100].decisions[0].margin == 0.0


def test_serialize_fingerprints_to_plain_data():
    fingerprints = _build_fingerprints(
        "bh",
        "pk",
        [100],
        8,
        20,
        routing_decisions=[],
        logit_decisions_per_nonce={
            0: [
                CapturedDecision(
                    kind=LOGIT_KIND,
                    site_id=[1],
                    topk_ids=[5],
                    margin=1.0,
                    scores_topk=[3.0, 2.0],
                )
            ]
        },
    )
    serialized = _serialize_fingerprints(fingerprints)
    assert set(serialized.keys()) == {100}
    assert serialized[100] == [
        {
            "kind": "logit",
            "site_id": [1],
            "topk_ids": [5],
            "margin": 1.0,
            "scores_topk": [3.0, 2.0],
        }
    ]


def test_serialize_none_is_empty():
    assert _serialize_fingerprints(None) == {}
