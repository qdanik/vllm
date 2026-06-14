# SPDX-License-Identifier: Apache-2.0
"""Tests for the PoC discrete-fingerprint calibration analytics library.

These are PURE tests: no model, no GPU, no consensus. They drive the design of
``vllm/poc/fingerprint/analytics.py`` via small, hand-constructed synthetic
decision dumps (see ``architecture.md`` Section 5.1, 5.2, 5.6).
"""

import pytest

from vllm.poc.fingerprint.analytics import (
    coverage_curve,
    cross_model_divergence,
    fraud_test,
    honest_flip_rate,
    hysteresis_compare,
    load_dump,
    save_dump,
    summarize,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic decision dumps.
# ---------------------------------------------------------------------------
def make_decision(
    site_id,
    top1,
    margin,
    kind="routing",
    runner_up=999,
):
    """Build one decision dict matching the capture-side schema."""
    return {
        "kind": kind,
        "site_id": site_id,
        "topk_ids": [top1, runner_up],
        "margin": float(margin),
        "scores_topk": [1.0, 1.0 - float(margin)],
    }


# ---------------------------------------------------------------------------
# 1. JSONL round-trip.
# ---------------------------------------------------------------------------
def test_save_and_load_dump_round_trip(tmp_path):
    dump = [
        [
            make_decision([0, 0], top1=7, margin=0.9),
            make_decision([1, 3], top1=2, margin=0.1, kind="logit"),
        ],
        [
            make_decision([0, 0], top1=7, margin=0.5),
        ],
    ]
    path = tmp_path / "dump.jsonl"
    save_dump(dump, path)
    loaded = load_dump(path)
    assert loaded == dump


def test_jsonl_is_one_nonce_per_line(tmp_path):
    dump = [
        [make_decision([0, 0], top1=1, margin=0.5)],
        [make_decision([0, 0], top1=1, margin=0.5)],
        [make_decision([0, 0], top1=1, margin=0.5)],
    ]
    path = tmp_path / "dump.jsonl"
    save_dump(dump, path)
    text = path.read_text().rstrip("\n").splitlines()
    assert len(text) == 3


def test_empty_dump_round_trip(tmp_path):
    path = tmp_path / "empty.jsonl"
    save_dump([], path)
    assert load_dump(path) == []


# ---------------------------------------------------------------------------
# 2. honest_flip_rate.
# ---------------------------------------------------------------------------
def test_flip_rate_high_margin_agree_is_zero():
    # Both sides agree on every high-margin decision -> flip_rate 0.
    site = [0, 0]
    dump_a = [[make_decision(site, top1=5, margin=0.9)]]
    dump_b = [[make_decision(site, top1=5, margin=0.9)]]
    result = honest_flip_rate(dump_a, dump_b, tau=0.5)
    assert result["flip_rate"] == 0.0
    assert result["total_scored"] == 1


def test_flip_rate_all_flip_is_one():
    site = [0, 0]
    dump_a = [[make_decision(site, top1=5, margin=0.9)]]
    dump_b = [[make_decision(site, top1=8, margin=0.9)]]
    result = honest_flip_rate(dump_a, dump_b, tau=0.5)
    assert result["flip_rate"] == 1.0
    assert result["total_scored"] == 1


def test_flip_rate_uses_reference_margin_to_gate():
    # Reference (dump_b) margin is below tau -> decision is NOT scored, even
    # though the ids differ.
    site = [0, 0]
    dump_a = [[make_decision(site, top1=5, margin=0.9)]]
    dump_b = [[make_decision(site, top1=8, margin=0.1)]]
    result = honest_flip_rate(dump_a, dump_b, tau=0.5)
    assert result["total_scored"] == 0
    assert result["flip_rate"] == 0.0


def test_flip_rate_mixed_high_and_low_margin():
    # site A: high reference margin, ids differ -> scored + flip.
    # site B: high reference margin, ids agree -> scored, no flip.
    # site C: low reference margin -> not scored.
    dump_a = [
        [
            make_decision([0, 0], top1=1, margin=0.9),
            make_decision([0, 1], top1=2, margin=0.9),
            make_decision([0, 2], top1=3, margin=0.9),
        ]
    ]
    dump_b = [
        [
            make_decision([0, 0], top1=99, margin=0.8),
            make_decision([0, 1], top1=2, margin=0.8),
            make_decision([0, 2], top1=77, margin=0.05),
        ]
    ]
    result = honest_flip_rate(dump_a, dump_b, tau=0.5)
    assert result["total_scored"] == 2
    assert result["flip_rate"] == pytest.approx(0.5)


def test_flip_rate_only_intersecting_site_ids_scored():
    # site present only in A is ignored (not aligned at that site).
    dump_a = [
        [
            make_decision([0, 0], top1=1, margin=0.9),
            make_decision([0, 1], top1=2, margin=0.9),  # not in B
        ]
    ]
    dump_b = [
        [
            make_decision([0, 0], top1=1, margin=0.9),
        ]
    ]
    result = honest_flip_rate(dump_a, dump_b, tau=0.5)
    assert result["total_scored"] == 1


def test_flip_rate_mean_scored_per_nonce():
    dump_a = [
        [make_decision([0, 0], top1=1, margin=0.9),
         make_decision([0, 1], top1=2, margin=0.9)],
        [make_decision([0, 0], top1=1, margin=0.9)],
    ]
    dump_b = [
        [make_decision([0, 0], top1=1, margin=0.9),
         make_decision([0, 1], top1=2, margin=0.9)],
        [make_decision([0, 0], top1=1, margin=0.9)],
    ]
    result = honest_flip_rate(dump_a, dump_b, tau=0.5)
    # nonce0 scores 2, nonce1 scores 1 -> mean 1.5.
    assert result["scored_count_per_nonce"] == [2, 1]
    assert result["total_scored"] == 3


def test_flip_rate_signal_filter():
    dump_a = [
        [
            make_decision([0, 0], top1=1, margin=0.9, kind="routing"),
            make_decision([5], top1=2, margin=0.9, kind="logit"),
        ]
    ]
    dump_b = [
        [
            make_decision([0, 0], top1=1, margin=0.9, kind="routing"),
            make_decision([5], top1=99, margin=0.9, kind="logit"),
        ]
    ]
    routing = honest_flip_rate(dump_a, dump_b, tau=0.5, signal="routing")
    assert routing["total_scored"] == 1
    assert routing["flip_rate"] == 0.0
    logit = honest_flip_rate(dump_a, dump_b, tau=0.5, signal="logit")
    assert logit["total_scored"] == 1
    assert logit["flip_rate"] == 1.0


def test_flip_rate_empty_scored_set_is_safe():
    dump_a = [[make_decision([0, 0], top1=1, margin=0.1)]]
    dump_b = [[make_decision([0, 0], top1=1, margin=0.1)]]
    result = honest_flip_rate(dump_a, dump_b, tau=0.5)
    assert result["total_scored"] == 0
    assert result["flip_rate"] == 0.0


# ---------------------------------------------------------------------------
# 3. coverage_curve.
# ---------------------------------------------------------------------------
def test_coverage_curve_S_decreases_with_tau():
    # Margins spread across a range; raising tau prunes the scored set.
    decisions_a = [make_decision([0, i], top1=i, margin=0.1 * i) for i in range(1, 11)]
    decisions_b = [make_decision([0, i], top1=i, margin=0.1 * i) for i in range(1, 11)]
    dump_a = [decisions_a]
    dump_b = [decisions_b]
    taus = [0.0, 0.3, 0.6, 0.9]
    curve = coverage_curve(dump_a, dump_b, taus)
    scored = [point["total_scored"] for point in curve]
    # Monotone non-increasing in tau.
    assert all(scored[i] >= scored[i + 1] for i in range(len(scored) - 1))
    assert curve[0]["tau"] == 0.0
    assert scored[0] == 10


def test_coverage_curve_returns_mean_S_and_flip_rate():
    dump_a = [[make_decision([0, 0], top1=1, margin=0.9)]]
    dump_b = [[make_decision([0, 0], top1=2, margin=0.9)]]
    curve = coverage_curve(dump_a, dump_b, [0.5])
    point = curve[0]
    assert set(point) == {"tau", "flip_rate", "mean_S", "total_scored"}
    assert point["flip_rate"] == 1.0
    assert point["mean_S"] == 1.0


# ---------------------------------------------------------------------------
# 4. cross_model_divergence.
# ---------------------------------------------------------------------------
def test_divergence_disjoint_ids_is_one():
    # Different model M' -> top-1 ids never coincide -> divergence ~1.0.
    dump_m = [[make_decision([0, i], top1=i, margin=0.9) for i in range(5)]]
    dump_mprime = [[make_decision([0, i], top1=100 + i, margin=0.9) for i in range(5)]]
    result = cross_model_divergence(dump_m, dump_mprime, tau=0.5)
    assert result["divergence"] == 1.0
    assert result["scored"] == 5


def test_divergence_identical_is_zero():
    dump_m = [[make_decision([0, i], top1=i, margin=0.9) for i in range(5)]]
    dump_mprime = [[make_decision([0, i], top1=i, margin=0.9) for i in range(5)]]
    result = cross_model_divergence(dump_m, dump_mprime, tau=0.5)
    assert result["divergence"] == 0.0


def test_divergence_empty_scored_set():
    dump_m = [[make_decision([0, 0], top1=1, margin=0.1)]]
    dump_mprime = [[make_decision([0, 0], top1=2, margin=0.1)]]
    result = cross_model_divergence(dump_m, dump_mprime, tau=0.5)
    assert result["scored"] == 0
    assert result["divergence"] == 0.0


# ---------------------------------------------------------------------------
# 5. hysteresis_compare (mirrors live Section 5.2 comparison).
# ---------------------------------------------------------------------------
def test_hysteresis_scores_only_above_band():
    # Validator margins: only the ones >= tau_sel + delta_hyst are scored.
    prover = [
        [
            make_decision([0, 0], top1=1, margin=0.9),
            make_decision([0, 1], top1=2, margin=0.9),
        ]
    ]
    validator = [
        [
            make_decision([0, 0], top1=1, margin=0.55),   # below band (0.5+0.1)
            make_decision([0, 1], top1=2, margin=0.65),   # above band
        ]
    ]
    per_nonce = hysteresis_compare(prover, validator, tau_sel=0.5, delta_hyst=0.1)
    assert per_nonce == [{"n_scored": 1, "n_mismatch": 0}]


def test_hysteresis_counts_mismatch():
    prover = [[make_decision([0, 0], top1=1, margin=0.9)]]
    validator = [[make_decision([0, 0], top1=99, margin=0.9)]]
    per_nonce = hysteresis_compare(prover, validator, tau_sel=0.5, delta_hyst=0.1)
    assert per_nonce == [{"n_scored": 1, "n_mismatch": 1}]


def test_hysteresis_uses_validator_margin_not_prover():
    # Prover margin huge, validator margin below band -> not scored.
    prover = [[make_decision([0, 0], top1=1, margin=5.0)]]
    validator = [[make_decision([0, 0], top1=99, margin=0.4)]]
    per_nonce = hysteresis_compare(prover, validator, tau_sel=0.5, delta_hyst=0.1)
    assert per_nonce == [{"n_scored": 0, "n_mismatch": 0}]


def test_hysteresis_signal_filter():
    prover = [
        [
            make_decision([0, 0], top1=1, margin=0.9, kind="routing"),
            make_decision([5], top1=2, margin=0.9, kind="logit"),
        ]
    ]
    validator = [
        [
            make_decision([0, 0], top1=1, margin=0.9, kind="routing"),
            make_decision([5], top1=99, margin=0.9, kind="logit"),
        ]
    ]
    per_nonce = hysteresis_compare(
        prover, validator, tau_sel=0.5, delta_hyst=0.1, signal="routing"
    )
    assert per_nonce == [{"n_scored": 1, "n_mismatch": 0}]


# ---------------------------------------------------------------------------
# 6. fraud_test (binomtest).
# ---------------------------------------------------------------------------
def test_fraud_test_honest_not_flagged():
    # Zero mismatches over a large scored set -> clearly honest.
    result = fraud_test(n_mismatch=0, n_scored=1000, p0=0.001, fraud_threshold=0.01)
    assert result["fraud"] is False
    assert result["p_value"] == pytest.approx(1.0)


def test_fraud_test_substitution_flagged():
    # ~100% mismatch (model substitution) -> overwhelming fraud signal.
    result = fraud_test(n_mismatch=950, n_scored=1000, p0=0.001, fraud_threshold=0.01)
    assert result["fraud"] is True
    assert result["p_value"] < 1e-9


def test_fraud_test_boundary_respects_threshold():
    # A p-value just under the threshold flags; just over does not.
    flagged = fraud_test(n_mismatch=5, n_scored=1000, p0=0.001, fraud_threshold=0.01)
    not_flagged = fraud_test(
        n_mismatch=1, n_scored=1000, p0=0.001, fraud_threshold=0.01
    )
    assert (flagged["p_value"] < 0.01) == flagged["fraud"]
    assert (not_flagged["p_value"] < 0.01) == not_flagged["fraud"]


def test_fraud_test_empty_scored_set_is_not_fraud():
    result = fraud_test(n_mismatch=0, n_scored=0, p0=0.001, fraud_threshold=0.01)
    assert result["fraud"] is False
    assert result["p_value"] == 1.0


# ---------------------------------------------------------------------------
# 7. summarize.
# ---------------------------------------------------------------------------
def _spread_dump(top1_offset=0, margins=None):
    margins = margins or [0.1, 0.3, 0.5, 0.7, 0.9]
    return [
        [
            make_decision([0, i], top1=i + top1_offset, margin=margin)
            for i, margin in enumerate(margins)
        ]
    ]


def test_summarize_returns_curves_and_operating_tau():
    prover = _spread_dump()
    validator = _spread_dump()
    taus = [0.0, 0.2, 0.4, 0.6, 0.8]
    report = summarize(prover, validator, taus=taus, min_mean_S=1.0)
    assert "honest_curve" in report
    assert "suggested_tau" in report
    assert len(report["honest_curve"]) == len(taus)
    # All-agree honest dump -> flip rate is 0 everywhere; operating tau is the
    # smallest tau that still keeps mean_S above the floor.
    assert report["suggested_tau"] is not None


def test_summarize_includes_cross_model_when_provided():
    prover = _spread_dump()
    validator = _spread_dump()
    other_model = _spread_dump(top1_offset=1000)
    report = summarize(
        prover, validator, taus=[0.0, 0.5], cross_model_dump=other_model, min_mean_S=1.0
    )
    assert "cross_model_curve" in report
    assert report["cross_model_curve"][0]["divergence"] == 1.0


def test_summarize_operating_tau_respects_mean_S_floor():
    # With a high floor that no tau satisfies, suggested_tau is None.
    prover = _spread_dump()
    validator = _spread_dump()
    report = summarize(prover, validator, taus=[0.95], min_mean_S=10.0)
    assert report["suggested_tau"] is None
