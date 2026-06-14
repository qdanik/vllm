# SPDX-License-Identifier: Apache-2.0
"""Tests for the calibration report-assembly logic (pure).

The CLI (``python -m vllm.poc.fingerprint.calibrate``) is a thin I/O shell over
``build_calibration_report`` + ``render_markdown`` + ``threshold_row``. We test
the pure assembly here on synthetic dumps; the underlying analytics
(``coverage_curve``, ``cross_model_divergence``, ``summarize``) are already
covered by ``test_analytics.py`` and are not retested.
"""
from vllm.poc.fingerprint.calibrate import (
    build_calibration_report,
    render_markdown,
    threshold_row_to_csv,
)


def _decision(kind, site_id, top_id, margin):
    return {
        "kind": kind,
        "site_id": list(site_id),
        "topk_ids": [top_id],
        "margin": float(margin),
        "scores_topk": [margin + 1.0, 1.0],
    }


def _honest_pair():
    """Same-model prover/validator dumps: identical high-margin decisions, one
    low-margin flip that gets gated out at higher tau."""
    prover = [
        [
            _decision("routing", [0, 0], top_id=1, margin=5.0),  # high, agrees
            _decision("routing", [1, 0], top_id=9, margin=0.1),  # low, flips
        ]
    ]
    validator = [
        [
            _decision("routing", [0, 0], top_id=1, margin=5.0),  # agrees
            _decision("routing", [1, 0], top_id=2, margin=0.1),  # flips (low)
        ]
    ]
    return prover, validator


def _cross_model_dump():
    """Different model M' recomputing the same sites: top ids all differ."""
    return [
        [
            _decision("routing", [0, 0], top_id=99, margin=5.0),
            _decision("routing", [1, 0], top_id=88, margin=5.0),
        ]
    ]


def test_report_has_curves_and_suggested_tau():
    prover, validator = _honest_pair()
    report = build_calibration_report(
        prover, validator, taus=[0.0, 1.0], signal="routing", min_mean_S=1.0
    )
    assert "honest_curve" in report
    assert len(report["honest_curve"]) == 2
    # At tau=0 both decisions scored, one flips -> flip_rate 0.5; at tau=1 only
    # the high-margin (agreeing) decision is scored -> flip_rate 0.0.
    by_tau = {point["tau"]: point for point in report["honest_curve"]}
    assert by_tau[0.0]["flip_rate"] == 0.5
    assert by_tau[1.0]["flip_rate"] == 0.0
    # Suggested tau is the low-flip one with coverage above the floor.
    assert report["suggested_tau"] == 1.0


def test_report_includes_cross_model_divergence_when_supplied():
    prover, validator = _honest_pair()
    report = build_calibration_report(
        prover,
        validator,
        taus=[0.0, 1.0],
        signal="routing",
        cross_model_dump=_cross_model_dump(),
        min_mean_S=1.0,
    )
    assert "cross_model_curve" in report
    by_tau = {point["tau"]: point for point in report["cross_model_curve"]}
    # A different model diverges on every scored site -> divergence 1.0.
    assert by_tau[1.0]["divergence"] == 1.0


def test_threshold_row_drafts_governance_fields():
    prover, validator = _honest_pair()
    report = build_calibration_report(
        prover, validator, taus=[0.0, 1.0], signal="routing", min_mean_S=1.0
    )
    row = report["threshold_row"]
    # Draft governance fields (architecture.md Section 6) derived from the
    # suggested operating tau.
    assert row["tau_route"] == 1.0 or row["tau_margin"] == 1.0
    assert "delta_hyst" in row
    assert "p0" in row
    assert "fraud_threshold" in row
    assert row["signal"] == "routing"


def test_threshold_row_p0_reflects_measured_flip_rate():
    prover, validator = _honest_pair()
    report = build_calibration_report(
        prover, validator, taus=[1.0], signal="routing", min_mean_S=1.0
    )
    # At the suggested tau the honest flip rate is 0; p0 must be a small
    # positive floor, never literally 0 (binomtest needs p0 > 0).
    assert report["threshold_row"]["p0"] > 0.0


def test_render_markdown_is_a_string_with_tables():
    prover, validator = _honest_pair()
    report = build_calibration_report(
        prover, validator, taus=[0.0, 1.0], signal="routing", min_mean_S=1.0
    )
    text = render_markdown(report)
    assert isinstance(text, str)
    assert "flip_rate" in text
    assert "Suggested operating" in text
    assert "| tau " in text  # a markdown table header


def test_threshold_row_to_csv_round_trips_header_and_values():
    prover, validator = _honest_pair()
    report = build_calibration_report(
        prover, validator, taus=[1.0], signal="routing", min_mean_S=1.0
    )
    header, values = threshold_row_to_csv(report["threshold_row"])
    assert "tau_route" in header
    assert len(header) == len(values)
