# SPDX-License-Identifier: Apache-2.0
"""Calibration CLI for the PoC discrete-fingerprint experiment (Phase 0).

Consumes two same-model decision dumps (prover GPU-A vs validator GPU-B) and an
optional different-model dump ``M'`` and produces the §5.6 gate outputs:

* the honest cross-GPU **flip-rate(tau)** and coverage **|S|(tau)** curves,
* the **cross-model divergence(tau)** curve (substitution check),
* a **suggested operating tau** (lowest flip-rate while coverage stays above a
  floor), and
* a **draft per-model governance threshold row** (``tau_route``/``tau_margin``,
  ``delta_hyst``, ``p0``, ``fraud_threshold`` — architecture.md Section 6),

emitted as a printed markdown table and as CSV + JSON artifacts.

Everything here is pure over the dump data structures plus the analytics in
``vllm.poc.fingerprint.analytics``; the underlying analytics are already tested,
so this module only adds (and tests) the report-assembly glue and the CLI shell.

Usage::

    python -m vllm.poc.fingerprint.calibrate \\
        --prover dump_gpuA.jsonl --validator dump_gpuB.jsonl \\
        [--cross-model dump_Mprime.jsonl] \\
        --signal routing|logit|both \\
        --taus 0,0.5,1,2,4,8 \\
        [--min-mean-s 8] [--out report]
"""
import argparse
import csv
import io
import json
import sys

from vllm.poc.fingerprint.analytics import (
    DEFAULT_FRAUD_THRESHOLD,
    cross_model_divergence,
    summarize,
)

# When the measured honest flip-rate at the operating tau is 0, p0 still needs
# to be a small positive value (binomtest requires p0 > 0). This floor mirrors
# ``analytics.DEFAULT_P0`` and is what the draft governance table records.
P0_FLOOR = 0.001

# Default hysteresis band drafted into the governance row. The calibration
# experiment can refine it; the live validator scores only decisions whose own
# margin is >= tau_sel + delta_hyst (architecture.md Section 5.2).
DEFAULT_DELTA_HYST = 0.0

# Signal selector -> analytics ``signal`` filter. ``both`` means no filter (all
# kinds scored together).
_SIGNAL_FILTER = {"routing": "routing", "logit": "logit", "both": None}


# ---------------------------------------------------------------------------
# Pure report assembly.
# ---------------------------------------------------------------------------
def build_calibration_report(
    prover_dump,
    validator_dump,
    taus,
    signal="both",
    cross_model_dump=None,
    min_mean_S=1.0,
):
    """Assemble the full calibration report from dumps and a tau grid.

    Wraps ``analytics.summarize`` (honest coverage curve + suggested tau +
    optional cross-model curve) and adds the draft governance threshold row.
    Pure: no I/O, deterministic over the inputs.

    Args:
        prover_dump: Prover-side honest dump (GPU-A).
        validator_dump: Validator-side honest dump (GPU-B); gates scoring.
        taus: Iterable of selection thresholds to sweep.
        signal: ``"routing"`` / ``"logit"`` / ``"both"``.
        cross_model_dump: Optional different-model dump for the divergence curve.
        min_mean_S: Coverage floor for the operating-tau selection.

    Returns:
        ``{"signal", "honest_curve", "suggested_tau", "threshold_row"
        [, "cross_model_curve", "cross_model_at_operating_tau"]}``.
    """
    signal_filter = _SIGNAL_FILTER[signal]
    summary = summarize(
        prover_dump,
        validator_dump,
        list(taus),
        cross_model_dump=cross_model_dump,
        signal=signal_filter,
        min_mean_S=min_mean_S,
    )

    report = {
        "signal": signal,
        "honest_curve": summary["honest_curve"],
        "suggested_tau": summary["suggested_tau"],
    }
    if "cross_model_curve" in summary:
        report["cross_model_curve"] = summary["cross_model_curve"]

    operating_tau = summary["suggested_tau"]
    honest_at_operating = _curve_point(summary["honest_curve"], operating_tau)

    cross_model_at_operating = None
    if cross_model_dump is not None and operating_tau is not None:
        divergence = cross_model_divergence(
            prover_dump, cross_model_dump, operating_tau, signal=signal_filter
        )
        cross_model_at_operating = divergence["divergence"]
        report["cross_model_at_operating_tau"] = cross_model_at_operating

    report["threshold_row"] = _threshold_row(
        signal=signal,
        operating_tau=operating_tau,
        honest_point=honest_at_operating,
        cross_model_divergence_value=cross_model_at_operating,
    )
    return report


def _curve_point(curve, tau):
    """Return the curve point at ``tau`` (or ``None`` if absent/``tau`` None)."""
    if tau is None:
        return None
    for point in curve:
        if point["tau"] == tau:
            return point
    return None


def _threshold_row(
    signal,
    operating_tau,
    honest_point,
    cross_model_divergence_value,
):
    """Draft the per-model governance threshold row (architecture.md Section 6).

    The operating tau becomes ``tau_route`` (routing signal) and/or
    ``tau_margin`` (logit signal). ``p0`` is the measured honest flip-rate at the
    operating tau, floored to ``P0_FLOOR`` (binomtest needs p0 > 0).
    """
    measured_flip = (
        honest_point["flip_rate"] if honest_point is not None else None
    )
    mean_S = honest_point["mean_S"] if honest_point is not None else None
    p0 = max(measured_flip if measured_flip is not None else P0_FLOOR, P0_FLOOR)

    applies_routing = signal in ("routing", "both")
    applies_logit = signal in ("logit", "both")
    return {
        "signal": signal,
        "tau_route": operating_tau if applies_routing else None,
        "tau_margin": operating_tau if applies_logit else None,
        "delta_hyst": DEFAULT_DELTA_HYST,
        "p0": p0,
        "fraud_threshold": DEFAULT_FRAUD_THRESHOLD,
        "measured_honest_flip_rate": measured_flip,
        "mean_S_at_tau": mean_S,
        "cross_model_divergence_at_tau": cross_model_divergence_value,
    }


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------
def render_markdown(report):
    """Render the calibration report as a printable markdown document."""
    lines = []
    lines.append(f"# PoC fingerprint calibration — signal = {report['signal']}")
    lines.append("")
    lines.append("## Honest cross-GPU flip-rate(tau) and coverage |S|(tau)")
    lines.append("")
    lines.append("| tau | flip_rate | mean_S | total_scored |")
    lines.append("| --- | --- | --- | --- |")
    for point in report["honest_curve"]:
        lines.append(
            f"| {point['tau']} | {point['flip_rate']:.6g} | "
            f"{point['mean_S']:.6g} | {point['total_scored']} |"
        )
    lines.append("")

    if "cross_model_curve" in report:
        lines.append("## Cross-model divergence(tau) (substitution check)")
        lines.append("")
        lines.append("| tau | divergence | scored |")
        lines.append("| --- | --- | --- |")
        for point in report["cross_model_curve"]:
            lines.append(
                f"| {point['tau']} | {point['divergence']:.6g} | "
                f"{point['scored']} |"
            )
        lines.append("")

    suggested = report["suggested_tau"]
    lines.append(f"## Suggested operating tau: {suggested}")
    lines.append("")
    lines.append("## Draft per-model governance threshold row")
    lines.append("")
    row = report["threshold_row"]
    header = list(row.keys())
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    lines.append("| " + " | ".join(_fmt(row[key]) for key in header) + " |")
    lines.append("")
    return "\n".join(lines)


def _fmt(value):
    """Format a cell value for the markdown table."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def threshold_row_to_csv(threshold_row):
    """Return ``(header, values)`` lists for the threshold row CSV artifact."""
    header = list(threshold_row.keys())
    values = [threshold_row[key] for key in header]
    return header, values


def _threshold_row_csv_text(threshold_row):
    """Serialise the threshold row to a one-data-row CSV string."""
    header, values = threshold_row_to_csv(threshold_row)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerow(["" if value is None else value for value in values])
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# CLI shell.
# ---------------------------------------------------------------------------
def _parse_taus(raw):
    """Parse a comma-separated tau grid into a sorted list of floats."""
    taus = sorted({float(part) for part in raw.split(",") if part.strip() != ""})
    if not taus:
        raise argparse.ArgumentTypeError("--taus must list at least one value")
    return taus


def _load(path):
    from vllm.poc.fingerprint.analytics import load_dump

    return load_dump(path)


def main(argv=None):
    """CLI entry point. Reads dumps, builds the report, prints + writes artifacts."""
    parser = argparse.ArgumentParser(
        prog="python -m vllm.poc.fingerprint.calibrate",
        description="PoC discrete-fingerprint calibration (architecture.md 5.6).",
    )
    parser.add_argument(
        "--prover", required=True, help="Prover-side dump JSONL (GPU-A)."
    )
    parser.add_argument(
        "--validator", required=True, help="Validator-side dump JSONL (GPU-B)."
    )
    parser.add_argument(
        "--cross-model",
        default=None,
        help="Optional different-model (M') dump JSONL for divergence.",
    )
    parser.add_argument(
        "--signal",
        choices=["routing", "logit", "both"],
        default="both",
        help="Which captured signal to calibrate.",
    )
    parser.add_argument(
        "--taus",
        type=_parse_taus,
        default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
        help="Comma-separated tau grid, e.g. 0,0.5,1,2,4,8.",
    )
    parser.add_argument(
        "--min-mean-s",
        type=float,
        default=1.0,
        help="Coverage floor: candidate taus must keep mean_S above this.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Artifact basename; writes <out>.json and <out>.csv if set.",
    )
    args = parser.parse_args(argv)

    prover_dump = _load(args.prover)
    validator_dump = _load(args.validator)
    cross_model_dump = _load(args.cross_model) if args.cross_model else None

    report = build_calibration_report(
        prover_dump,
        validator_dump,
        taus=args.taus,
        signal=args.signal,
        cross_model_dump=cross_model_dump,
        min_mean_S=args.min_mean_s,
    )

    sys.stdout.write(render_markdown(report))
    sys.stdout.write("\n")

    if args.out:
        json_path = f"{args.out}.json"
        csv_path = f"{args.out}.csv"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        with open(csv_path, "w", encoding="utf-8") as handle:
            handle.write(_threshold_row_csv_text(report["threshold_row"]))
        sys.stderr.write(f"wrote {json_path} and {csv_path}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
