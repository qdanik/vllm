# SPDX-License-Identifier: Apache-2.0
"""Pure analytics for the PoC discrete-fingerprint calibration experiment.

This module is the *consumer* side of the calibration pipeline. The capture
side (a flagged path inside the model forward) emits "decision dumps"; here we
turn those dumps into the curves and tests that gate the design described in
``vllm/poc/architecture.md`` Section 5.6.

No model, no GPU, no consensus: every function below is a deterministic pure
function over plain Python data structures.

Decision schema (the contract with the capture side)
----------------------------------------------------
A single *decision* is a dict::

    {
        "kind": "routing" | "logit",   # which seeded signal produced it
        "site_id": <hashable seeded id>,  # routing: [layer_idx, position_idx]
                                          # logit:   [position_idx]
        "topk_ids": [int, ...],         # selected expert/token ids, top-1 first
        "margin": float,                # score gap top-1 minus top-2 (>= 0)
        "scores_topk": [float, ...],    # optional raw scores
    }

A *decision dump* for one ``(model, gpu, config)`` run is a ``list`` over
nonces; each nonce is a ``list`` of decisions::

    dump: list[list[dict]]

Two dumps are *aligned* when they share the model-seed and therefore expose the
same set of ``site_id`` values per nonce. Alignment is matched positionally by
nonce index and by ``site_id`` within a nonce, so a partially-overlapping pair
is handled by scoring only the intersecting sites.
"""
import json
from collections.abc import Hashable

from scipy.stats import binomtest

# Defaults mirror ``vllm/poc/data.py`` so the calibration test and the live
# validator share the same baseline honest-flip-rate assumptions.
DEFAULT_P0 = 0.001
DEFAULT_FRAUD_THRESHOLD = 0.01

# Type aliases for readability.
Decision = dict
Nonce = list  # list[Decision]
Dump = list  # list[Nonce]


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------
def _site_key(decision: Decision) -> Hashable:
    """Return a hashable key for a decision's seeded site id.

    ``site_id`` arrives as a JSON list (e.g. ``[layer_idx, position_idx]``),
    which is not hashable; convert lists to tuples so they can index a dict.
    """
    site_id = decision["site_id"]
    if isinstance(site_id, list):
        return tuple(site_id)
    return site_id


def _top1(decision: Decision):
    """Return the top-1 selected id (expert or token) of a decision."""
    return decision["topk_ids"][0]


def _index_by_site(
    nonce: Nonce,
    signal: str | None,
) -> dict:
    """Build a ``site_key -> decision`` map for one nonce, optionally filtered.

    When ``signal`` is given, only decisions of that ``kind`` are kept.
    """
    indexed: dict = {}
    for decision in nonce:
        if signal is not None and decision["kind"] != signal:
            continue
        indexed[_site_key(decision)] = decision
    return indexed


def _aligned_sites(
    nonce_a: Nonce,
    nonce_b: Nonce,
    signal: str | None,
):
    """Yield ``(decision_a, decision_b)`` pairs for sites present in both nonces."""
    indexed_a = _index_by_site(nonce_a, signal)
    indexed_b = _index_by_site(nonce_b, signal)
    for site_key, decision_a in indexed_a.items():
        decision_b = indexed_b.get(site_key)
        if decision_b is not None:
            yield decision_a, decision_b


# ---------------------------------------------------------------------------
# 1. JSONL round-trip.
# ---------------------------------------------------------------------------
def save_dump(dump: Dump, path) -> None:
    """Write a decision dump to JSONL: one nonce (list of decisions) per line.

    JSONL is the primary format because it is human-inspectable and streamable.
    """
    with open(path, "w", encoding="utf-8") as handle:
        for nonce in dump:
            handle.write(json.dumps(nonce))
            handle.write("\n")


def load_dump(path) -> Dump:
    """Load a decision dump previously written by :func:`save_dump`."""
    dump: Dump = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            dump.append(json.loads(line))
    return dump


# ---------------------------------------------------------------------------
# 2. honest_flip_rate.
# ---------------------------------------------------------------------------
def honest_flip_rate(
    dump_a: Dump,
    dump_b: Dump,
    tau: float,
    signal: str | None = None,
) -> dict:
    """Cross-GPU honest flip-rate at selection threshold ``tau``.

    For each nonce, take the set of decisions present at the same ``site_id`` in
    BOTH dumps. ``dump_b`` is the reference / validator side: a decision is
    *scored* iff its reference margin is ``>= tau``. A scored decision is a
    *flip* iff its top-1 id differs between A and B.

    Args:
        dump_a: Prover-side dump (``list[list[dict]]``).
        dump_b: Reference / validator-side dump; its margin gates scoring.
        tau: Selection threshold on the reference margin.
        signal: Optional ``"routing"``/``"logit"`` filter.

    Returns:
        ``{"flip_rate", "scored_count_per_nonce", "total_scored"}`` where
        ``flip_rate`` is flips / total_scored (0.0 when nothing is scored).
    """
    scored_count_per_nonce: list = []
    total_scored = 0
    total_flips = 0

    for nonce_a, nonce_b in zip(dump_a, dump_b):
        nonce_scored = 0
        for decision_a, reference in _aligned_sites(nonce_a, nonce_b, signal):
            if reference["margin"] < tau:
                continue
            nonce_scored += 1
            if _top1(decision_a) != _top1(reference):
                total_flips += 1
        scored_count_per_nonce.append(nonce_scored)
        total_scored += nonce_scored

    flip_rate = total_flips / total_scored if total_scored else 0.0
    return {
        "flip_rate": flip_rate,
        "scored_count_per_nonce": scored_count_per_nonce,
        "total_scored": total_scored,
    }


# ---------------------------------------------------------------------------
# 3. coverage_curve.
# ---------------------------------------------------------------------------
def coverage_curve(
    dump_a: Dump,
    dump_b: Dump,
    taus,
    signal: str | None = None,
) -> list:
    """Sweep ``tau`` and report flip-rate and coverage |S| at each step.

    This is the key calibration output: ``flip_rate(tau)`` and ``mean_S(tau)``.
    As ``tau`` rises, fewer decisions clear the margin gate, so ``total_scored``
    (and ``mean_S``) are non-increasing.

    Returns:
        A list of ``{"tau", "flip_rate", "mean_S", "total_scored"}`` dicts, one
        per value in ``taus``.
    """
    curve: list = []
    n_nonces = len(dump_a)
    for tau in taus:
        result = honest_flip_rate(dump_a, dump_b, tau, signal=signal)
        total_scored = result["total_scored"]
        mean_S = total_scored / n_nonces if n_nonces else 0.0
        curve.append({
            "tau": tau,
            "flip_rate": result["flip_rate"],
            "mean_S": mean_S,
            "total_scored": total_scored,
        })
    return curve


# ---------------------------------------------------------------------------
# 4. cross_model_divergence.
# ---------------------------------------------------------------------------
def cross_model_divergence(
    dump_m: Dump,
    dump_mprime: Dump,
    tau: float,
    signal: str | None = None,
) -> dict:
    """Fraction of scored decisions whose top-1 id differs between two models.

    Same shape as :func:`honest_flip_rate`, but the expectation is the opposite:
    when ``dump_mprime`` is a *different* model recomputing ``M``'s decisions,
    divergence should be ~1.0 (confirming model substitution is caught). Here
    the reference / scoring side is ``dump_mprime``.

    Returns:
        ``{"divergence", "scored"}``; ``divergence`` is 0.0 for an empty scored
        set.
    """
    scored = 0
    diverged = 0
    for nonce_m, nonce_mprime in zip(dump_m, dump_mprime):
        for decision_m, reference in _aligned_sites(nonce_m, nonce_mprime, signal):
            if reference["margin"] < tau:
                continue
            scored += 1
            if _top1(decision_m) != _top1(reference):
                diverged += 1
    divergence = diverged / scored if scored else 0.0
    return {"divergence": divergence, "scored": scored}


# ---------------------------------------------------------------------------
# 5. hysteresis_compare (mirrors live Section 5.2 comparison).
# ---------------------------------------------------------------------------
def hysteresis_compare(
    prover_dump: Dump,
    validator_dump: Dump,
    tau_sel: float,
    delta_hyst: float,
    signal: str | None = None,
) -> list:
    """Live-style per-nonce comparison with a hysteresis band.

    Mirrors ``architecture.md`` Section 5.2: the validator is the anchor and
    scores *only* decisions where its OWN margin is ``>= tau_sel + delta_hyst``.
    A scored decision is a mismatch iff the prover and validator top-1 ids
    differ.

    Returns:
        A list of ``{"n_scored", "n_mismatch"}`` dicts, one per nonce. These
        feed :func:`fraud_test` after summation across a validation window.
    """
    band = tau_sel + delta_hyst
    per_nonce: list = []
    for prover_nonce, validator_nonce in zip(prover_dump, validator_dump):
        n_scored = 0
        n_mismatch = 0
        for prover_decision, validator_decision in _aligned_sites(
            prover_nonce, validator_nonce, signal
        ):
            if validator_decision["margin"] < band:
                continue
            n_scored += 1
            if _top1(prover_decision) != _top1(validator_decision):
                n_mismatch += 1
        per_nonce.append({"n_scored": n_scored, "n_mismatch": n_mismatch})
    return per_nonce


# ---------------------------------------------------------------------------
# 6. fraud_test (binomtest).
# ---------------------------------------------------------------------------
def fraud_test(
    n_mismatch: int,
    n_scored: int,
    p0: float = DEFAULT_P0,
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD,
) -> dict:
    """One-sided binomial fraud test over a scored set.

    Tests whether the observed mismatch count exceeds the honest baseline ``p0``
    by more than chance, using ``binomtest(..., alternative="greater")`` (same
    structure as the live validator in ``vllm/poc/data.py``).

    Returns:
        ``{"p_value", "fraud"}``; ``fraud`` is ``p_value < fraud_threshold``.
        An empty scored set is never fraud (``p_value == 1.0``).
    """
    if n_scored == 0:
        return {"p_value": 1.0, "fraud": False}
    result = binomtest(
        k=n_mismatch,
        n=n_scored,
        p=p0,
        alternative="greater",
    )
    p_value = float(result.pvalue)
    return {"p_value": p_value, "fraud": p_value < fraud_threshold}


# ---------------------------------------------------------------------------
# 7. summarize.
# ---------------------------------------------------------------------------
def summarize(
    prover_dump: Dump,
    validator_dump: Dump,
    taus,
    cross_model_dump: Dump | None = None,
    signal: str | None = None,
    min_mean_S: float = 1.0,
) -> dict:
    """Build a calibration report and suggest an operating ``tau``.

    Combines the honest coverage curve (and, when provided, the cross-model
    divergence curve) and picks a *suggested operating ``tau``*: the smallest
    ``tau`` that minimises the honest flip-rate while keeping ``mean_S`` at or
    above ``min_mean_S``. This feeds the gate decision and the draft per-model
    threshold table of Section 5.6.

    Args:
        prover_dump: Prover-side honest dump.
        validator_dump: Validator / reference honest dump (gates scoring).
        taus: Iterable of selection thresholds to sweep.
        cross_model_dump: Optional dump from a different model ``M'`` for the
            divergence curve. The cross-model side gates scoring (its margin
            must clear ``tau``); divergence is expected near ~1.0.
        signal: Optional ``"routing"``/``"logit"`` filter.
        min_mean_S: Coverage floor; candidate taus must keep ``mean_S`` above it.

    Returns:
        ``{"honest_curve", "suggested_tau"[, "cross_model_curve"]}``.
    """
    honest_curve = coverage_curve(prover_dump, validator_dump, taus, signal=signal)

    report = {
        "honest_curve": honest_curve,
        "suggested_tau": _select_operating_tau(honest_curve, min_mean_S),
    }

    if cross_model_dump is not None:
        cross_model_curve = []
        for tau in taus:
            divergence = cross_model_divergence(
                prover_dump, cross_model_dump, tau, signal=signal
            )
            cross_model_curve.append({
                "tau": tau,
                "divergence": divergence["divergence"],
                "scored": divergence["scored"],
            })
        report["cross_model_curve"] = cross_model_curve

    return report


def _select_operating_tau(honest_curve: list, min_mean_S: float):
    """Pick the operating tau: minimal flip-rate subject to a coverage floor.

    Among curve points whose ``mean_S >= min_mean_S``, choose the lowest
    flip-rate; break ties by the smallest ``tau`` (keeps the largest scored
    set). Returns ``None`` when no point clears the coverage floor.
    """
    eligible = [point for point in honest_curve if point["mean_S"] >= min_mean_S]
    if not eligible:
        return None
    best = min(eligible, key=lambda point: (point["flip_rate"], point["tau"]))
    return best["tau"]
