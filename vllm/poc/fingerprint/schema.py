# SPDX-License-Identifier: Apache-2.0
"""Capture-side data model for the PoC discrete-fingerprint experiment.

This is the *producer* counterpart to ``vllm/poc/fingerprint/analytics.py``.
The capture path (a flagged branch inside the PoC forward) builds the
dataclasses below and serialises them to the exact JSONL shape that
:func:`vllm.poc.fingerprint.analytics.load_dump` consumes.

Decision schema (contract with the analytics side)
--------------------------------------------------
A single captured decision serialises to::

    {
        "kind": "routing" | "logit",
        "site_id": [...],          # routing: [layer_idx, position_idx]
                                   # logit:   [position_idx]
        "topk_ids": [int, ...],    # selected expert/token ids, top-1 first
        "margin": float,           # score[k-1] - score[k] (>= 0)
        "scores_topk": [float, ...]  # raw top-(k+1) scores, length >= 2
    }

A *decision dump* is one JSONL line per nonce, each line being a JSON list of
those decision dicts. A sidecar ``<path>.meta.json`` records the run context
(model, seed, gpu, backend, tp, seq_len, nonce order, signal set, top_k).
"""
import json
from dataclasses import dataclass, field

# The two discrete signals captured per nonce (architecture.md Section 5.1).
ROUTING_KIND = "routing"
LOGIT_KIND = "logit"


@dataclass
class CapturedDecision:
    """One discrete decision captured at a single seeded site.

    Attributes:
        kind: ``"routing"`` (MoE expert selection) or ``"logit"`` (LM-head
            token argmax).
        site_id: Canonical seeded-site id. Routing sites encode
            ``[layer_idx, position_idx]``; logit sites encode
            ``[position_idx]``. A list (not a tuple) so it round-trips through
            JSON unchanged.
        topk_ids: Selected expert/token ids, top-1 first. Length ``top_k``.
        margin: Score gap ``score[k-1] - score[k]`` over the raw scores. Always
            ``>= 0``; this is the high-margin filter knob the analytics sweeps.
        scores_topk: Raw top-``(top_k + 1)`` scores, descending. Length is
            always ``>= 2`` so a margin is well defined.
    """

    kind: str
    site_id: list[int]
    topk_ids: list[int]
    margin: float
    scores_topk: list[float]

    def to_dict(self) -> dict:
        """Serialise to the plain dict that the analytics ``load_dump`` reads."""
        return {
            "kind": self.kind,
            "site_id": list(self.site_id),
            "topk_ids": [int(token_id) for token_id in self.topk_ids],
            "margin": float(self.margin),
            "scores_topk": [float(score) for score in self.scores_topk],
        }


@dataclass
class NonceFingerprint:
    """All captured decisions for a single nonce's forward pass."""

    nonce_id: int
    decisions: list[CapturedDecision] = field(default_factory=list)

    def to_decision_dicts(self) -> list[dict]:
        """Serialise to the JSON list that becomes one ``load_dump`` line."""
        return [decision.to_dict() for decision in self.decisions]


def to_dump_jsonl(fingerprints: list[NonceFingerprint], path) -> list[int]:
    """Write fingerprints to JSONL plus a ``<path>.meta.json`` is written by the
    caller; here we only emit the decision lines in nonce order.

    One line per nonce, each line a JSON list of decision dicts. This is byte-
    for-byte the format :func:`vllm.poc.fingerprint.analytics.load_dump`
    expects, so a dump written here loads straight back for calibration.

    Returns:
        The nonce ids in the order they were written (so the caller can store
        them in the sidecar meta).
    """
    nonce_ids: list[int] = []
    with open(path, "w", encoding="utf-8") as handle:
        for fingerprint in fingerprints:
            handle.write(json.dumps(fingerprint.to_decision_dicts()))
            handle.write("\n")
            nonce_ids.append(int(fingerprint.nonce_id))
    return nonce_ids


def write_meta_sidecar(
    path,
    model_id: str,
    seed: str,
    gpu: str,
    backend: str,
    tp: int,
    seq_len: int,
    nonce_ids: list[int],
    signal_set: list[str],
    top_k: int,
) -> str:
    """Write the ``<path>.meta.json`` sidecar describing the run context.

    ``seed`` is the run seed string (block_hash and/or public_key) that fixes
    the seeded-site selection, so a validator can regenerate the identical set.

    Returns:
        The sidecar path that was written.
    """
    meta_path = f"{path}.meta.json"
    meta = {
        "model_id": model_id,
        "seed": seed,
        "gpu": gpu,
        "backend": backend,
        "tp": int(tp),
        "seq_len": int(seq_len),
        "nonce_ids": [int(nonce_id) for nonce_id in nonce_ids],
        "signal_set": list(signal_set),
        "top_k": int(top_k),
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    return meta_path
