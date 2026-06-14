# SPDX-License-Identifier: Apache-2.0
"""Driver-side writer that turns a PoC forward's captured fingerprints into a
calibration dump on disk.

``execute_poc_forward(..., capture_fingerprint=True)`` returns its fingerprints
as plain picklable data: ``{nonce_id: [decision_dict, ...]}`` (already in the
exact shape :func:`vllm.poc.fingerprint.analytics.load_dump` reads). This module
is the thin glue on the *driver rank* (in ``engine_patch.poc_request``) that:

1. rebuilds :class:`~vllm.poc.fingerprint.schema.NonceFingerprint` objects from
   that data,
2. writes the JSONL decision lines via
   :func:`~vllm.poc.fingerprint.schema.to_dump_jsonl`,
3. writes the ``<path>.meta.json`` run-context sidecar via
   :func:`~vllm.poc.fingerprint.schema.write_meta_sidecar`.

It is pure (filesystem only, no GPU/model/consensus), so it unit-tests on
synthetic fingerprint dicts. All of it runs ONLY when the
``VLLM_POC_FINGERPRINT_CAPTURE`` flag is set; with the flag unset this module is
never imported on the hot path.
"""
import json
import os
import re

from vllm.poc.fingerprint.schema import (
    CapturedDecision,
    NonceFingerprint,
    write_meta_sidecar,
)


def _fingerprints_from_serialized(
    serialized: dict,
) -> list[NonceFingerprint]:
    """Rebuild ``NonceFingerprint`` objects from ``{nonce_id: [decision_dict]}``.

    The inverse of
    :func:`vllm.poc.poc_model_runner._serialize_fingerprints`; nonces are
    emitted in ascending nonce-id order so the JSONL and the meta sidecar share a
    stable, reproducible ordering.
    """
    fingerprints: list[NonceFingerprint] = []
    for nonce_id in sorted(serialized, key=int):
        decisions = [
            CapturedDecision(
                kind=decision_dict["kind"],
                site_id=list(decision_dict["site_id"]),
                topk_ids=list(decision_dict["topk_ids"]),
                margin=float(decision_dict["margin"]),
                scores_topk=list(decision_dict["scores_topk"]),
            )
            for decision_dict in serialized[nonce_id]
        ]
        fingerprints.append(
            NonceFingerprint(nonce_id=int(nonce_id), decisions=decisions)
        )
    return fingerprints


def _slug(value: str) -> str:
    """Make ``value`` safe for use inside a filename.

    Collapses any run of characters that are not alphanumerics, dash, or dot
    into a single underscore, then trims leading/trailing underscores. Keeps the
    name readable (e.g. ``Qwen3-30B-A3B`` stays intact) while preventing path
    separators or spaces from leaking into the filename.
    """
    cleaned = re.sub(r"[^0-9A-Za-z.\-]+", "_", str(value))
    return cleaned.strip("_") or "unknown"


def build_dump_filename(
    model_id: str,
    gpu: str,
    backend: str,
    tp: int,
    block_hash: str,
) -> str:
    """Build a collision-resistant dump filename for one capture run.

    Named by ``(model, gpu, backend, tp, block_hash)`` so concurrent or repeated
    runs on different GPUs/backends/blocks never clobber each other's dumps. The
    block hash is truncated to its first 12 chars (enough to disambiguate a
    generation phase) to keep the filename short.
    """
    block_short = _slug(block_hash)[:12] or "noblock"
    parts = [
        _slug(model_id),
        _slug(gpu),
        _slug(backend),
        f"tp{int(tp)}",
        block_short,
    ]
    return "fp__" + "__".join(parts) + ".jsonl"


def write_capture_dump(
    fingerprints_serialized: dict,
    output_dir: str,
    *,
    model_id: str,
    seed: str,
    gpu: str,
    backend: str,
    tp: int,
    seq_len: int,
    block_hash: str,
    signal_set: list[str],
    top_k: int,
) -> tuple[str, str] | None:
    """Append one PoC batch's fingerprints to a run dump + refresh its sidecar.

    Multiple PoC batches in one generation phase share the same
    ``(model, gpu, backend, tp, block_hash)`` and therefore the same dump file;
    each batch *appends* its nonce lines so the full window accumulates into a
    single dump the calibrate CLI can read directly. The meta sidecar is
    rewritten each time with the cumulative nonce-id list.

    Returns ``(dump_path, meta_path)``, or ``None`` when there is nothing to
    write (empty fingerprints).
    """
    fingerprints = _fingerprints_from_serialized(fingerprints_serialized or {})
    if not fingerprints:
        return None

    os.makedirs(output_dir, exist_ok=True)
    filename = build_dump_filename(model_id, gpu, backend, tp, block_hash)
    dump_path = os.path.join(output_dir, filename)

    # Append this batch's nonce lines (one JSON list per nonce). We reuse the
    # schema's JSONL serialisation but in append mode so a multi-batch window
    # lands in one file.
    new_nonce_ids: list[int] = []
    with open(dump_path, "a", encoding="utf-8") as handle:
        for fingerprint in fingerprints:
            handle.write(json.dumps(fingerprint.to_decision_dicts()))
            handle.write("\n")
            new_nonce_ids.append(int(fingerprint.nonce_id))

    # Rewrite the sidecar with the cumulative nonce ids seen so far.
    all_nonce_ids = _read_existing_nonce_ids(dump_path, new_nonce_ids)
    meta_path = write_meta_sidecar(
        dump_path,
        model_id=model_id,
        seed=seed,
        gpu=gpu,
        backend=backend,
        tp=tp,
        seq_len=seq_len,
        nonce_ids=all_nonce_ids,
        signal_set=signal_set,
        top_k=top_k,
    )
    return dump_path, meta_path


def _read_existing_nonce_ids(dump_path: str, fallback: list[int]) -> list[int]:
    """Best-effort count of nonce lines already in ``dump_path``.

    The dump is one nonce per JSONL line, so the cumulative nonce count is just
    the number of non-empty lines. We do not need to parse them; we only need a
    count for the sidecar, but we keep the explicit ids the current batch added
    so a single-batch run still records real nonce ids. Falls back to the
    supplied ids on any read error.
    """
    try:
        line_count = 0
        with open(dump_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    line_count += 1
        # If only this batch was written, return its real ids; otherwise record
        # the cumulative count as a synthetic 0..N-1 index (the real per-nonce
        # ids live in the dump lines themselves, in order).
        if line_count == len(fallback):
            return fallback
        return list(range(line_count))
    except OSError:
        return fallback
