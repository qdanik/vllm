# SPDX-License-Identifier: Apache-2.0
"""Tests for the driver-side fingerprint dump writer.

Pure filesystem logic, fed synthetic ``{nonce_id: [decision_dict]}`` payloads of
exactly the shape ``execute_poc_forward(capture_fingerprint=True)`` returns. We
do not retest the analytics or schema serialisation (covered elsewhere); we test
the writer's own behaviour: filename construction, append semantics across
batches, the meta sidecar, and round-trip back through ``load_dump``.
"""
import json

from vllm.poc.fingerprint.analytics import load_dump
from vllm.poc.fingerprint.dump_writer import (
    build_dump_filename,
    write_capture_dump,
)


def _serialized(nonce_to_decisions):
    """Build a {nonce_id: [decision_dict]} payload from compact specs."""
    payload = {}
    for nonce_id, decisions in nonce_to_decisions.items():
        payload[nonce_id] = [
            {
                "kind": kind,
                "site_id": list(site_id),
                "topk_ids": list(topk_ids),
                "margin": float(margin),
                "scores_topk": [float(margin) + 1.0, 1.0],
            }
            for (kind, site_id, topk_ids, margin) in decisions
        ]
    return payload


def test_build_dump_filename_is_collision_safe_and_readable():
    name = build_dump_filename(
        "Qwen3-30B-A3B", "NVIDIA H100 PCIe", "FLASHINFER", 8, "abc123def456789"
    )
    assert name.startswith("fp__")
    assert name.endswith(".jsonl")
    assert "Qwen3-30B-A3B" in name
    assert "tp8" in name
    # spaces collapsed, block hash truncated to 12 chars
    assert " " not in name
    assert "abc123def456" in name
    assert "abc123def456789" not in name  # truncated


def test_distinct_runs_get_distinct_filenames():
    base = dict(model_id="M", backend="b", tp=2, block_hash="hash0000")
    name_gpu_a = build_dump_filename(gpu="A100", **base)
    name_gpu_b = build_dump_filename(gpu="H100", **base)
    assert name_gpu_a != name_gpu_b


def test_write_dump_round_trips_through_load_dump(tmp_path):
    payload = _serialized({
        100: [("routing", [3, 1], [7, 2], 2.0)],
        101: [("logit", [5], [42, 9], 1.5)],
    })
    written = write_capture_dump(
        payload,
        str(tmp_path),
        model_id="M",
        seed="bh_pk",
        gpu="A100",
        backend="FLASH",
        tp=1,
        seq_len=8,
        block_hash="bh",
        signal_set=["routing", "logit"],
        top_k=8,
    )
    assert written is not None
    dump_path, meta_path = written

    dump = load_dump(dump_path)
    # Two nonces, ascending order by nonce id.
    assert len(dump) == 2
    assert dump[0][0]["kind"] == "routing"
    assert dump[1][0]["kind"] == "logit"

    with open(meta_path, encoding="utf-8") as meta_handle:
        meta = json.load(meta_handle)
    assert meta["model_id"] == "M"
    assert meta["seed"] == "bh_pk"
    assert meta["tp"] == 1
    assert meta["signal_set"] == ["routing", "logit"]
    assert meta["nonce_ids"] == [100, 101]


def test_multiple_batches_append_to_same_dump(tmp_path):
    common = dict(
        model_id="M",
        seed="bh_pk",
        gpu="A100",
        backend="FLASH",
        tp=1,
        seq_len=8,
        block_hash="bh",
        signal_set=["routing"],
        top_k=8,
    )
    first = write_capture_dump(
        _serialized({1: [("routing", [0, 0], [1], 1.0)]}), str(tmp_path), **common
    )
    second = write_capture_dump(
        _serialized({2: [("routing", [0, 0], [2], 1.0)]}), str(tmp_path), **common
    )
    assert first is not None and second is not None
    # Same (model, gpu, backend, tp, block_hash) -> same file.
    assert first[0] == second[0]

    dump = load_dump(first[0])
    assert len(dump) == 2  # both batches accumulated


def test_empty_fingerprints_writes_nothing(tmp_path):
    assert write_capture_dump(
        {},
        str(tmp_path),
        model_id="M",
        seed="s",
        gpu="g",
        backend="b",
        tp=1,
        seq_len=8,
        block_hash="bh",
        signal_set=["routing"],
        top_k=8,
    ) is None
    # No files created.
    assert list(tmp_path.iterdir()) == []
