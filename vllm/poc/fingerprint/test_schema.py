# SPDX-License-Identifier: Apache-2.0
"""Tests for the capture-side fingerprint schema and JSONL dump format.

The critical contract is that :func:`to_dump_jsonl` writes exactly what the
analytics :func:`load_dump` reads back, so a captured dump feeds calibration
unchanged.
"""
import json

from vllm.poc.fingerprint.analytics import load_dump
from vllm.poc.fingerprint.schema import (
    LOGIT_KIND,
    ROUTING_KIND,
    CapturedDecision,
    NonceFingerprint,
    to_dump_jsonl,
    write_meta_sidecar,
)


def _routing_decision(layer_idx, position_idx):
    return CapturedDecision(
        kind=ROUTING_KIND,
        site_id=[layer_idx, position_idx],
        topk_ids=[3, 7],
        margin=1.25,
        scores_topk=[5.0, 3.75, 1.0],
    )


def test_captured_decision_to_dict_coerces_types():
    decision = CapturedDecision(
        kind=LOGIT_KIND,
        site_id=[4],
        topk_ids=[10, 11],
        margin=0.5,
        scores_topk=[2.0, 1.5, 0.1],
    )
    as_dict = decision.to_dict()
    assert as_dict == {
        "kind": "logit",
        "site_id": [4],
        "topk_ids": [10, 11],
        "margin": 0.5,
        "scores_topk": [2.0, 1.5, 0.1],
    }


def test_to_dump_jsonl_roundtrips_through_load_dump(tmp_path):
    fingerprints = [
        NonceFingerprint(
            nonce_id=100,
            decisions=[_routing_decision(0, 0), _routing_decision(1, 2)],
        ),
        NonceFingerprint(
            nonce_id=101,
            decisions=[_routing_decision(0, 1)],
        ),
    ]
    path = tmp_path / "dump.jsonl"
    nonce_ids = to_dump_jsonl(fingerprints, path)

    assert nonce_ids == [100, 101]

    loaded = load_dump(path)
    assert len(loaded) == 2
    assert len(loaded[0]) == 2
    assert len(loaded[1]) == 1
    first_decision = loaded[0][0]
    assert first_decision["kind"] == "routing"
    assert first_decision["site_id"] == [0, 0]
    assert first_decision["topk_ids"] == [3, 7]
    assert first_decision["margin"] == 1.25
    assert first_decision["scores_topk"] == [5.0, 3.75, 1.0]


def test_dump_is_one_json_list_per_line(tmp_path):
    fingerprints = [
        NonceFingerprint(nonce_id=0, decisions=[_routing_decision(0, 0)]),
        NonceFingerprint(nonce_id=1, decisions=[]),
    ]
    path = tmp_path / "dump.jsonl"
    to_dump_jsonl(fingerprints, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert isinstance(parsed, list)
    # Empty nonce serialises to an empty list.
    assert json.loads(lines[1]) == []


def test_write_meta_sidecar(tmp_path):
    path = tmp_path / "dump.jsonl"
    meta_path = write_meta_sidecar(
        path,
        model_id="Qwen3-30B-A3B",
        seed="blockhashABC",
        gpu="RTX-PRO-6000",
        backend="FLASHINFER",
        tp=4,
        seq_len=8,
        nonce_ids=[100, 101],
        signal_set=["routing", "logit"],
        top_k=8,
    )
    assert meta_path == f"{path}.meta.json"
    meta = json.loads((tmp_path / "dump.jsonl.meta.json").read_text())
    assert meta["model_id"] == "Qwen3-30B-A3B"
    assert meta["seed"] == "blockhashABC"
    assert meta["tp"] == 4
    assert meta["seq_len"] == 8
    assert meta["nonce_ids"] == [100, 101]
    assert meta["signal_set"] == ["routing", "logit"]
    assert meta["top_k"] == 8
