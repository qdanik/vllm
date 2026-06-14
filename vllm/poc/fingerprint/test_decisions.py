# SPDX-License-Identifier: Apache-2.0
"""Tests for the pure score-tensor -> CapturedDecision helpers."""
import pytest
import torch

from vllm.poc.fingerprint.decisions import (
    logits_row_to_decision,
    router_logits_to_decisions,
    topk_ids_and_margin,
)
from vllm.poc.fingerprint.schema import LOGIT_KIND, ROUTING_KIND


def test_topk_ids_and_margin_top1():
    # Scores: index 2 is top-1 (9.0), index 0 is top-2 (5.0).
    scores = torch.tensor([5.0, 1.0, 9.0, 0.0])
    topk_ids, margin, scores_topk = topk_ids_and_margin(scores, top_k=1)
    assert topk_ids == [2]
    assert margin == pytest.approx(9.0 - 5.0)
    assert scores_topk == [9.0, 5.0]  # top-(k+1) = top-2
    assert len(scores_topk) >= 2


def test_topk_ids_and_margin_topk_gap_is_k_minus_kplus1():
    # Descending scores 10,8,6,4,2; top_k=2 -> kept ids [0,1], margin = 8-6.
    scores = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])
    topk_ids, margin, scores_topk = topk_ids_and_margin(scores, top_k=2)
    assert topk_ids == [0, 1]
    assert margin == pytest.approx(8.0 - 6.0)
    assert scores_topk == [10.0, 8.0, 6.0]


def test_topk_margin_never_negative():
    scores = torch.tensor([3.0, 3.0, 3.0, 3.0])
    _ids, margin, _scores = topk_ids_and_margin(scores, top_k=2)
    assert margin >= 0.0


def test_topk_requires_two_candidates():
    with pytest.raises(ValueError):
        topk_ids_and_margin(torch.tensor([1.0]), top_k=1)


def test_router_logits_to_decisions_maps_rows_to_sites():
    # batch_size=2, seq_len=3 -> 6 token rows, 4 experts.
    seq_len = 3
    # Build router logits where the argmax expert is predictable per row.
    router_logits = torch.zeros(6, 4)
    for row in range(6):
        router_logits[row, row % 4] = 10.0  # top-1 expert = row % 4
    decisions = router_logits_to_decisions(
        router_logits, top_k=1, layer_idx=5, seq_len=seq_len
    )
    assert len(decisions) == 6
    # Row 4 -> nonce_idx 1, position_idx 1, top expert 0.
    nonce_idx, position_idx, decision = decisions[4]
    assert nonce_idx == 1
    assert position_idx == 1
    assert decision.kind == ROUTING_KIND
    assert decision.site_id == [5, 1]
    assert decision.topk_ids == [4 % 4]
    assert decision.margin > 0.0


def test_router_logits_requires_2d():
    with pytest.raises(ValueError):
        router_logits_to_decisions(
            torch.zeros(3, 4, 5), top_k=1, layer_idx=0, seq_len=1
        )


def test_logits_row_to_decision():
    vocab_logits = torch.tensor([0.1, 9.0, 0.2, 3.0])
    decision = logits_row_to_decision(vocab_logits, top_k=1, position_idx=7)
    assert decision.kind == LOGIT_KIND
    assert decision.site_id == [7]
    assert decision.topk_ids == [1]
    assert decision.margin == pytest.approx(9.0 - 3.0)
