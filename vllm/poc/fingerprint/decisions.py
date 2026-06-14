# SPDX-License-Identifier: Apache-2.0
"""Pure score-tensor -> CapturedDecision helpers for the PoC fingerprint.

These are the deterministic numeric core shared by the routing hook
(:mod:`vllm.poc.fingerprint_hooks`) and the seeded-logit capture: turn a raw
score tensor into top-``(k+1)`` ids, the high-margin gap ``score[k-1] -
score[k]``, and the top-``(k+1)`` scores. Kept free of any model/GPU coupling
so they unit-test on tiny synthetic CPU tensors.

Margin definition (architecture.md Section 5.1): take ``top_k + 1`` scores; the
margin is ``score[k-1] - score[k]`` (the gap between the last kept id and the
first discarded one). For ``top_k == 1`` this is the familiar top-1 minus top-2
gap. The margin is the high-margin filter knob the analytics sweeps from 0, so
we never pre-filter here.
"""
import torch

from vllm.poc.fingerprint.schema import (
    LOGIT_KIND,
    ROUTING_KIND,
    CapturedDecision,
)


def topk_ids_and_margin(
    scores_row: torch.Tensor,
    top_k: int,
) -> tuple[list[int], float, list[float]]:
    """Compute top-``k`` ids, the margin, and the top-``(k+1)`` scores.

    Args:
        scores_row: 1-D score tensor for a single decision (e.g. one token's
            router logits over experts, or one position's logits over the
            vocab).
        top_k: Number of ids to keep. Internally takes ``top_k + 1`` so a margin
            is always defined; the kept ``topk_ids`` has length ``top_k``.

    Returns:
        ``(topk_ids, margin, scores_topk)`` where ``topk_ids`` has length
        ``top_k`` (top-1 first), ``margin = scores_topk[k-1] - scores_topk[k]``
        (``>= 0``), and ``scores_topk`` has length ``min(top_k + 1, n)`` and is
        always at least 2 (we require at least 2 candidates).
    """
    num_candidates = scores_row.shape[-1]
    if num_candidates < 2:
        raise ValueError(
            f"need at least 2 scores to define a margin, got {num_candidates}"
        )
    keep = min(top_k + 1, num_candidates)
    top_scores, top_indices = torch.topk(scores_row, keep, sorted=True)
    scores_topk = [float(score) for score in top_scores.tolist()]
    ids_full = [int(index) for index in top_indices.tolist()]
    # Keep top_k ids (may be fewer than top_k only if num_candidates < top_k).
    kept_count = min(top_k, num_candidates)
    topk_ids = ids_full[:kept_count]
    # Margin = gap between the last kept score and the first discarded score.
    # scores_topk has the extra (k+1)-th score precisely to define this gap.
    margin = scores_topk[kept_count - 1] - scores_topk[kept_count]
    return topk_ids, float(margin), scores_topk


def router_logits_to_decisions(
    router_logits: torch.Tensor,
    top_k: int,
    layer_idx: int,
    seq_len: int,
) -> list[tuple[int, int, CapturedDecision]]:
    """Convert one MoE layer's ``router_logits`` into per-token routing decisions.

    Args:
        router_logits: ``[num_tokens, num_experts]`` raw gate logits, where
            ``num_tokens == batch_size * seq_len``.
        top_k: The ``FusedMoE.top_k`` for this layer.
        layer_idx: This MoE layer's index (the routing site's layer component).
        seq_len: Per-nonce sequence length, used to map a flat token row back to
            ``(nonce_idx, position_idx)``: ``nonce_idx = row // seq_len``,
            ``position_idx = row % seq_len``.

    Returns:
        A list of ``(nonce_idx, position_idx, CapturedDecision)``. The decision's
        ``site_id`` is ``[layer_idx, position_idx]``. One entry per token row.
    """
    if router_logits.dim() != 2:
        raise ValueError(
            f"router_logits must be 2-D [num_tokens, num_experts], got shape "
            f"{tuple(router_logits.shape)}"
        )
    scores = router_logits.detach().to(torch.float32)
    num_tokens = scores.shape[0]
    results: list[tuple[int, int, CapturedDecision]] = []
    for row in range(num_tokens):
        nonce_idx = row // seq_len
        position_idx = row % seq_len
        topk_ids, margin, scores_topk = topk_ids_and_margin(scores[row], top_k)
        decision = CapturedDecision(
            kind=ROUTING_KIND,
            site_id=[layer_idx, position_idx],
            topk_ids=topk_ids,
            margin=margin,
            scores_topk=scores_topk,
        )
        results.append((nonce_idx, position_idx, decision))
    return results


def logits_row_to_decision(
    logits_row: torch.Tensor,
    top_k: int,
    position_idx: int,
) -> CapturedDecision:
    """Convert one position's LM-head logits into a logit ``CapturedDecision``.

    Args:
        logits_row: 1-D ``[vocab]`` logits for a single seeded position.
        top_k: Number of token ids to keep.
        position_idx: The seeded position; becomes ``site_id == [position_idx]``.
    """
    topk_ids, margin, scores_topk = topk_ids_and_margin(
        logits_row.detach().to(torch.float32), top_k
    )
    return CapturedDecision(
        kind=LOGIT_KIND,
        site_id=[position_idx],
        topk_ids=topk_ids,
        margin=margin,
        scores_topk=scores_topk,
    )
