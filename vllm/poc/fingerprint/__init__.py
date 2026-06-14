# SPDX-License-Identifier: Apache-2.0
"""PoC discrete-fingerprint calibration analytics.

Pure, offline analysis helpers for the calibration experiment described in
``vllm/poc/architecture.md`` Section 5.6. No model, no GPU, no consensus state:
these functions consume "decision dumps" produced by the capture side and
compute the calibration curves (honest flip-rate, coverage |S|, cross-model
divergence) plus the live-style hysteresis comparison and binomial fraud test.
"""
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
from vllm.poc.fingerprint.decisions import (
    logits_row_to_decision,
    router_logits_to_decisions,
    topk_ids_and_margin,
)
from vllm.poc.fingerprint.dump_writer import (
    build_dump_filename,
    write_capture_dump,
)
from vllm.poc.fingerprint.logit_capture import capture_seeded_logits
from vllm.poc.fingerprint.schema import (
    LOGIT_KIND,
    ROUTING_KIND,
    CapturedDecision,
    NonceFingerprint,
    to_dump_jsonl,
    write_meta_sidecar,
)
from vllm.poc.fingerprint.site_selection import (
    pick_logit_positions,
    pick_routing_sites,
)

__all__ = [
    # analytics (consumer side)
    "coverage_curve",
    "cross_model_divergence",
    "fraud_test",
    "honest_flip_rate",
    "hysteresis_compare",
    "load_dump",
    "save_dump",
    "summarize",
    # schema (capture side)
    "CapturedDecision",
    "NonceFingerprint",
    "ROUTING_KIND",
    "LOGIT_KIND",
    "to_dump_jsonl",
    "write_meta_sidecar",
    # site selection
    "pick_routing_sites",
    "pick_logit_positions",
    # decision computation
    "topk_ids_and_margin",
    "router_logits_to_decisions",
    "logits_row_to_decision",
    "capture_seeded_logits",
    # dump writer (driver side)
    "write_capture_dump",
    "build_dump_filename",
]
