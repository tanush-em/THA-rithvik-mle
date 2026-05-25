"""Confidence calibration using signal-based buckets.

Brier-aware: avoids flat/constant scores. Uses discrete buckets driven by
pipeline signals, blended with the LLM's self-reported confidence at 25% weight.

Bucket map is tunable in one place.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Signal bucket definitions
# (bucket_base, max_llm_blend_delta)
# ---------------------------------------------------------------------------

# Ordered evaluation (first matching wins)
# Keys: tuple of (status, risk_level, injection, no_corpus, judge_verdict, source_count)
# Values: (base_confidence, llm_blend_weight)

_BUCKETS: list[tuple[dict, float]] = [
    # Canned invalid / out-of-scope — very high confidence
    ({"request_type": "invalid", "status": "replied", "no_llm": True}, 0.97),

    # Injection-only escalation — high confidence in the classification
    ({"injection_detected": True, "status": "replied", "no_llm": True}, 0.95),

    # Critical risk escalation (legal/identity/safety)
    ({"risk_level": "critical", "status": "escalated"}, 0.88),

    # High risk escalation with PII
    ({"risk_level": "high", "status": "escalated", "pii_detected": True}, 0.85),

    # High risk escalation without PII
    ({"risk_level": "high", "status": "escalated"}, 0.82),

    # Medium risk escalation
    ({"risk_level": "medium", "status": "escalated"}, 0.78),

    # Low risk escalation (no corpus results)
    ({"no_corpus_results": True, "status": "escalated"}, 0.65),

    # Judge escalated (safety issue found in LLM response)
    ({"judge_verdict": "escalate", "status": "escalated"}, 0.80),

    # Replied — strong retrieval (2+ sources, judge=keep)
    ({"status": "replied", "source_count_ge": 2, "judge_verdict": "keep"}, 0.88),

    # Replied — good retrieval (1 source, high BM25)
    ({"status": "replied", "source_count_ge": 1, "high_bm25": True, "judge_verdict": "keep"}, 0.72),

    # Replied — weak retrieval
    ({"status": "replied", "judge_verdict": "keep"}, 0.55),

    # Replied — judge sanitized
    ({"status": "replied", "judge_verdict": "sanitize"}, 0.50),

    # Fallback
    ({}, 0.60),
]

_LLM_BLEND_WEIGHT = 0.25
_MAX_DELTA = 0.07  # Max adjustment from LLM self-confidence


def calibrate(
    *,
    status: str,
    risk_level: str,
    request_type: str,
    injection_detected: bool = False,
    pii_detected: bool = False,
    no_corpus_results: bool = False,
    no_llm: bool = False,
    source_count: int = 0,
    high_bm25: bool = False,
    judge_verdict: str = "keep",
    llm_self_confidence: float = 0.7,
) -> float:
    """Return a calibrated confidence score [0.0, 1.0].

    Args:
        status: "replied" | "escalated"
        risk_level: "low" | "medium" | "high" | "critical"
        request_type: "product_issue" | "feature_request" | "bug" | "invalid"
        injection_detected: Whether adversarial injection was detected.
        pii_detected: Whether PII was found.
        no_corpus_results: Whether BM25 retrieval returned no results.
        no_llm: Whether the LLM was skipped (canned response path).
        source_count: Number of valid source documents cited.
        high_bm25: Whether the top BM25 score exceeded the threshold.
        judge_verdict: "keep" | "sanitize" | "escalate"
        llm_self_confidence: The LLM's self-reported confidence (0.0–1.0).

    Returns:
        Calibrated confidence float in [0.0, 1.0].
    """
    signals = {
        "status": status,
        "risk_level": risk_level,
        "request_type": request_type,
        "injection_detected": injection_detected,
        "pii_detected": pii_detected,
        "no_corpus_results": no_corpus_results,
        "no_llm": no_llm,
        "source_count": source_count,
        "high_bm25": high_bm25,
        "judge_verdict": judge_verdict,
    }

    base = 0.60  # fallback
    for conditions, bucket_value in _BUCKETS:
        if _matches(conditions, signals):
            base = bucket_value
            break

    if no_llm:
        return round(min(max(base, 0.0), 1.0), 4)

    # Blend with LLM self-confidence
    llm_contribution = (llm_self_confidence - base) * _LLM_BLEND_WEIGHT
    llm_contribution = max(-_MAX_DELTA, min(_MAX_DELTA, llm_contribution))
    score = base + llm_contribution

    return round(min(max(score, 0.01), 0.99), 4)


def _matches(conditions: dict, signals: dict) -> bool:
    """Check whether all conditions match the signals dict."""
    for key, expected in conditions.items():
        if key == "source_count_ge":
            if signals.get("source_count", 0) < expected:
                return False
        elif key == "high_bm25":
            if signals.get("high_bm25") != expected:
                return False
        else:
            if signals.get(key) != expected:
                return False
    return True
