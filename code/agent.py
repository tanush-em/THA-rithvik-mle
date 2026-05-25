"""Per-ticket orchestrator for the support triage pipeline.

Implements the 11-stage deterministic pipeline:
  1. Parse + NFKC normalize
  2. PII detect/redact
  3. Language detect
  4. Adversarial screen
  5. Domain classify
  6. Request type + risk (policy)
  7. Retrieval (BM25, domain-filtered)
  8. LLM 1: grounded response
  9. Post-filter: PII rescan + citation validation
  10. LLM 2: safety judge
  11. Action selection + confidence calibration

Every ticket is wrapped in try/except; failures emit a safe default row.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from pii import detect_and_redact, redact, normalize
from safety import screen, SafetyResult
from lang import detect as detect_language
from classify import classify, ClassificationResult
from policy import assess, PolicyResult
from retrieve import retrieve
from llm import call_responder, call_judge, ResponderResult, JudgeResult
from tools import select_actions
from calibrate import calibrate

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# BM25 threshold for "high quality" retrieval signal
BM25_HIGH_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Canned responses (no LLM needed)
# ---------------------------------------------------------------------------

_CANNED_INJECTION_REPLY = (
    "This request contains content that appears to be an attempt to manipulate "
    "this support system. I'm unable to process that instruction. "
    "If you have a genuine support question, please resubmit it as a clear request."
)

_CANNED_OUT_OF_SCOPE = (
    "I'm sorry, this question is outside the scope of our support. "
    "I can only assist with topics related to DevPlatform, Claude, or Visa."
)

_CANNED_GREETING = (
    "Hello! How can I help you today? Please let me know your support question."
)

_CANNED_NO_CORPUS = (
    "I wasn't able to find relevant documentation for your request. "
    "Your ticket has been escalated to a human agent who will follow up shortly."
)

_CANNED_ESCALATED = (
    "Your request has been escalated to our support team. "
    "A human agent will review your case and get back to you shortly."
)

_CANNED_ERROR = (
    "We encountered an issue processing your request. "
    "Your ticket has been escalated to a human agent for manual review."
)


# ---------------------------------------------------------------------------
# Product area inference
# ---------------------------------------------------------------------------

_PRODUCT_AREA_HINTS: dict[str, list[tuple[re.Pattern, str]]] = {
    "devplatform": [
        (re.compile(r"\b(test|assessment|screen|codescreen)\b", re.I), "screen"),
        (re.compile(r"\b(candidate|invite|time\s+accommodation)\b", re.I), "candidates"),
        (re.compile(r"\b(interview|codepair|live\s+coding)\b", re.I), "interviews"),
        (re.compile(r"\b(subscription|billing|plan|payment)\b", re.I), "billing"),
        (re.compile(r"\b(account|login|password|delete\s+account)\b", re.I), "account"),
        (re.compile(r"\b(api|integration|ats)\b", re.I), "integrations"),
        (re.compile(r"\b(library|question|challenge|content)\b", re.I), "library"),
        (re.compile(r"\b(community|hackerrank|profile|badge)\b", re.I), "community"),
        (re.compile(r"\b(security|data|breach|soc|pentest)\b", re.I), "security"),
    ],
    "claude": [
        (re.compile(r"\b(billing|subscription|refund|payment|charge|pro|max|team)\b", re.I), "billing"),
        (re.compile(r"\b(privacy|data|gdpr|delete|export|conversation)\b", re.I), "privacy"),
        (re.compile(r"\b(api|bedrock|aws|integration)\b", re.I), "api"),
        (re.compile(r"\b(code|claude\s+code)\b", re.I), "claude_code"),
        (re.compile(r"\b(project|artifact|memory|feature)\b", re.I), "features"),
        (re.compile(r"\b(education|lti|university|campus)\b", re.I), "education"),
        (re.compile(r"\b(account|login|password|sso|scim)\b", re.I), "account"),
        (re.compile(r"\b(outage|error|failing|down|stop)\b", re.I), "technical"),
        (re.compile(r"\b(safety|harm|content|policy|bug\s+bounty)\b", re.I), "safety"),
    ],
    "visa": [
        (re.compile(r"\b(dispute|chargeback|unauthorized)\b", re.I), "dispute_resolution"),
        (re.compile(r"\b(fraud|stolen|cloned|lost\s+card|block)\b", re.I), "fraud_protection"),
        (re.compile(r"\b(travel|abroad|overseas|atm|pin)\b", re.I), "travel_support"),
        (re.compile(r"\b(cheque|traveller|traveler)\b", re.I), "travellers_cheques"),
        (re.compile(r"\b(merchant|business|payment\s+accept)\b", re.I), "merchant"),
        (re.compile(r"\b(liability|policy|consumer\s+right)\b", re.I), "consumer_rights"),
        (re.compile(r"\b(identity|theft)\b", re.I), "identity_protection"),
    ],
}


def _infer_product_area(text: str, domains: list[str]) -> str:
    for domain in domains:
        hints = _PRODUCT_AREA_HINTS.get(domain, [])
        for pattern, area in hints:
            if pattern.search(text):
                return area
    if domains:
        return domains[0]
    return "general_support"


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------

def _validate_citations(cited_paths: list[str]) -> list[str]:
    """Keep only citations that exist on disk."""
    valid = []
    for p in cited_paths:
        abs_path = os.path.join(REPO_ROOT, p.replace("/", os.sep))
        if os.path.isfile(abs_path):
            valid.append(p)
    return valid


# ---------------------------------------------------------------------------
# Safe default row
# ---------------------------------------------------------------------------

def _safe_default_row(input_row: dict[str, str], reason: str) -> dict[str, Any]:
    return {
        "issue": input_row.get("Issue", input_row.get("issue", "")),
        "subject": input_row.get("Subject", input_row.get("subject", "")),
        "company": input_row.get("Company", input_row.get("company", "")),
        "response": _CANNED_ERROR,
        "product_area": "general_support",
        "status": "escalated",
        "request_type": "invalid",
        "justification": f"Pipeline error — escalated for safety. Reason: {reason}",
        "confidence_score": 0.5,
        "source_documents": "",
        "risk_level": "high",
        "pii_detected": "false",
        "language": "en",
        "actions_taken": json.dumps([{
            "action": "escalate_to_human",
            "parameters": {"priority": "normal", "department": "general", "summary": reason},
        }]),
    }


# ---------------------------------------------------------------------------
# Main per-ticket async processor
# ---------------------------------------------------------------------------

async def process_ticket(input_row: dict[str, str], dry_run: bool = False) -> dict[str, Any]:
    """Process a single support ticket through the full pipeline.

    Args:
        input_row: Dict with 'Issue', 'Subject', 'Company' keys (case-insensitive).
        dry_run: If True, skip LLM calls (use canned strings). For testing.

    Returns:
        Output dict with all required columns.
    """
    try:
        return await _process(input_row, dry_run)
    except Exception as exc:
        return _safe_default_row(input_row, str(exc)[:200])


async def _process(input_row: dict[str, str], dry_run: bool) -> dict[str, Any]:
    # Normalize key casing (CSV headers vary)
    row = {k.lower(): v for k, v in input_row.items()}
    raw_issue = row.get("issue", "")
    raw_subject = row.get("subject", "")
    raw_company = row.get("company", "")

    # --- Stage 1: Parse issue JSON ---
    from io_csv import parse_issue
    turns = parse_issue(raw_issue)

    # Empty or malformed issue
    if not turns:
        return {
            "issue": raw_issue,
            "subject": raw_subject,
            "company": raw_company,
            "response": "I'm sorry, I didn't receive a valid support message.",
            "product_area": "general_support",
            "status": "replied",
            "request_type": "invalid",
            "justification": "Empty or malformed issue field — no actionable request.",
            "confidence_score": 0.97,
            "source_documents": "",
            "risk_level": "low",
            "pii_detected": "false",
            "language": "en",
            "actions_taken": "[]",
        }

    # Concatenate user turns as the full ticket text
    full_text = " ".join(
        t.get("content", "") for t in turns if t.get("role") in ("user", "human")
    )
    # Include subject as context (but mark it as potentially misleading)
    combined_text = f"{raw_subject} {full_text}".strip()

    # --- Stage 2: NFKC normalize ---
    full_text = normalize(full_text)
    combined_text = normalize(combined_text)

    # --- Stage 3: PII detection + redaction ---
    pii_result = detect_and_redact(combined_text)
    safe_text = redact(full_text, pii_result)
    safe_combined = redact(combined_text, pii_result)

    # --- Stage 4: Language detection ---
    language = detect_language(full_text)

    # --- Stage 5: Adversarial screen ---
    # Screen both the combined text and the raw full_text separately
    # (Excel injection pattern requires start-of-string match, so check full_text alone)
    safety_result: SafetyResult = screen(combined_text)
    if not safety_result.injection:
        raw_safety = screen(full_text)
        if raw_safety.injection:
            safety_result = raw_safety

    # --- Stage 6: Domain + request type classification ---
    clf: ClassificationResult = classify(safe_combined, company=raw_company, subject=raw_subject)

    # Determine effective text for downstream: use legit_fragment if injection found
    if safety_result.injection and safety_result.legit_fragment:
        effective_text = safety_result.legit_fragment
    elif safety_result.injection:
        effective_text = ""  # Nothing usable
    else:
        effective_text = safe_text

    # --- Stage 7: Policy / risk assessment ---
    policy: PolicyResult = assess(
        text=safe_combined,
        request_type=clf.request_type,
        domains=clf.domains,
        pii_detected=pii_result.detected,
        injection_detected=safety_result.injection,
        no_corpus_results=False,  # Will update after retrieval
    )

    # --- Short-circuit paths (no LLM) ---

    # Pure injection with no legit fragment
    if safety_result.injection and not safety_result.legit_fragment:
        product_area = _infer_product_area(safe_combined, clf.domains) if clf.domains else "general_support"
        justification = (
            f"Adversarial input detected: {', '.join(safety_result.reasons)}. "
            "Refused to comply. No legitimate support request found."
        )
        return {
            "issue": raw_issue, "subject": raw_subject, "company": raw_company,
            "response": _CANNED_INJECTION_REPLY,
            "product_area": product_area,
            "status": "replied",
            "request_type": "invalid",  # always invalid for pure injection
            "justification": justification,
            "confidence_score": calibrate(
                status="replied", risk_level="high", request_type="invalid",
                injection_detected=True, no_llm=True,
            ),
            "source_documents": "",
            "risk_level": "high",
            "pii_detected": str(pii_result.detected).lower(),
            "language": language,
            "actions_taken": "[]",
        }

    # Out-of-scope / invalid (greeting, trivia)
    if clf.request_type == "invalid" and clf.is_out_of_scope and not clf.domains:
        is_greeting = bool(re.match(
            r"^(?:hi|hello|hey|thanks?|thank\s+you|cheers)[^a-z]*$", full_text.strip(), re.I
        ))
        response = _CANNED_GREETING if is_greeting else _CANNED_OUT_OF_SCOPE
        return {
            "issue": raw_issue, "subject": raw_subject, "company": raw_company,
            "response": response,
            "product_area": "general_support",
            "status": "replied",
            "request_type": "invalid",
            "justification": "Request is out of scope or a simple greeting. No relevant corpus documentation.",
            "confidence_score": calibrate(
                status="replied", risk_level="low", request_type="invalid", no_llm=True,
            ),
            "source_documents": "",
            "risk_level": "low",
            "pii_detected": str(pii_result.detected).lower(),
            "language": language,
            "actions_taken": "[]",
        }

    # --- Stage 8: Retrieval ---
    results = retrieve(effective_text or safe_combined, clf.domains, top_k=5)
    no_corpus = len(results) == 0

    # Re-assess policy with corpus signal
    policy = assess(
        text=safe_combined,
        request_type=clf.request_type,
        domains=clf.domains,
        pii_detected=pii_result.detected,
        injection_detected=safety_result.injection,
        no_corpus_results=no_corpus,
    )

    snippets = [
        {"rel_path": r.rel_path, "text": r.text, "heading": r.heading}
        for r in results
    ]

    high_bm25 = bool(results and results[0].bm25_score >= BM25_HIGH_THRESHOLD)

    # --- LLM or canned response for escalated-no-corpus ---
    if policy.status == "escalated" and no_corpus:
        product_area = _infer_product_area(safe_combined, clf.domains)
        actions = select_actions(
            safe_combined, policy.status, policy.risk_level, clf.domains,
            pii_detected=pii_result.detected,
        )
        return {
            "issue": raw_issue, "subject": raw_subject, "company": raw_company,
            "response": _CANNED_ESCALATED,
            "product_area": product_area,
            "status": "escalated",
            "request_type": clf.request_type,
            "justification": f"Escalated: {policy.escalation_reason}. No corpus documentation found.",
            "confidence_score": calibrate(
                status="escalated", risk_level=policy.risk_level,
                request_type=clf.request_type, no_corpus_results=True, no_llm=True,
            ),
            "source_documents": "",
            "risk_level": policy.risk_level,
            "pii_detected": str(pii_result.detected).lower(),
            "language": language,
            "actions_taken": json.dumps(actions),
        }

    # --- Stage 9: LLM 1 — grounded response ---
    if dry_run:
        llm_response = "This is a dry-run placeholder response."
        llm_sources: list[str] = [r.rel_path for r in results[:2]]
        llm_confidence = 0.7
        judge_verdict = "keep"
        judge_reasons: list[str] = []
    else:
        resp: ResponderResult = await call_responder(effective_text or safe_combined, snippets)
        llm_response = resp.response
        llm_sources = resp.sources_used
        llm_confidence = resp.self_confidence

        # --- Stage 10: Post-filter: citation validation ---
        llm_sources = _validate_citations(llm_sources)

        # Re-scan response for PII
        response_pii = detect_and_redact(llm_response)
        if response_pii.detected:
            llm_response = redact(llm_response, response_pii)

        # --- Stage 11: LLM 2 — safety judge ---
        judge: JudgeResult = await call_judge(
            effective_text or safe_combined,
            llm_response,
            llm_sources,
        )
        judge_verdict = judge.verdict
        judge_reasons = judge.reasons

        if judge_verdict == "escalate":
            policy.status = "escalated"
            if not policy.escalation_reason:
                policy.escalation_reason = f"Safety judge escalated: {'; '.join(judge_reasons)}"
        elif judge_verdict == "sanitize" and judge.fix_hint:
            # Best-effort sanitize: remove flagged spans
            for phrase in judge.fix_hint.split(";"):
                phrase = phrase.strip()
                if phrase and phrase in llm_response:
                    llm_response = llm_response.replace(phrase, "[redacted]")

    # --- Action selection ---
    identity_in_context = bool(
        any(t.get("role") == "agent" for t in turns)
    )
    actions = select_actions(
        safe_combined, policy.status, policy.risk_level, clf.domains,
        pii_detected=pii_result.detected,
        identity_in_context=identity_in_context,
    )

    # --- Confidence calibration ---
    confidence = calibrate(
        status=policy.status,
        risk_level=policy.risk_level,
        request_type=clf.request_type,
        injection_detected=safety_result.injection,
        pii_detected=pii_result.detected,
        no_corpus_results=no_corpus,
        no_llm=dry_run,
        source_count=len(llm_sources),
        high_bm25=high_bm25,
        judge_verdict=judge_verdict if not dry_run else "keep",
        llm_self_confidence=llm_confidence,
    )

    # --- Build output row ---
    product_area = _infer_product_area(safe_combined, clf.domains)
    source_documents = "|".join(llm_sources)

    # Justification
    justification_parts = []
    if safety_result.injection:
        justification_parts.append(f"Adversarial content detected ({', '.join(safety_result.reasons)}) but legitimate sub-question answered.")
    if clf.domains:
        justification_parts.append(f"Domain: {', '.join(clf.domains)}.")
    if policy.status == "escalated":
        justification_parts.append(f"Escalated: {policy.escalation_reason}.")
    if pii_result.detected:
        justification_parts.append(f"PII detected ({', '.join(set(pii_result.types))}); redacted from response.")
    if not justification_parts:
        justification_parts.append(f"Answered from corpus. Sources: {len(llm_sources)} document(s).")
    if not dry_run and judge_verdict != "keep":
        justification_parts.append(f"Safety judge verdict: {judge_verdict}.")

    justification = " ".join(justification_parts)

    return {
        "issue": raw_issue,
        "subject": raw_subject,
        "company": raw_company,
        "response": llm_response,
        "product_area": product_area,
        "status": policy.status,
        "request_type": clf.request_type,
        "justification": justification,
        "confidence_score": confidence,
        "source_documents": source_documents,
        "risk_level": policy.risk_level,
        "pii_detected": str(pii_result.detected).lower(),
        "language": language,
        "actions_taken": json.dumps(actions),
    }
