"""Risk assessment and escalation policy.

All decisions are deterministic rule tables — no LLM. The principle is:
"err on the side of caution" — ambiguous risk → escalate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Risk patterns
# ---------------------------------------------------------------------------

_CRITICAL_PATTERNS = re.compile(
    r"\b("
    r"lawsuit|legal\s+action|legal\s+threat|class\s+action|file\s+a\s+(?:complaint|lawsuit|suit)"
    r"|attorney|lawyer|legal\s+counsel|discrimination\s+lawsuit|damages"
    r"|identity\s+(?:theft|stolen)|(?:my\s+)?identity\s+has\s+been\s+stolen"
    r"|account\s+(?:hacked|compromised|taken\s+over)"
    r"|dangerous\s+(?:medical|advice)|stop\s+taking\s+(?:my\s+)?(?:medication|medicine|prescription)"
    r"|someone\s+else'?s\s+conversation|privacy\s+breach|data\s+breach"
    r"|security\s+vulnerability|cve-\d|bug\s+bounty"
    r"|gdpr.*article\s*17|right\s+to\s+erasure|ico\s+(?:complaint|filing)"
    r"|hipaa|baa\s+(?:sign|agreement)"
    r"|child|minor\s+(?:access|data)"
    r")\b",
    re.IGNORECASE,
)

_HIGH_PATTERNS = re.compile(
    r"\b("
    r"stolen|lost.*card|card.*(?:lost|stolen|blocked|cloned)"
    r"|fraud(?:ulent)?|unauthorized\s+(?:charge|transaction|access)"
    r"|stranded|emergency\s+(?:cash|options)|no\s+other\s+payment"
    r"|account\s+(?:delete|deletion|remove\s+all|block)"
    r"|outage|site.*down|none\s+of\s+the\s+pages|complete.*(?:failure|outage)"
    r"|refund\s+\$[\d,]+|chargeback"
    r"|suspend|terminate\s+(?:contract|subscription|account)"
    r"|data\s+(?:leak|leakage|exposure|exfiltration)"
    r"|forgot\s+pin.*(?:card|atm)|atm\s+(?:ate|took|kept)\s+(?:my\s+)?card"
    r")\b",
    re.IGNORECASE,
)

_MEDIUM_PATTERNS = re.compile(
    r"\b("
    r"billing|refund|subscription|payment|charge|invoice|cancel|pause"
    r"|delete\s+(?:account|my\s+account)|account\s+(?:recovery|access)"
    r"|data\s+(?:export|download|deletion)"
    r"|api\s+(?:error|issue|failing|500)"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Escalation trigger patterns (beyond just risk level)
# ---------------------------------------------------------------------------

_ESCALATION_PATTERNS = re.compile(
    r"\b("
    # Legal / regulatory
    r"lawsuit|legal\s+action|class\s+action|discrimination|attorney|legal\s+counsel"
    r"|gdpr.*article\s*17|right\s+to\s+erasure|ico\s+complaint"
    r"|hipaa|baa"
    # Safety
    r"|dangerous\s+(?:medical|advice)|stop\s+taking\s+(?:my\s+)?(?:medication|medicine)"
    r"|life-?threatening"
    # Account takeover / identity
    r"|identity\s+(?:theft|stolen)|account\s+hacked|account\s+(?:taken\s+over|compromised)"
    r"|someone\s+else'?s\s+conversation|cross.user\s+data"
    # Major financial
    r"|stolen.*(?:card|cheque)|card.*stolen|card.*cloned|unauthorized\s+(?:charge|transaction)"
    r"|atm.*(?:ate|took|kept).*card|stranded.*(?:overseas|abroad|no\s+payment)"
    # Enterprise / contract
    r"|enterprise\s+contract|breach\s+of\s+contract|early\s+termination"
    r")\b",
    re.IGNORECASE,
)

# Tickets that can be replied but need verification first
_NEEDS_VERIFICATION = re.compile(
    r"\b("
    r"delete\s+(?:my\s+)?account|cancel\s+(?:my\s+)?subscription"
    r"|pause\s+(?:(?:our|my)\s+)?subscription|pause\s+(?:our|my)\s+(?:plan|account)"
    r"|refund|issue.*refund"
    r")\b",
    re.IGNORECASE,
)

# Truly out-of-scope / no-domain / no-action needed
_OUT_OF_SCOPE = re.compile(
    r"\b("
    r"iron\s+man|actor|movie|film|celebrity|recipe|weather|stock\s+price"
    r"|thank\s+you\s+for\s+helping|happy\s+to\s+help|no\s+(?:support\s+)?request"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PolicyResult:
    status: str = "replied"          # "replied" | "escalated"
    risk_level: str = "low"          # "low" | "medium" | "high" | "critical"
    escalation_reason: str = ""      # human-readable reason if escalated
    needs_verification: bool = False


# ---------------------------------------------------------------------------
# Main policy function
# ---------------------------------------------------------------------------

def assess(
    text: str,
    request_type: str,
    domains: list[str],
    pii_detected: bool = False,
    injection_detected: bool = False,
    no_corpus_results: bool = False,
) -> PolicyResult:
    """Apply policy rules to determine status and risk_level.

    Args:
        text: Sanitized ticket text.
        request_type: From classifier ('product_issue', 'bug', etc.).
        domains: Detected domain list.
        pii_detected: Whether PII was found in the ticket.
        injection_detected: Whether injection was detected.
        no_corpus_results: Whether retrieval returned no results.

    Returns:
        PolicyResult with status and risk_level.
    """
    result = PolicyResult()

    # --- Risk level ---
    if _CRITICAL_PATTERNS.search(text):
        result.risk_level = "critical"
    elif _HIGH_PATTERNS.search(text) or (pii_detected and any(
        kw in text.lower() for kw in ("stolen", "unauthorized", "fraud", "hacked", "blocked")
    )):
        result.risk_level = "high"
    elif _MEDIUM_PATTERNS.search(text) or pii_detected:
        result.risk_level = "medium"
    else:
        result.risk_level = "low"

    # --- Escalation rules (ordered by priority) ---

    # 1. Injection-only ticket (no legit fragment) — still reply with refusal
    if injection_detected and request_type == "invalid":
        result.status = "replied"
        result.risk_level = "high"
        return result

    # 2. Critical risk → always escalate
    if result.risk_level == "critical":
        result.status = "escalated"
        result.escalation_reason = "Critical risk: legal/safety/regulatory/severe breach"
        return result

    # 3. Explicit escalation triggers
    if _ESCALATION_PATTERNS.search(text):
        result.status = "escalated"
        result.escalation_reason = "Explicit escalation trigger detected"
        return result

    # 4. High risk AND PII → escalate
    if result.risk_level == "high" and pii_detected:
        result.status = "escalated"
        result.escalation_reason = "High-risk ticket with PII detected"
        return result

    # 5. No corpus results for a non-trivial ticket → escalate
    if no_corpus_results and request_type not in ("invalid",) and domains:
        result.status = "escalated"
        result.escalation_reason = "No corpus documentation found for this topic"
        return result

    # 6. Pure outage bug with no domain → escalate
    if request_type == "bug" and not domains:
        result.status = "escalated"
        result.escalation_reason = "Site-wide outage or bug report with no identifiable product"
        return result

    # 7. Needs verification but conditions met for reply
    if _NEEDS_VERIFICATION.search(text):
        result.needs_verification = True

    return result
