"""Domain and request-type classification using weighted keyword maps.

The `company` field from the CSV is treated as a *weak prior* (+2 points) but
can be overridden by strong content signals. This prevents misclassification
when the company field is deliberately wrong (as the spec warns).

All decisions are deterministic — no LLM calls here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Domain keyword maps
# Higher weight = stronger signal for that domain
# ---------------------------------------------------------------------------

_DEVPLATFORM_KEYWORDS: dict[str, float] = {
    # Product names
    "devplatform": 3.0,
    "hackerrank": 3.0,
    "codepair": 3.0,
    "codescreen": 3.0,
    "codingscreen": 2.0,
    # DevPlatform-specific features
    "assessment": 2.0,
    "candidates": 1.5,
    "candidate": 1.5,
    "test link": 2.0,
    "test variant": 2.0,
    "test settings": 2.0,
    "proctor": 2.0,
    "proctoring": 2.0,
    "time accommodation": 2.5,
    "interviewer": 1.5,
    "recruiter": 1.5,
    "hiring": 1.5,
    "resume builder": 2.0,
    "mock interview": 2.0,
    "library": 1.0,
    "challenge": 1.0,
    "leaderboard": 1.5,
    "badge": 1.0,
    # DevPlatform-specific URLs/domains
    "support.devplatform.com": 3.0,
    "devplatform.com": 2.5,
    "api.devplatform.com": 2.5,
}

_CLAUDE_KEYWORDS: dict[str, float] = {
    # Product names
    "claude": 3.0,
    "anthropic": 3.0,
    "claude pro": 3.0,
    "claude team": 3.0,
    "claude max": 3.0,
    # Claude-specific features
    "projects": 1.5,
    "artifacts": 1.5,
    "claude code": 2.5,
    "aws bedrock": 2.0,
    "bedrock": 2.0,
    "claude api": 2.5,
    "model": 1.0,
    "system prompt": 1.5,
    "lti": 2.0,
    "incognito chat": 2.0,
    "claude.ai": 3.0,
    "support.claude.com": 3.0,
    "support.anthropic.com": 3.0,
    "anthropic.com": 2.5,
    "anthropic-billing.com": 2.0,
    "claude for education": 2.5,
    "claude for government": 2.5,
    "sso": 1.0,
    "scim": 1.5,
    "gdpr": 1.5,
    "baa": 2.0,
    "hipaa": 2.0,
    "bug bounty": 2.0,
    "cowork": 2.0,
    "skills": 1.0,
    "connectors": 1.5,
}

_VISA_KEYWORDS: dict[str, float] = {
    # Product names
    "visa": 3.0,
    "visa card": 3.0,
    "visa infinite": 2.5,
    "visa signature": 2.5,
    "visa platinum": 2.5,
    # Visa-specific features
    "chargeback": 3.0,
    "dispute": 2.0,
    "zero liability": 2.5,
    "traveller's cheque": 3.0,
    "travelers cheque": 3.0,
    "traveler cheque": 3.0,
    "cheque": 2.0,
    "card blocked": 2.5,
    "card cloned": 2.5,
    "unauthorized transaction": 2.5,
    "fraud": 2.0,
    "cash advance": 2.0,
    "atm": 1.5,
    "pin": 1.5,
    "issuer": 1.5,
    "merchant": 1.5,
    "payment network": 2.0,
    "visa.co.in": 3.0,
    "visa.com": 2.5,
    "mastercard": 1.5,
    "citicorp": 2.0,
    "refund": 1.5,
    "exchange rate": 1.5,
    "minimum transaction": 2.0,
    "checkout fee": 1.5,
}

# Company field value → domain name
_COMPANY_DOMAIN_MAP: dict[str, str] = {
    "devplatform": "devplatform",
    "claude": "claude",
    "visa": "visa",
}


# ---------------------------------------------------------------------------
# Request type keywords
# ---------------------------------------------------------------------------

_BUG_KEYWORDS = re.compile(
    r"\b("
    r"not\s+working|doesn'?t\s+work|isn'?t\s+working|stopped\s+working"
    r"|down\b|outage|crash(?:ed)?|error\b|bug\b|broken|fail(?:ed|ing)?"
    r"|500\s+error|internal\s+server\s+error|can'?t\s+(?:access|connect|load|open)"
    r"|not\s+(?:loading|accessible|responding|available)|glitch|freeze|hung"
    r"|none\s+of\s+the\s+\w+\s+(?:are|is)\s+working"
    r"|(?:submissions?|pages?|requests?|challenges?)\s+(?:are|is|are\s+not|is\s+not)\s+working"
    r")\b",
    re.IGNORECASE,
)

_FEATURE_REQUEST_KEYWORDS = re.compile(
    r"\b("
    r"feature\s+request|would\s+(?:be\s+)?(?:nice|great|helpful)\s+(?:to|if)"
    r"|please\s+add|can\s+you\s+add|wish\s+(?:you\s+)?(?:had|would)"
    r"|suggestion|suggest(?:ion)?|recommend(?:ation)?"
    r"|when\s+will\s+(?:you|it)|roadmap|future\s+plan"
    r")\b",
    re.IGNORECASE,
)

_INVALID_PATTERNS = re.compile(
    r"^(?:"
    r"(?:hi|hello|hey|thanks?|thank\s+you|cheers|bye|goodbye|good\s+(?:morning|afternoon|evening|day))"
    r"(?:\s+(?:there|everyone|team|support|for\s+helping\s+me|:?))?"
    r"[\s!.,]*"
    r")$",
    re.IGNORECASE,
)

_OUT_OF_SCOPE_PATTERNS = re.compile(
    r"\b("
    r"iron\s+man|actor|movie|film|celebrity|sport|recipe|weather|stock\s+price"
    r"|write\s+(?:me\s+)?(?:a\s+)?(?:python|code|script|program)"
    r"|(?:job\s+application|hiring\s+(?:team|manager)|recruitment|apply\s+for)"
    r"|financial\s+advis(?:e|or|ory)|investment\s+advice|cryptocurrency|crypto"
    r"|cash\s+advance\s+(?:to\s+invest|for\s+investment)"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    domains: list[str] = field(default_factory=list)   # detected domain(s)
    primary_domain: str = ""                             # single best domain or ""
    request_type: str = "product_issue"                  # product_issue|feature_request|bug|invalid
    is_out_of_scope: bool = False
    domain_scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def _score_text(text: str, keywords: dict[str, float]) -> float:
    """Score text against a keyword map; case-insensitive partial matching."""
    text_lower = text.lower()
    score = 0.0
    for kw, weight in keywords.items():
        if kw.lower() in text_lower:
            score += weight
    return score


def classify(
    text: str,
    company: str = "",
    subject: str = "",
) -> ClassificationResult:
    """Classify a ticket into domain(s) and request_type.

    Args:
        text: The sanitized ticket text (full conversation history concatenated).
        company: The raw `company` field from CSV (weak prior).
        subject: The `subject` field from CSV (weak prior, may be misleading).

    Returns:
        ClassificationResult with domains, primary_domain, request_type.
    """
    combined = f"{text} {subject}"

    # --- Domain scoring ---
    dp_score = _score_text(combined, _DEVPLATFORM_KEYWORDS)
    cl_score = _score_text(combined, _CLAUDE_KEYWORDS)
    vi_score = _score_text(combined, _VISA_KEYWORDS)

    # Apply company field weak prior (+2)
    company_lower = company.strip().lower()
    mapped = _COMPANY_DOMAIN_MAP.get(company_lower, "")
    if mapped == "devplatform":
        dp_score += 2.0
    elif mapped == "claude":
        cl_score += 2.0
    elif mapped == "visa":
        vi_score += 2.0

    domain_scores = {
        "devplatform": dp_score,
        "claude": cl_score,
        "visa": vi_score,
    }

    # Determine active domains (score > threshold)
    threshold = 1.5
    active_domains = sorted(
        [d for d, s in domain_scores.items() if s >= threshold],
        key=lambda d: -domain_scores[d],
    )

    primary_domain = active_domains[0] if active_domains else ""

    # --- Request type ---
    # Check for out-of-scope first (no domain signals + known OOS patterns)
    is_out_of_scope = False
    if not active_domains and _OUT_OF_SCOPE_PATTERNS.search(text):
        is_out_of_scope = True
        request_type = "invalid"
    elif not active_domains and _INVALID_PATTERNS.match(text.strip()):
        is_out_of_scope = True
        request_type = "invalid"
    elif _INVALID_PATTERNS.match(text.strip()):
        request_type = "invalid"
        is_out_of_scope = True
    elif _BUG_KEYWORDS.search(text):
        request_type = "bug"
    elif _FEATURE_REQUEST_KEYWORDS.search(text):
        request_type = "feature_request"
    else:
        request_type = "product_issue"

    return ClassificationResult(
        domains=active_domains,
        primary_domain=primary_domain,
        request_type=request_type,
        is_out_of_scope=is_out_of_scope,
        domain_scores=domain_scores,
    )
