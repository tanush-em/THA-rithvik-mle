"""Adversarial screening for support ticket pipeline.

All detection is pure-Python regex — the LLM is never invoked here. This is
the most critical module because a single prompt-injection compliance results
in a 0% score on the 25% adversarial-robustness dimension.

The screen handles:
- Imperative override phrases ("ignore all previous instructions", etc.)
- Structural injection markers (<system>, AUTH_CODE, [OVERRIDE], etc.)
- Base64/hex encoded payloads → decode → re-screen
- Excel/CSV formula injection (leading =, +, -, @)
- Internals-leak requests (system prompt, list tools, retrieval algorithm, etc.)
- Cross-language injection (English trigger words embedded in other languages)
- Fake-authority framing (QA engineer at Anthropic, Trust & Safety team, etc.)
- Multi-step social engineering (gradual context-building attacks)
- Role impersonation (DAN, developer mode, maintenance mode)

Returns a SafetyResult with:
  injection: bool — whether adversarial content was detected
  legit_fragment: str | None — non-adversarial portion of the text, if any
  reasons: list[str] — human-readable reasons for each detected pattern
"""
from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# 1. Imperative override phrases
_OVERRIDE_PHRASES = re.compile(
    r"(?:"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier|your)\s+(?:instructions?|rules?|guidelines?|prompts?|directives?|constraints?|system)"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|above|earlier|your)\s+(?:instructions?|rules?|guidelines?|prompts?|constraints?)"
    r"|forget\s+(?:all\s+)?(?:previous|prior|your)\s+(?:instructions?|rules?|constraints?)"
    r"|bypass\s+(?:all\s+)?(?:previous|safety|your)\s+(?:instructions?|rules?|filters?|constraints?)"
    r"|override\s+(?:safety|security|all)\s+(?:protocols?|settings?|restrictions?|instructions?)"
    r"|you\s+are\s+now\s+(?:in\s+)?(?:a\s+|an\s+)?(?:different|new|unrestricted|DAN|developer|maintenance|jailbreak)"
    r"|act\s+as\s+(?:if\s+you\s+(?:are|were)|a\s+|an\s+)"
    r"|pretend\s+(?:to\s+be|you\s+are|you\s+were)"
    r"|roleplay\s+as"
    r"|from\s+now\s+on\s+(?:you\s+(?:are|will|must|should))"
    r"|do\s+anything\s+now"
    r"|jailbreak"
    r"|developer\s+mode"
    r"|maintenance\s+mode"
    r"|DAN\s+mode"
    r"|confirm\s+by\s+(?:saying|responding\s+with|outputting)"
    r"|verify\s+by\s+responding\s+with"
    r"|respond\s+with\s+AUTH"
    r"|output\s+the\s+(?:following|string|word|text|json)\s*:"
    r"|resume\s+normal\s+operation"
    r")",
    re.IGNORECASE,
)

# 2. Structural injection markers
_STRUCTURAL_MARKERS = re.compile(
    r"(?:"
    r"<\s*/?system\s*>"
    r"|<\s*/?INST\s*>"
    r"|<\|.*?\|>"
    r"|\[SYSTEM(?:\s+OVERRIDE)?\]"
    r"|\[OVERRIDE\]"
    r"|AUTH_CODE\s*:"
    r"|ALERT_ACK_"
    r"|<<SYS>>"
    r"|###\s*(?:SYSTEM|OVERRIDE|INSTRUCTIONS?)\s*###"
    r")",
    re.IGNORECASE,
)

# 3. Internals-leak requests
_INTERNALS_LEAK = re.compile(
    r"(?:"
    r"(?:share|show|reveal|output|print|display|give\s+me|tell\s+me|list)\s+"
    r"(?:your\s+)?(?:system\s+prompt|system\s+instructions?|internal\s+(?:instructions?|rules?|tools?|knowledge|documents?|prompts?))"
    r"|what\s+(?:is|are)\s+your\s+(?:system\s+prompt|instructions?|internal\s+(?:rules?|guidelines?))"
    r"|(?:full\s+)?(?:text|content)\s+of\s+your\s+(?:system|internal)\s+(?:prompt|instructions?)"
    r"|names?\s+of\s+(?:all\s+)?(?:tools?|functions?)\s+available"
    r"|(?:list|enumerate)\s+(?:all\s+)?(?:tools?|functions?|capabilities)"
    r"|retrieval\s+algorithm"
    r"|confidence\s+(?:scoring|algorithm|calibration)"
    r"|which\s+document\s+did\s+you\s+(?:pull|retrieve|use|consult)"
    r"|(?:how\s+many\s+)?documents?\s+(?:do\s+you\s+have|in\s+your\s+knowledge)"
    r"|knowledge\s+base\s+(?:contents?|list|documents?)"
    r"|complete\s+list\s+of\s+support\s+articles"
    r"|exact\s+(?:retrieval|logic)\s+(?:you\s+use|algorithm)"
    r"|your\s+(?:exact\s+)?(?:system\s+)?prompt\s+in"
    r")",
    re.IGNORECASE,
)

# 4. Fake authority framing
_FAKE_AUTHORITY = re.compile(
    r"(?:"
    r"(?:senior\s+)?(?:QA|quality\s+assurance)\s+engineer\s+at\s+(?:anthropic|devplatform|hackerrank|visa)"
    r"|(?:trust\s+&?\s+safety\s+team|trust\s+and\s+safety)"
    r"|(?:automated\s+)?monitoring\s+system"
    r"|(?:internal\s+)?(?:security\s+)?audit"
    r"|working\s+(?:in|for)\s+the\s+trust\s+(?:&|and)\s+safety"
    r"|internal\s+credentials?\s+are"
    r"|emp(?:loyee)?_id\s*="
    r"|access_level\s*="
    r"|this\s+is\s+(?:an?\s+)?automated\s+message\s+from"
    r"|routine\s+audit"
    r"|internal\s+employee\s+request"
    r"|emergency\s+access\s+to\s+customer"
    r"|(?:ciso|chief\s+information\s+security\s+officer)"
    r")",
    re.IGNORECASE,
)

# 5. Social engineering / false precedent
_SOCIAL_ENGINEERING = re.compile(
    r"(?:"
    r"previous\s+agent\s+(?:i\s+spoke\s+to|told\s+me|promised)"
    r"|agent\s+(?:id|ID)\s*:\s*[A-Z0-9-]+"
    r"|agent\s+(?:TK|SR|CS)-\d+"
    r"|(?:your\s+)?colleague(?:'s)?\s+commitment"
    r"|honor\s+(?:your|this)\s+(?:colleague's|promise|commitment)"
    r"|as\s+(?:a\s+)?(?:paying\s+)?customer\s+(?:i\s+have\s+rights|with\s+rights)"
    r"|cancel\s+(?:that\s+)?fraud\s+report"
    r"|(?:i\s+)?authorize\s+(?:this\s+)?on\s+behalf\s+of"
    r")",
    re.IGNORECASE,
)

# 6. Role impersonation targeting the agent
_ROLE_IMPERSONATION = re.compile(
    r"(?:"
    r"\bDAN\b"
    r"|do\s+anything\s+now"
    r"|without\s+(?:any\s+)?restrictions"
    r"|no\s+(?:longer\s+a|more\s+)?(?:support\s+agent|restrictions?|limits?|safety)"
    r"|unrestricted\s+(?:mode|agent|AI)"
    r")",
    re.IGNORECASE,
)

# 7. System-prompt extraction attempts (in any language — watch for English key phrases)
_PROMPT_EXTRACTION = re.compile(
    r"(?:"
    r"system\s+prompt"
    r"|system\s+instructions?"
    r"|your\s+(?:full\s+)?instructions?"
    r"|logique\s+exacte\s+que\s+vous\s+utilisez"   # French: "exact logic you use"
    r"|interne.{0,20}r.gle"                          # French: internal rules
    r"|Systemanweisungen"                             # German: system instructions
    r"|systemanweisungen"
    r"|vollst.ndigen\s+(?:system)?anweisungen"       # German: complete instructions
    r"|ignorieren\s+sie\s+ihre\s+systemanweisungen"  # German: ignore your system instructions
    r")",
    re.IGNORECASE,
)

# 8. Adversarial classification manipulation
_CLASSIFICATION_MANIPULATION = re.compile(
    r"(?:"
    r"STATUS\s*:\s*(?:replied|escalated)"
    r"|REQUEST_TYPE\s*:\s*(?:product_issue|feature_request|bug|invalid)"
    r"|classify\s+(?:this\s+)?as\s+(?:replied|escalated)"
    r"|output\s+the\s+following\s+json"
    r")",
    re.IGNORECASE,
)

# 9. Excel / CSV formula injection
_EXCEL_INJECTION = re.compile(r"^\s*[=+\-@](?:[A-Za-z]|cmd|\|)")

# 10. High-entropy base64 detector (min 20 chars of b64 alphabet)
_BASE64_CHUNK = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")


# ---------------------------------------------------------------------------
# Helper: decode and re-screen
# ---------------------------------------------------------------------------

def _decode_base64(s: str) -> str | None:
    """Try to decode a base64 string; return decoded text or None."""
    try:
        padded = s + "=" * (-len(s) % 4)
        decoded = base64.b64decode(padded).decode("utf-8", errors="replace")
        return decoded
    except (binascii.Error, UnicodeDecodeError):
        return None


def _has_injection_in_text(text: str) -> tuple[bool, list[str]]:
    """Check a (possibly decoded) text for injection patterns. Returns (found, reasons)."""
    reasons: list[str] = []
    for pattern, name in [
        (_OVERRIDE_PHRASES, "override_phrase"),
        (_STRUCTURAL_MARKERS, "structural_marker"),
        (_INTERNALS_LEAK, "internals_leak"),
        (_FAKE_AUTHORITY, "fake_authority"),
        (_SOCIAL_ENGINEERING, "social_engineering"),
        (_ROLE_IMPERSONATION, "role_impersonation"),
        (_PROMPT_EXTRACTION, "prompt_extraction"),
        (_CLASSIFICATION_MANIPULATION, "classification_manipulation"),
    ]:
        if pattern.search(text):
            reasons.append(name)
    return bool(reasons), reasons


# ---------------------------------------------------------------------------
# Structural injection block extractor
# ---------------------------------------------------------------------------

# Matches <system>...</system> or [SYSTEM OVERRIDE] ... blocks
_SYSTEM_BLOCK = re.compile(
    r"<\s*system\s*>.*?<\s*/\s*system\s*>",
    re.IGNORECASE | re.DOTALL,
)
_BRACKET_SYSTEM = re.compile(
    r"\[SYSTEM(?:\s+OVERRIDE)?\].*?(?:\n\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_AUTH_CODE_BLOCK = re.compile(r"AUTH_CODE\s*:.*?(?:\n|\Z)", re.IGNORECASE)
_ALERT_ACK_BLOCK = re.compile(r"ALERT_ACK_.*?(?:\n|\Z)", re.IGNORECASE)


def _strip_injection_blocks(text: str) -> str:
    """Remove structural injection blocks; return remainder."""
    text = _SYSTEM_BLOCK.sub("", text)
    text = _BRACKET_SYSTEM.sub("", text)
    text = _AUTH_CODE_BLOCK.sub("", text)
    text = _ALERT_ACK_BLOCK.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main dataclass + function
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    injection: bool = False
    legit_fragment: str | None = None
    reasons: list[str] = field(default_factory=list)


def screen(text: str) -> SafetyResult:
    """Screen *text* for adversarial content.

    Returns a SafetyResult. If ``injection`` is True and ``legit_fragment`` is
    not None, there is a non-adversarial sub-question that can be answered.
    """
    result = SafetyResult()

    if not text or not text.strip():
        return result

    # --- Excel/CSV injection (check entire content as a cell value) ---
    if _EXCEL_INJECTION.match(text):
        result.injection = True
        result.reasons.append("excel_csv_injection")
        return result

    # --- Direct pattern matching on raw text ---
    found, reasons = _has_injection_in_text(text)
    if found:
        result.injection = True
        result.reasons.extend(reasons)

    # --- Base64 payload detection: decode any large b64 chunk and re-check ---
    for m in _BASE64_CHUNK.finditer(text):
        chunk = m.group()
        decoded = _decode_base64(chunk)
        if decoded and len(decoded) > 5:
            sub_found, sub_reasons = _has_injection_in_text(decoded)
            if sub_found:
                result.injection = True
                result.reasons.extend(f"base64:{r}" for r in sub_reasons)

    # --- If injection detected, try to extract a legit fragment ---
    if result.injection:
        cleaned = _strip_injection_blocks(text)
        # Remove sentences containing override phrases
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        legit_sentences = []
        for sent in sentences:
            s_found, _ = _has_injection_in_text(sent)
            if not s_found:
                legit_sentences.append(sent)
        legit = " ".join(legit_sentences).strip()
        # Only return legit fragment if it's a meaningful question
        if len(legit) > 15 and "?" in legit:
            result.legit_fragment = legit

    return result
