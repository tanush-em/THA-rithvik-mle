"""PII detection and redaction for support tickets.

All detection is regex + Luhn — no LLM. This module must be called early in
the pipeline so that downstream components (including the LLM) never see raw PII.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Luhn checksum
# ---------------------------------------------------------------------------

def _luhn_valid(digits: str) -> bool:
    total = 0
    reverse = digits[::-1]
    for i, ch in enumerate(reverse):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Credit/debit card: 13–19 digits optionally separated by spaces, dashes, or Xs
# We also handle partially-masked forms like 4532-XXXX-XXXX-8901
_CARD_RAW = re.compile(
    r"\b(?:\d[ -]?){12,18}\d\b"
)
_CARD_MASKED = re.compile(
    r"\b\d{4}[-\s]?(?:X{4}[-\s]?){1,3}\d{4}\b",
    re.IGNORECASE,
)

# US SSN
_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")

# Email
_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# Phone: E.164 or national variants; at least 7 digits, various separators
_PHONE = re.compile(
    r"(?<!\d)"
    r"(?:\+?1[-.\s]?)?"
    r"(?:\(?\d{3}\)?[-.\s]?)?"
    r"\d{3}[-.\s]?\d{4}"
    r"(?!\d)"
)

# IBAN: 2-letter country code + 2 check digits + up to 30 alphanumeric
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b")

# IPv4
_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# Date of birth heuristic: MM/DD/YYYY or DD/MM/YYYY or YYYY-MM-DD
_DOB = re.compile(
    r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b"
)

# Street address: number + street name + suffix
_STREET_SUFFIXES = (
    r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|"
    r"court|ct|place|pl|way|terrace|ter|circle|cir|highway|hwy|parkway|pkwy)"
)
_ADDRESS = re.compile(
    rf"\b\d{{1,5}}\s+(?:[A-Za-z]{{2,}}\s+){{1,3}}{_STREET_SUFFIXES}\b",
    re.IGNORECASE,
)

# US ZIP (standalone, with optional +4)
_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PiiResult:
    detected: bool = False
    types: list[str] = field(default_factory=list)
    # mapping: original span → generic replacement
    redaction_map: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def detect_and_redact(text: str) -> PiiResult:
    """Detect PII in *text* and return a PiiResult with a redaction map.

    The redaction_map maps the exact matched string to a generic reference so
    callers can produce a safe version of the text by calling ``redact(text)``.
    """
    result = PiiResult()
    redact: dict[str, str] = {}

    # --- masked cards first (before raw card so they don't partially collide) ---
    for m in _CARD_MASKED.finditer(text):
        span = m.group()
        # Extract last 4 digits
        digits_only = re.sub(r"[^0-9]", "", span)
        last4 = digits_only[-4:] if len(digits_only) >= 4 else "XXXX"
        replacement = f"card ending in {last4}"
        redact[span] = replacement
        result.types.append("credit_card_masked")

    # --- raw card numbers ---
    for m in _CARD_RAW.finditer(text):
        span = m.group()
        # Skip if already in redact map (masked form matched earlier)
        if any(span in k or k in span for k in redact):
            continue
        digits = re.sub(r"[^0-9X]", "", span.upper())
        numeric_digits = re.sub(r"[^0-9]", "", span)
        if len(numeric_digits) < 13:
            continue
        if _luhn_valid(numeric_digits):
            last4 = numeric_digits[-4:]
            replacement = f"card ending in {last4}"
            redact[span] = replacement
            result.types.append("credit_card")

    # --- SSN ---
    for m in _SSN.finditer(text):
        span = m.group()
        if span not in redact:
            redact[span] = "SSN [REDACTED]"
            result.types.append("ssn")

    # --- email ---
    for m in _EMAIL.finditer(text):
        span = m.group()
        if span not in redact:
            redact[span] = "[email redacted]"
            result.types.append("email")

    # --- phone ---
    for m in _PHONE.finditer(text):
        span = m.group().strip()
        digits = re.sub(r"\D", "", span)
        if len(digits) < 7:
            continue
        if span not in redact:
            redact[span] = "[phone redacted]"
            result.types.append("phone")

    # --- address ---
    for m in _ADDRESS.finditer(text):
        span = m.group()
        if span not in redact:
            redact[span] = "[address redacted]"
            result.types.append("address")

    # --- DOB ---
    for m in _DOB.finditer(text):
        span = m.group()
        # Only flag if near DOB-like words to avoid false positives
        start = max(0, m.start() - 30)
        context = text[start: m.end() + 10].lower()
        if any(kw in context for kw in ("dob", "birth", "born", "birthday", "age", "date of")):
            if span not in redact:
                redact[span] = "[date redacted]"
                result.types.append("dob")

    # --- IBAN ---
    for m in _IBAN.finditer(text):
        span = m.group()
        if span not in redact:
            redact[span] = "[account number redacted]"
            result.types.append("iban")

    result.redaction_map = redact
    result.detected = bool(redact)
    return result


def redact(text: str, pii_result: PiiResult) -> str:
    """Apply redaction_map to text, replacing PII with generic references.

    Replacements are applied longest-match-first to avoid partial overlaps.
    """
    if not pii_result.redaction_map:
        return text

    # Sort by length descending so longer matches take precedence
    sorted_entries = sorted(
        pii_result.redaction_map.items(), key=lambda kv: len(kv[0]), reverse=True
    )
    for original, replacement in sorted_entries:
        text = text.replace(original, replacement)
    return text


def normalize(text: str) -> str:
    """NFKC normalize and strip control/zero-width characters."""
    text = unicodedata.normalize("NFKC", text)
    # Strip zero-width chars, BiDi overrides
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)
    return text
