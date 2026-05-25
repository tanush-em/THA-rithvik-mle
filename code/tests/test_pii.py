"""Unit tests for code/pii.py"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pii import detect_and_redact, redact, normalize


def test_credit_card_luhn():
    text = "My card is 4111111111111111"
    r = detect_and_redact(text)
    assert r.detected
    assert "credit_card" in r.types
    assert "4111111111111111" in r.redaction_map
    assert "1111" in r.redaction_map["4111111111111111"]


def test_credit_card_with_dashes():
    text = "Card number 4111-1111-1111-1111 was stolen"
    r = detect_and_redact(text)
    assert r.detected
    assert any("credit_card" in t for t in r.types)


def test_credit_card_masked_form():
    text = "My Visa card ending in 4532-XXXX-XXXX-8901 was used for two unauthorized transactions"
    r = detect_and_redact(text)
    assert r.detected
    assert "credit_card_masked" in r.types
    safe = redact(text, r)
    assert "4532-XXXX-XXXX-8901" not in safe
    assert "8901" in safe  # generic reference keeps last 4


def test_ssn():
    text = "My SSN is 287-65-4321 and my DOB is 03/15/1989."
    r = detect_and_redact(text)
    assert r.detected
    assert "ssn" in r.types
    safe = redact(text, r)
    assert "287-65-4321" not in safe


def test_email():
    text = "Contact me at john.doe@example.com please"
    r = detect_and_redact(text)
    assert r.detected
    assert "email" in r.types
    safe = redact(text, r)
    assert "john.doe@example.com" not in safe


def test_phone():
    text = "Call me at +86 138 XXXX 7890 for help"
    # The XXXX part means it won't match the digit regex cleanly — that's fine
    # Test with a real phone number
    text2 = "My number is 555-867-5309"
    r = detect_and_redact(text2)
    assert r.detected
    assert "phone" in r.types


def test_address():
    # Full address as it appears in the visible ticket set
    text = "My home address is 1847 Oak Street, Springfield, IL 62704."
    r = detect_and_redact(text)
    assert r.detected
    assert "address" in r.types
    safe = redact(text, r)
    assert "1847 Oak Street" not in safe


def test_no_pii():
    text = "How do I reset my password?"
    r = detect_and_redact(text)
    assert not r.detected


def test_invalid_luhn_ignored():
    text = "Reference number 1234567890123456"
    r = detect_and_redact(text)
    # 1234567890123456 fails Luhn
    assert "credit_card" not in r.types


def test_normalize_strips_zero_width():
    text = "hello\u200bworld"
    result = normalize(text)
    assert "\u200b" not in result
    assert "hello" in result
    assert "world" in result


def test_redact_applies_longest_first():
    """Ensure longer matches are replaced before shorter partial matches."""
    text = "card 4532-XXXX-XXXX-8901 plus 4532"
    r = detect_and_redact(text)
    safe = redact(text, r)
    assert "4532-XXXX-XXXX-8901" not in safe


def test_multiple_pii_types():
    text = (
        "My card ending in 4532-XXXX-XXXX-8901, SSN is 287-65-4321, "
        "email zhang.wei@example.com"
    )
    r = detect_and_redact(text)
    assert r.detected
    assert len(r.types) >= 3
    safe = redact(text, r)
    assert "287-65-4321" not in safe
    assert "zhang.wei@example.com" not in safe
