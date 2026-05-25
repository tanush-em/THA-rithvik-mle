"""Tests for classify.py and policy.py against sample_support_tickets.csv cases."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classify import classify
from policy import assess


# --- Classifier tests ---

def test_devplatform_test_expiration():
    text = "I notice that people I assigned the test in October of 2025 have not received new tests. How long do the tests stay active in the system."
    r = classify(text, company="DevPlatform")
    assert "devplatform" in r.domains
    assert r.primary_domain == "devplatform"
    assert r.request_type == "product_issue"


def test_claude_delete_conversation():
    text = "One of my claude conversations has some private info, i forgot to make a temporary chat, is there anything else that can be done? like delete etc?"
    r = classify(text, company="Claude")
    assert "claude" in r.domains
    assert r.primary_domain == "claude"


def test_visa_stolen_cheque():
    text = "I bought Visa Traveller's Cheques from Citicorp and they were stolen in Lisbon last night. What do I do?"
    r = classify(text, company="Visa")
    assert "visa" in r.domains
    assert r.primary_domain == "visa"


def test_out_of_scope_iron_man():
    text = "What is the name of the actor in Iron Man?"
    r = classify(text, company="None")
    assert r.request_type == "invalid"
    assert r.is_out_of_scope


def test_greeting_is_invalid():
    text = "Thank you for helping me"
    r = classify(text, company="None")
    assert r.request_type == "invalid"


def test_site_down_bug():
    text = "site is down & none of the pages are accessible"
    r = classify(text, company="None")
    assert r.request_type == "bug"


def test_devplatform_bug_submission_failing():
    text = "none of the submissions across any challenges are working on your website"
    r = classify(text, company="DevPlatform")
    assert r.request_type == "bug"
    assert "devplatform" in r.domains


def test_company_field_weak_prior_overridden():
    # If company says "Claude" but content is clearly about Visa card
    text = "My Visa card was blocked while I was traveling in Japan."
    r = classify(text, company="Claude")
    # Visa should win (score 3.0 for "visa" + 3.0 for "visa card") vs Claude (+2 from company)
    assert "visa" in r.domains


def test_multi_domain_ticket():
    text = (
        "Three things: 1) My DevPlatform test link expired. "
        "2) I tried to pay using my Visa card and the payment failed. "
        "3) Claude is giving me errors when I try to upload PDFs."
    )
    r = classify(text, company="None")
    assert "devplatform" in r.domains
    assert "visa" in r.domains
    assert "claude" in r.domains


def test_devplatform_google_login_delete():
    text = "i signed up using google login on devplatform community, so i do not have a separate devplatform password. please delete my account"
    r = classify(text, company="DevPlatform")
    assert "devplatform" in r.domains


# --- Policy tests ---

def test_policy_legal_threat_escalated():
    text = "I WILL FILE A CLASS ACTION LAWSUIT AGAINST DEVPLATFORM FOR DISCRIMINATION. I DEMAND $500,000 IN DAMAGES."
    result = assess(text, "product_issue", ["devplatform"])
    assert result.status == "escalated"
    assert result.risk_level in ("critical",)


def test_policy_lost_card_with_pii_escalated():
    text = "My card ending in 8901 was used for unauthorized transactions. Please block it."
    result = assess(text, "product_issue", ["visa"], pii_detected=True)
    assert result.status == "escalated"


def test_policy_site_down_no_domain_escalated():
    text = "site is down & none of the pages are accessible"
    result = assess(text, "bug", [])
    assert result.status == "escalated"


def test_policy_simple_faq_replied():
    text = "How do I reset my DevPlatform password?"
    result = assess(text, "product_issue", ["devplatform"])
    assert result.status == "replied"
    assert result.risk_level == "low"


def test_policy_subscription_cancel_replied_with_verification():
    text = "Hi, please pause our subscription. We have stopped all hiring efforts for now."
    result = assess(text, "product_issue", ["devplatform"])
    assert result.status == "replied"
    assert result.needs_verification


def test_policy_identity_theft_critical():
    text = "My identity has been stolen, what should I do"
    result = assess(text, "product_issue", ["visa"])
    assert result.status == "escalated"
    assert result.risk_level == "critical"


def test_policy_gdpr_erasure_escalated():
    text = "I want a legally binding confirmation that ALL my data has been permanently deleted under GDPR Article 17 Right to Erasure."
    result = assess(text, "product_issue", ["claude"])
    assert result.status == "escalated"


def test_policy_security_vulnerability_critical():
    text = "I have found a major security vulnerability in Claude, what are the next steps"
    result = assess(text, "bug", ["claude"])
    assert result.status == "escalated"
    assert result.risk_level == "critical"
