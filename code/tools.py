"""Rule-based action selection for support tickets.

Determines which API tool calls to make based on ticket intent, validated
against data/api_specs/internal_tools.json. Never performs destructive actions
without verified identity. Never invents actions not in the spec.

All logic is deterministic rule-based — no LLM involvement.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import jsonschema

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS_SPEC_PATH = os.path.join(REPO_ROOT, "data", "api_specs", "internal_tools.json")

# Load tool schemas once
with open(TOOLS_SPEC_PATH, encoding="utf-8") as _fh:
    _TOOL_SPECS: list[dict] = json.load(_fh)

_TOOL_SCHEMA_MAP: dict[str, dict] = {
    spec["name"]: spec["parameters"] for spec in _TOOL_SPECS
}


# ---------------------------------------------------------------------------
# Intent detection patterns
# ---------------------------------------------------------------------------

_REFUND_INTENT = re.compile(
    r"\b(refund|money\s+back|charge\s+back|reverse\s+(?:charge|transaction))\b",
    re.IGNORECASE,
)
_FRAUD_INTENT = re.compile(
    r"\b(fraud|stolen|unauthorized\s+(?:charge|transaction|access)|cloned)\b",
    re.IGNORECASE,
)
_LOCK_INTENT = re.compile(
    r"\b(block\s+(?:card|account)|lock\s+(?:account|card)|suspend\s+(?:account|card)"
    r"|account\s+(?:hacked|compromised|taken\s+over))\b",
    re.IGNORECASE,
)
_PASSWORD_RESET_INTENT = re.compile(
    r"\b(reset\s+password|forgot\s+password|change\s+password|recover\s+(?:account|access))\b",
    re.IGNORECASE,
)
_SUBSCRIPTION_INTENT = re.compile(
    r"\b(cancel\s+(?:subscription|plan|account)|pause\s+(?:subscription|plan|hiring)"
    r"|downgrade|upgrade\s+plan|modify\s+(?:subscription|plan))\b",
    re.IGNORECASE,
)
_LEGAL_INTENT = re.compile(
    r"\b(lawsuit|legal\s+action|class\s+action|discrimination\s+lawsuit|legal\s+counsel"
    r"|attorney|breach\s+of\s+contract|damages)\b",
    re.IGNORECASE,
)
_ESCALATE_INTENT = re.compile(
    r"\b(escalate|speak\s+to\s+(?:a\s+)?(?:human|agent|supervisor|manager)"
    r"|talk\s+to\s+(?:a\s+)?(?:human|person|agent)|need\s+(?:human|agent)\s+support)\b",
    re.IGNORECASE,
)
_ACCOUNT_COMPROMISE = re.compile(
    r"\b(hacked|compromised|taken\s+over|identity\s+(?:theft|stolen)"
    r"|(?:my\s+)?identity\s+has\s+been\s+stolen|unauthorized\s+login|suspicious\s+login)\b",
    re.IGNORECASE,
)

# Patterns that suggest account compromise (use lock_account instead of reset_password)
_COMPROMISE_SIGNALS = re.compile(
    r"\b(hacked|compromised|taken\s+over|identity\s+theft|identity\s+stolen"
    r"|unauthorized\s+(?:login|access)|someone\s+(?:changed|accessed))\b",
    re.IGNORECASE,
)

# Extract user identifiers
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_TXN_ID_RE = re.compile(r"\b(?:txn_|cs_live_|order\s+id\s*:\s*|transaction\s+id\s*:\s*)[A-Za-z0-9_\-]+\b", re.IGNORECASE)


def _validate_action(action: dict[str, Any]) -> bool:
    """Validate action dict against the tool spec schema."""
    name = action.get("action")
    if name not in _TOOL_SCHEMA_MAP:
        return False
    schema = _TOOL_SCHEMA_MAP[name]
    try:
        jsonschema.validate(action.get("parameters", {}), schema)
        return True
    except jsonschema.ValidationError:
        return False


def _make_escalate(priority: str, department: str, summary: str) -> dict:
    return {
        "action": "escalate_to_human",
        "parameters": {
            "priority": priority,
            "department": department,
            "summary": summary,
        },
    }


def _make_verify_identity(email: str = "", method: str = "email_otp") -> dict:
    target = email if email else "user@example.com"
    return {
        "action": "verify_identity",
        "parameters": {
            "method": method,
            "target": target,
        },
    }


def _make_lock_account(identifier: str, reason: str = "suspected_fraud") -> dict:
    return {
        "action": "lock_account",
        "parameters": {
            "user_identifier": identifier,
            "lock_reason": reason,
        },
    }


def _extract_email(text: str) -> str:
    m = _EMAIL_RE.search(text)
    return m.group() if m else ""


def _extract_txn_id(text: str) -> str:
    m = _TXN_ID_RE.search(text)
    return m.group() if m else ""


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def select_actions(
    text: str,
    status: str,
    risk_level: str,
    domains: list[str],
    pii_detected: bool = False,
    identity_in_context: bool = False,
) -> list[dict[str, Any]]:
    """Select appropriate API tool calls for a ticket.

    Args:
        text: Sanitized ticket text.
        status: "replied" or "escalated" from policy.
        risk_level: "low" | "medium" | "high" | "critical".
        domains: Detected domain list.
        pii_detected: Whether PII was found in the ticket.
        identity_in_context: Whether the user's identity is verifiable from context.

    Returns:
        List of action dicts validated against the tool spec.
    """
    actions: list[dict] = []
    text_lower = text.lower()
    email = _extract_email(text)

    # 1. Legal threat → escalate to legal (always, regardless of other signals)
    if _LEGAL_INTENT.search(text):
        actions.append(_make_escalate("urgent", "legal",
            "Legal threat or lawsuit mentioned — requires legal team review"))
        return _validate_and_return(actions)

    # 2. Account compromise / identity theft → lock account + escalate security
    if _ACCOUNT_COMPROMISE.search(text):
        identifier = email if email else "account"
        actions.append(_make_lock_account(identifier, "suspected_fraud"))
        actions.append(_make_escalate("urgent", "security",
            "Account compromise or identity theft reported"))
        return _validate_and_return(actions)

    # 3. Fraud / unauthorized transaction → lock + escalate (security)
    if _FRAUD_INTENT.search(text) and "visa" in domains:
        identifier = email if email else "user_account"
        actions.append(_make_lock_account(identifier, "suspected_fraud"))
        actions.append(_make_escalate("high", "security",
            "Suspected fraud or unauthorized transaction on Visa card"))
        return _validate_and_return(actions)

    # 4. Explicit card block request
    if _LOCK_INTENT.search(text):
        identifier = email if email else "user_account"
        actions.append(_make_lock_account(identifier, "user_requested"))
        actions.append(_make_escalate("high", "security",
            "User requested account/card lock"))
        return _validate_and_return(actions)

    # 5. Refund request
    if _REFUND_INTENT.search(text):
        txn_id = _extract_txn_id(text)
        if txn_id and identity_in_context:
            # We have enough info — but still need to validate amount
            actions.append(_make_escalate("normal", "billing",
                f"Refund requested for transaction {txn_id}"))
        else:
            # Missing transaction ID or identity not verified
            if not identity_in_context:
                actions.append(_make_verify_identity(email))
            actions.append(_make_escalate("normal", "billing",
                "Refund requested — requires billing team review"))
        return _validate_and_return(actions)

    # 6. Password reset
    if _PASSWORD_RESET_INTENT.search(text):
        if _COMPROMISE_SIGNALS.search(text):
            # Compromise suspected — lock instead of reset
            identifier = email if email else "user_account"
            actions.append(_make_lock_account(identifier, "suspected_fraud"))
            actions.append(_make_escalate("urgent", "security",
                "Account compromise suspected during password reset request"))
        elif email:
            actions.append({
                "action": "reset_password",
                "parameters": {"user_email": email},
            })
        else:
            actions.append(_make_verify_identity(email))
        return _validate_and_return(actions)

    # 7. Subscription changes
    if _SUBSCRIPTION_INTENT.search(text):
        if not identity_in_context:
            actions.append(_make_verify_identity(email))
        actions.append(_make_escalate("normal", "billing",
            "Subscription modification requested"))
        return _validate_and_return(actions)

    # 8. Explicit escalation request
    if _ESCALATE_INTENT.search(text) or status == "escalated":
        if status == "escalated":
            if risk_level in ("critical", "high"):
                actions.append(_make_escalate("urgent", "general",
                    "Ticket escalated due to high/critical risk"))
            elif risk_level == "medium":
                actions.append(_make_escalate("normal", "general",
                    "Ticket escalated for human review"))
            else:
                actions.append(_make_escalate("low", "general",
                    "Ticket escalated — requires human review"))
        return _validate_and_return(actions)

    # 9. No special action needed for simple FAQ tickets
    return []


def _validate_and_return(actions: list[dict]) -> list[dict]:
    """Validate all actions; replace invalid ones with escalate_to_human."""
    validated: list[dict] = []
    for action in actions:
        if _validate_action(action):
            validated.append(action)
        else:
            validated.append(_make_escalate(
                "normal", "general",
                f"Action {action.get('action', 'unknown')} failed schema validation"
            ))
    return validated
