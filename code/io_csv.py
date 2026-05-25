"""Deterministic CSV read/write for the support triage pipeline."""
from __future__ import annotations

import csv
import json
import os
import tempfile
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_CSV = os.path.join(REPO_ROOT, "support_tickets", "support_tickets.csv")
OUTPUT_CSV = os.path.join(REPO_ROOT, "support_tickets", "output.csv")

OUTPUT_COLUMNS = [
    "issue", "subject", "company",
    "response", "product_area", "status", "request_type",
    "justification", "confidence_score", "source_documents",
    "risk_level", "pii_detected", "language", "actions_taken",
]


def read_tickets() -> list[dict[str, str]]:
    """Read support_tickets.csv preserving insertion order."""
    rows: list[dict[str, str]] = []
    with open(INPUT_CSV, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def write_output(rows: list[dict[str, Any]]) -> None:
    """Write output.csv atomically with all required columns."""
    dir_ = os.path.dirname(OUTPUT_CSV)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp", prefix="output_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=OUTPUT_COLUMNS,
                quoting=csv.QUOTE_ALL,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                # Normalise actions_taken to JSON string
                at = row.get("actions_taken", [])
                if isinstance(at, list):
                    row = dict(row)
                    row["actions_taken"] = json.dumps(at, ensure_ascii=False)
                writer.writerow(row)
        os.replace(tmp_path, OUTPUT_CSV)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def parse_issue(raw: str) -> list[dict[str, str]]:
    """Parse the JSON-encoded issue field; returns [] on failure."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        return []
    except (json.JSONDecodeError, ValueError):
        return []
