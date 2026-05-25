# Support Triage Agent — Setup & Run Guide

## Overview

A deterministic, rule-orchestrated support triage agent that resolves support tickets across DevPlatform, Claude, and Visa using:
- Pure-Python adversarial safety screening (no LLM for this)
- BM25 retrieval over the provided corpus
- Anthropic Claude Haiku for response generation and safety judging (2 calls per ticket max)
- SQLite response cache for deterministic re-runs

---

## Prerequisites

- Python 3.11+
- An Anthropic API key with access to `claude-haiku-4-5-20251001` (or set `ANTHROPIC_MODEL`)

---

## Setup (exact reproduction steps)

```bash
# 1. Clone the repo (if needed)
cd /path/to/tha-rithvik-mle

# 2. Create virtual environment (recommended)
python3.11 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install dependencies (pinned)
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and set: ANTHROPIC_API_KEY=sk-ant-...

# 5. Build corpus index (run once; ~1 second)
python code/index_corpus.py

# 6. Run the agent
python code/main.py
```

Output is written to `support_tickets/output.csv`.

---

## Run Options

```bash
# Process all tickets (default)
python code/main.py

# Dry run (skip LLM, use canned responses — for testing structure)
python code/main.py --dry-run

# Process single ticket by 0-based index (for debugging)
python code/main.py --ticket-index 5

# Override concurrency (default 10; global rate limit ~40 req/min with 429 retry)
python code/main.py --concurrency 8

# Override output path
python code/main.py --output /tmp/test_output.csv
```

---

## Validate Output Format

```bash
python code/validate_output.py
```

This checks structure only (correct columns, valid enums, row count). Run before submitting.

---

## Run Tests

```bash
python -m pytest code/tests/ -v
```

Currently: **91 tests**, all passing.
Test categories:
- `test_pii.py` — PII regex + Luhn (12 tests)
- `test_safety.py` — Adversarial screen (14 tests)
- `test_lang.py` — Language detection (8 tests)
- `test_retrieve.py` — BM25 retrieval (10 tests)
- `test_classify_policy.py` — Classifier + policy (18 tests)
- `test_adversarial.py` — End-to-end adversarial (29 tests)

---

## Determinism Verification

The agent uses `temperature=0` + SQLite cache keyed by SHA256 of all inputs.
After the first run, a second run should produce byte-identical output:

```bash
python code/main.py
cp support_tickets/output.csv /tmp/output_first.csv
python code/main.py
diff /tmp/output_first.csv support_tickets/output.csv
# Should show no differences
```

---

## Performance

- Dry run: ~0.04s/ticket (no LLM)
- With LLM (Haiku, concurrency=20): ~0.3–0.5s/ticket for first run
- Second run (cache hits): ~0.04s/ticket
- 89 visible tickets: ~30–45s first run, ~4s cached
- 150 hidden tickets: estimated ~60–75s first run, well within 3-minute limit

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes |
| `ANTHROPIC_MODEL` | Model override (default: `claude-haiku-4-5-20251001`) | No |

---

## Module Reference

| File | Purpose |
|---|---|
| `main.py` | CLI entry point |
| `agent.py` | Per-ticket orchestrator (11-stage pipeline) |
| `pii.py` | PII detection + redaction (regex + Luhn) |
| `safety.py` | Adversarial injection screening |
| `lang.py` | Language detection (lingua) |
| `classify.py` | Domain + request-type classification |
| `policy.py` | Risk assessment + escalation rules |
| `retrieve.py` | BM25 retrieval |
| `index_corpus.py` | Build BM25 corpus index |
| `llm.py` | Anthropic client + SQLite cache |
| `tools.py` | Rule-based tool-call selection |
| `calibrate.py` | Confidence calibration |
| `io_csv.py` | CSV read/write |
| `corpus_overrides.json` | Per-file BM25 weight overrides |
| `validate_output.py` | Format validation (from repo starter) |
