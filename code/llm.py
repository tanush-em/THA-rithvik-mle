"""Anthropic Claude Haiku client with deterministic caching.

Two entry points:
  - call_responder(...)  → grounded response via tool-use
  - call_judge(...)      → safety judge verdict via tool-use

Both use:
  - temperature=0 for determinism
  - SQLite cache keyed by SHA256(model + prompt_version + inputs)
  - Global rate limiter (~40 req/min) + retry with backoff on 429
  - asyncio.Semaphore(10) for in-flight request cap
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass

import anthropic

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache", "anthropic.sqlite")

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = 1024
MAX_RETRIES = 6
BASE_BACKOFF_SEC = 2.0
RATE_LIMIT_CALLS = int(os.environ.get("ANTHROPIC_RATE_LIMIT_PER_MIN", "40"))
RATE_LIMIT_PERIOD = 60.0

# Global semaphore for in-flight API calls
_semaphore = asyncio.Semaphore(10)


class _RateLimiter:
    """Token-bucket style limiter: max N API calls per period."""

    def __init__(self, max_calls: int, period: float) -> None:
        self._max_calls = max_calls
        self._period = period
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < self._period]
            if len(self._timestamps) >= self._max_calls:
                wait = self._period - (now - self._timestamps[0]) + 0.1
                if wait > 0:
                    await asyncio.sleep(wait)
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if now - t < self._period]
            self._timestamps.append(time.monotonic())


_rate_limiter = _RateLimiter(RATE_LIMIT_CALLS, RATE_LIMIT_PERIOD)


def set_max_concurrency(n: int) -> None:
    """Adjust in-flight cap (called from main.py --concurrency)."""
    global _semaphore
    _semaphore = asyncio.Semaphore(max(1, n))

# Thread-local Anthropic client (avoid sharing across event loops)
_client_lock = threading.Lock()
_async_client: anthropic.AsyncAnthropic | None = None


def _get_async_client() -> anthropic.AsyncAnthropic:
    global _async_client
    if _async_client is None:
        with _client_lock:
            if _async_client is None:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError(
                        "ANTHROPIC_API_KEY not set. Copy .env.example → .env and add your key."
                    )
                _async_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _async_client


# ---------------------------------------------------------------------------
# Prompt template loading + SHA hashing
# ---------------------------------------------------------------------------

def _load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _prompt_sha(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

_db_lock = threading.Lock()


def _cache_get(key: str) -> dict | None:
    with _db_lock:
        try:
            con = sqlite3.connect(CACHE_PATH)
            row = con.execute(
                "SELECT value FROM llm_cache WHERE key=?", (key,)
            ).fetchone()
            con.close()
            if row:
                return json.loads(row[0])
        except Exception:
            pass
    return None


def _cache_set(key: str, value: dict) -> None:
    with _db_lock:
        try:
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            con = sqlite3.connect(CACHE_PATH)
            con.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT)"
            )
            con.execute(
                "INSERT OR REPLACE INTO llm_cache (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
            con.commit()
            con.close()
        except Exception:
            pass


def _make_cache_key(*parts: str) -> str:
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()


def _parse_confidence(value: object) -> float:
    """Parse self_confidence; tolerate malformed tool output (e.g. XML tags)."""
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    text = str(value).strip()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return 0.5


def _is_rate_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, anthropic.RateLimitError):
        return True
    msg = str(exc).lower()
    return "rate_limit" in msg or "429" in msg


async def _messages_create(**kwargs: object) -> anthropic.types.Message:
    """Call Anthropic API with rate limiting and exponential backoff on 429."""
    client = _get_async_client()
    last_err: BaseException | None = None
    for attempt in range(MAX_RETRIES):
        await _rate_limiter.acquire()
        async with _semaphore:
            try:
                return await client.messages.create(**kwargs)  # type: ignore[arg-type]
            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    raise
                last_err = exc
        wait = BASE_BACKOFF_SEC * (2 ** attempt)
        await asyncio.sleep(wait)
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_RESPONDER_TOOLS = [
    {
        "name": "respond_to_ticket",
        "description": "Provide a grounded response to the support ticket.",
        "input_schema": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "User-facing answer grounded in corpus snippets. No PII echo.",
                },
                "sources_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths of corpus documents actually used.",
                },
                "self_confidence": {
                    "type": "number",
                    "description": "Calibrated confidence 0.0–1.0.",
                },
            },
            "required": ["response", "sources_used", "self_confidence"],
        },
    }
]

_JUDGE_TOOLS = [
    {
        "name": "judge_response",
        "description": "Evaluate the draft response for safety issues.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["keep", "sanitize", "escalate"],
                    "description": "Safety verdict.",
                },
                "reasons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of issues found.",
                },
                "fix_hint": {
                    "type": "string",
                    "description": "What to fix (sanitize verdict only).",
                },
            },
            "required": ["verdict", "reasons"],
        },
    }
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ResponderResult:
    response: str
    sources_used: list[str]
    self_confidence: float
    from_cache: bool = False


@dataclass
class JudgeResult:
    verdict: str   # "keep" | "sanitize" | "escalate"
    reasons: list[str]
    fix_hint: str = ""
    from_cache: bool = False


# ---------------------------------------------------------------------------
# Core async callers
# ---------------------------------------------------------------------------

async def call_responder(
    user_text: str,
    snippets: list[dict],
) -> ResponderResult:
    """Call Haiku to generate a grounded response from corpus snippets.

    Args:
        user_text: Sanitized user ticket text (PII redacted).
        snippets: List of {rel_path, text, heading} dicts from retriever.
    """
    system_prompt = _load_prompt("responder_system.md")
    prompt_version = _prompt_sha(system_prompt)

    # Build corpus context
    corpus_block = "\n\n---\n\n".join(
        f"[Source: {s['rel_path']}]\n{s.get('heading', '')}\n{s['text']}"
        for s in snippets
    )

    user_message = (
        f"<CORPUS_SNIPPETS>\n{corpus_block}\n</CORPUS_SNIPPETS>\n\n"
        f"<USER_TICKET>\n{user_text}\n</USER_TICKET>\n\n"
        "Please call `respond_to_ticket` to answer the user's ticket using only the snippets above."
    )

    snippets_sha = hashlib.sha256(corpus_block.encode()).hexdigest()[:16]
    cache_key = _make_cache_key("responder", MODEL, prompt_version, user_text[:500], snippets_sha)

    cached = _cache_get(cache_key)
    if cached:
        return ResponderResult(
            response=cached["response"],
            sources_used=cached["sources_used"],
            self_confidence=cached["self_confidence"],
            from_cache=True,
        )

    msg = await _messages_create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        system=system_prompt,
        tools=_RESPONDER_TOOLS,
        tool_choice={"type": "tool", "name": "respond_to_ticket"},
        messages=[{"role": "user", "content": user_message}],
    )

    # Extract tool use result
    tool_input: dict = {}
    for block in msg.content:
        if block.type == "tool_use" and block.name == "respond_to_ticket":
            tool_input = block.input
            break

    result = ResponderResult(
        response=tool_input.get("response", "I was unable to find relevant information."),
        sources_used=tool_input.get("sources_used", []),
        self_confidence=_parse_confidence(tool_input.get("self_confidence", 0.5)),
    )

    _cache_set(cache_key, {
        "response": result.response,
        "sources_used": result.sources_used,
        "self_confidence": result.self_confidence,
    })

    return result


async def call_judge(
    user_text: str,
    draft_response: str,
    cited_paths: list[str],
) -> JudgeResult:
    """Call Haiku safety judge to evaluate a draft response.

    Args:
        user_text: Sanitized user ticket text.
        draft_response: The draft response to evaluate.
        cited_paths: Source paths cited in the response.
    """
    system_prompt = _load_prompt("judge_system.md")
    prompt_version = _prompt_sha(system_prompt)

    paths_str = "\n".join(cited_paths) if cited_paths else "(none)"
    user_message = (
        f"<USER_TICKET>\n{user_text}\n</USER_TICKET>\n\n"
        f"<DRAFT_RESPONSE>\n{draft_response}\n</DRAFT_RESPONSE>\n\n"
        f"<CITED_SOURCES>\n{paths_str}\n</CITED_SOURCES>\n\n"
        "Please call `judge_response` to evaluate this draft."
    )

    cache_key = _make_cache_key(
        "judge", MODEL, prompt_version, user_text[:300], draft_response[:300]
    )

    cached = _cache_get(cache_key)
    if cached:
        return JudgeResult(
            verdict=cached["verdict"],
            reasons=cached["reasons"],
            fix_hint=cached.get("fix_hint", ""),
            from_cache=True,
        )

    msg = await _messages_create(
        model=MODEL,
        max_tokens=512,
        temperature=0,
        system=system_prompt,
        tools=_JUDGE_TOOLS,
        tool_choice={"type": "tool", "name": "judge_response"},
        messages=[{"role": "user", "content": user_message}],
    )

    tool_input: dict = {}
    for block in msg.content:
        if block.type == "tool_use" and block.name == "judge_response":
            tool_input = block.input
            break

    result = JudgeResult(
        verdict=tool_input.get("verdict", "keep"),
        reasons=tool_input.get("reasons", []),
        fix_hint=tool_input.get("fix_hint", ""),
    )

    _cache_set(cache_key, {
        "verdict": result.verdict,
        "reasons": result.reasons,
        "fix_hint": result.fix_hint,
    })

    return result


# ---------------------------------------------------------------------------
# Sync wrappers for non-async contexts
# ---------------------------------------------------------------------------

def call_responder_sync(user_text: str, snippets: list[dict]) -> ResponderResult:
    return asyncio.get_event_loop().run_until_complete(
        call_responder(user_text, snippets)
    )


def call_judge_sync(
    user_text: str, draft_response: str, cited_paths: list[str]
) -> JudgeResult:
    return asyncio.get_event_loop().run_until_complete(
        call_judge(user_text, draft_response, cited_paths)
    )
