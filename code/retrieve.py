"""BM25 retrieval over the pre-built corpus index.

Loads the index once at import time (lazy, thread-safe via module-level
singleton). All query operations are deterministic and in-memory.
"""
from __future__ import annotations

import os
import pickle
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(os.path.dirname(__file__), "cache", "corpus_index.pkl")

# Domain → path prefix(es) filter
DOMAIN_PREFIXES: dict[str, list[str]] = {
    "devplatform": ["data/devplatform/"],
    "claude": ["data/claude/"],
    "visa": ["data/visa/"],
}


# ---------------------------------------------------------------------------
# Singleton index loader
# ---------------------------------------------------------------------------

_index_lock = threading.Lock()
_index_cache: dict[str, Any] = {}


def _load_index() -> dict[str, Any]:
    global _index_cache
    if _index_cache:
        return _index_cache
    with _index_lock:
        if _index_cache:
            return _index_cache
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"Corpus index not found at {INDEX_PATH}. "
                "Run: python code/index_corpus.py"
            )
        with open(INDEX_PATH, "rb") as fh:
            data = pickle.load(fh)
        _index_cache = data
        return data


# ---------------------------------------------------------------------------
# Tokenizer (must match index_corpus.py)
# ---------------------------------------------------------------------------

_NON_ALPHA = re.compile(r"[^a-z0-9\s]")


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"#+\s+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = _NON_ALPHA.sub(" ", text)
    return [t for t in text.split() if len(t) > 1]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    rel_path: str
    text: str
    heading: str
    bm25_score: float
    weight: float
    final_score: float


# ---------------------------------------------------------------------------
# Path-depth canonicality prior
# ---------------------------------------------------------------------------

def _path_depth_bonus(rel_path: str) -> float:
    """Deeper paths (more specific docs) get a small bonus. Max 0.15."""
    depth = rel_path.count("/")
    return min(depth * 0.03, 0.15)


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    domains: list[str],
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Retrieve top-k chunks most relevant to *query* within the given *domains*.

    Args:
        query: The sanitized user query text.
        domains: List of domain names ('devplatform', 'claude', 'visa') to
                 restrict retrieval. Pass empty list for no filter.
        top_k: Number of results to return.

    Returns:
        List of RetrievalResult sorted by final_score descending.
    """
    index_data = _load_index()
    bm25 = index_data["bm25"]
    chunks: list[dict[str, Any]] = index_data["chunks"]

    if not query.strip():
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # BM25 scores for all chunks
    scores = bm25.get_scores(query_tokens)

    # Build domain prefix filter
    allowed_prefixes: list[str] = []
    for d in domains:
        allowed_prefixes.extend(DOMAIN_PREFIXES.get(d, []))

    results: list[RetrievalResult] = []
    for i, (chunk, raw_score) in enumerate(zip(chunks, scores)):
        rel_path: str = chunk["rel_path"]

        # Domain filter
        if allowed_prefixes and not any(rel_path.startswith(p) for p in allowed_prefixes):
            continue

        if raw_score <= 0:
            continue

        weight: float = chunk.get("weight", 1.0)
        depth_bonus = _path_depth_bonus(rel_path)
        final_score = raw_score * weight + depth_bonus

        results.append(RetrievalResult(
            rel_path=rel_path,
            text=chunk["text"],
            heading=chunk.get("heading", ""),
            bm25_score=float(raw_score),
            weight=weight,
            final_score=final_score,
        ))

    # Sort by final_score descending, then rel_path for determinism
    results.sort(key=lambda r: (-r.final_score, r.rel_path))

    # Deduplicate by rel_path (keep highest-scoring chunk per file)
    seen: set[str] = set()
    deduped: list[RetrievalResult] = []
    for r in results:
        if r.rel_path not in seen:
            seen.add(r.rel_path)
            deduped.append(r)
        if len(deduped) >= top_k:
            break

    return deduped


def retrieve_multi_domain(
    query: str,
    domains: list[str],
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Retrieve across multiple domains, merging and re-ranking results."""
    if not domains:
        return retrieve(query, [], top_k)
    # For multi-domain, retrieve with no prefix filter but apply domain weighting
    return retrieve(query, domains, top_k)
