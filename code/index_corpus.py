#!/usr/bin/env python3
"""Build BM25 corpus index from data/ markdown files.

Run once before using the agent:
    python code/index_corpus.py

Produces code/cache/corpus_index.pkl containing:
  - BM25Okapi index
  - list of chunk metadata dicts: {path, rel_path, heading, text, weight}
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
from typing import Any

from rank_bm25 import BM25Okapi

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OVERRIDES_PATH = os.path.join(os.path.dirname(__file__), "corpus_overrides.json")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "cache", "corpus_index.pkl")

CHUNK_TARGET_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100

# Rough heuristic: 1 token ≈ 4 characters
_CHARS_PER_TOKEN = 4
CHUNK_TARGET_CHARS = CHUNK_TARGET_TOKENS * _CHARS_PER_TOKEN
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Load overrides
# ---------------------------------------------------------------------------

def _load_overrides() -> dict[str, float]:
    """Return a mapping of relative_path → weight float."""
    weights: dict[str, float] = {}
    if not os.path.exists(OVERRIDES_PATH):
        return weights
    with open(OVERRIDES_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    for rel_path, info in data.get("overrides", {}).items():
        weights[rel_path] = float(info.get("weight", 1.0))
    return weights


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_by_headings(text: str, file_path: str) -> list[dict[str, str]]:
    """Split markdown text into heading-anchored chunks."""
    # Split on any # heading
    heading_re = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
    parts: list[tuple[str, str]] = []  # (heading, body)

    splits = heading_re.split(text)
    # splits alternates: [pre-content, heading, body, heading, body, ...]
    # First element is pre-heading content
    if splits[0].strip():
        parts.append(("", splits[0]))

    i = 1
    while i < len(splits) - 1:
        heading = splits[i].strip()
        body = splits[i + 1] if i + 1 < len(splits) else ""
        parts.append((heading, body))
        i += 2

    if not parts:
        parts = [("", text)]

    chunks = []
    for heading, body in parts:
        full_text = (f"{heading}\n\n{body}" if heading else body).strip()
        if not full_text:
            continue
        # If chunk is small enough, keep as-is
        if len(full_text) <= CHUNK_TARGET_CHARS:
            chunks.append({"heading": heading, "text": full_text, "path": file_path})
        else:
            # Split large chunks with overlap
            start = 0
            while start < len(full_text):
                end = start + CHUNK_TARGET_CHARS
                piece = full_text[start:end]
                chunks.append({"heading": heading, "text": piece, "path": file_path})
                if end >= len(full_text):
                    break
                start = end - CHUNK_OVERLAP_CHARS
    return chunks


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_NON_ALPHA = re.compile(r"[^a-z0-9\s]")


def _tokenize(text: str) -> list[str]:
    """Simple lowercase alphanumeric tokenizer."""
    text = text.lower()
    # Remove markdown formatting
    text = re.sub(r"#+\s+", " ", text)          # headings
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # links
    text = re.sub(r"`[^`]+`", " ", text)         # code spans
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # code blocks
    text = _NON_ALPHA.sub(" ", text)
    return [t for t in text.split() if len(t) > 1]


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_index() -> None:
    weights = _load_overrides()
    chunks: list[dict[str, Any]] = []

    # Walk data/ in deterministic sorted order
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        dirnames.sort()
        for filename in sorted(filenames):
            if not filename.endswith(".md"):
                continue
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, REPO_ROOT).replace(os.sep, "/")
            weight = weights.get(rel_path, 1.0)

            with open(abs_path, encoding="utf-8", errors="replace") as fh:
                content = fh.read()

            file_chunks = _split_by_headings(content, abs_path)
            for chunk in file_chunks:
                chunk["rel_path"] = rel_path
                chunk["weight"] = weight
                chunk["tokens"] = _tokenize(chunk["text"])
                chunks.append(chunk)

    if not chunks:
        print("ERROR: No chunks found!", file=sys.stderr)
        sys.exit(1)

    print(f"Indexed {len(chunks)} chunks from {DATA_DIR}")

    # Build BM25 index
    corpus_tokens = [c["tokens"] for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    # Strip tokens from stored chunks to save space (we only need text + metadata)
    for chunk in chunks:
        del chunk["tokens"]

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "wb") as fh:
        pickle.dump({"bm25": bm25, "chunks": chunks}, fh, protocol=5)

    print(f"Index written to {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
