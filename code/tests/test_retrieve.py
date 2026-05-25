"""Unit tests for code/retrieve.py"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from retrieve import retrieve


def test_devplatform_password_reset():
    results = retrieve("forgot password reset devplatform", ["devplatform"], top_k=5)
    assert len(results) > 0
    # Should retrieve a devplatform doc
    assert any("devplatform" in r.rel_path for r in results)


def test_claude_delete_conversation():
    results = retrieve("delete conversation privacy Claude", ["claude"], top_k=5)
    assert len(results) > 0
    assert any("claude" in r.rel_path for r in results)


def test_visa_lost_stolen_card():
    results = retrieve("lost stolen visa card report", ["visa"], top_k=5)
    assert len(results) > 0
    assert any("visa" in r.rel_path for r in results)


def test_domain_filter_respected():
    """Results should only come from the specified domain."""
    results = retrieve("test assessment devplatform candidates", ["devplatform"], top_k=5)
    for r in results:
        assert r.rel_path.startswith("data/devplatform/"), (
            f"Got non-devplatform result: {r.rel_path}"
        )


def test_empty_query_returns_empty():
    results = retrieve("", ["devplatform"], top_k=5)
    assert results == []


def test_no_results_for_irrelevant_query():
    results = retrieve("iron man actor superhero movie", ["devplatform"], top_k=5)
    # May or may not return results, but should not crash
    assert isinstance(results, list)


def test_results_sorted_by_score():
    results = retrieve("time accommodation extra time candidate devplatform", ["devplatform"], top_k=5)
    if len(results) >= 2:
        for i in range(len(results) - 1):
            assert results[i].final_score >= results[i + 1].final_score


def test_visa_chargeback():
    results = retrieve("dispute charge chargeback visa", ["visa"], top_k=5)
    assert len(results) > 0


def test_claude_subscription_billing():
    results = retrieve("cancel subscription Claude Pro billing", ["claude"], top_k=5)
    assert len(results) > 0
    assert any("claude" in r.rel_path for r in results)


def test_deduplication_one_result_per_file():
    """Ensure no two results come from the same file."""
    results = retrieve("devplatform test settings expiration", ["devplatform"], top_k=5)
    paths = [r.rel_path for r in results]
    assert len(paths) == len(set(paths)), "Duplicate file paths in results"
