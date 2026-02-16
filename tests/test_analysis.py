"""Tests for LLM tension analysis with mocked API responses."""

from __future__ import annotations

import numpy as np
import pytest


class TestAnalyzeLogicalTension:
    def test_returns_valid_tension_result(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["Free will exists", "Determinism is true"])
        result = bm.analyze_logical_tension(
            bm.get_belief(0), bm.get_belief(1)
        )
        assert -1.0 <= result.score <= 1.0
        assert result.category.value in {
            "mutually_entailed",
            "compatible_harmonious",
            "neutral",
            "tensioned",
            "contradictory",
        }
        assert len(result.justification) > 0

    def test_uses_cache_on_second_call(self, belief_map_factory, mock_anthropic):
        bm = belief_map_factory(beliefs=["A", "B"])
        bm.analyze_logical_tension(bm.get_belief(0), bm.get_belief(1))
        bm.analyze_logical_tension(bm.get_belief(0), bm.get_belief(1))
        # Only one actual API call; second was served from cache
        assert mock_anthropic.messages.create.call_count == 1

    def test_cache_miss_different_texts(self, belief_map_factory, mock_anthropic):
        bm = belief_map_factory(beliefs=["A", "B", "C"])
        bm.analyze_logical_tension(bm.get_belief(0), bm.get_belief(1))
        bm.analyze_logical_tension(bm.get_belief(0), bm.get_belief(2))
        assert mock_anthropic.messages.create.call_count == 2


class TestAnalyzePair:
    def test_writes_score_to_matrix(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["A", "B"])
        result = bm.analyze_pair(0, 1)
        assert bm.get_score(0, 1) == result.score
        assert bm.get_score(1, 0) == result.score  # symmetric

    def test_unknown_id_raises(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["A"])
        with pytest.raises(KeyError):
            bm.analyze_pair(0, 99)


class TestAnalyzeInteresting:
    def test_skips_already_scored(self, belief_map_factory, mock_anthropic):
        bm = belief_map_factory()
        # Give all pairs shared tags so they're all "interesting"
        bm.add_belief("A", tags=["shared"])
        bm.add_belief("B", tags=["shared"])
        bm.add_belief("C", tags=["shared"])
        # Pre-score one pair
        bm.set_score(0, 1, 0.5)
        results = bm.analyze_interesting(top_n=50, threshold=0.0)
        analyzed_pairs = {(a, b) for a, b, _ in results}
        assert (0, 1) not in analyzed_pairs  # was skipped
        assert len(analyzed_pairs) > 0  # others were analyzed

    def test_force_reanalyzes_scored(self, belief_map_factory, mock_anthropic):
        bm = belief_map_factory()
        bm.add_belief("A", tags=["shared"])
        bm.add_belief("B", tags=["shared"])
        bm.set_score(0, 1, 0.5)
        results = bm.analyze_interesting(top_n=50, threshold=0.0, force=True)
        assert len(results) == 1
        assert results[0][0] == 0 and results[0][1] == 1

    def test_empty_when_no_interesting_pairs(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["A", "B"])
        # No tags, no similarity → nothing interesting
        results = bm.analyze_interesting(threshold=0.99)
        assert results == []
