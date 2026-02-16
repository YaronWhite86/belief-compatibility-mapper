"""End-to-end pipeline tests simulating different users."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_profile


class TestFullPipeline:
    """Run the complete pipeline for each user profile."""

    @pytest.mark.parametrize("profile", ["philosopher", "economist", "mixed_worldview"])
    def test_profile_pipeline(self, belief_map_factory, profile):
        bm = belief_map_factory(profile)
        beliefs = bm.list_beliefs()
        profile_data = load_profile(profile)
        assert len(beliefs) == len(profile_data)

        # Embed
        count = bm.generate_embeddings()
        assert count == len(beliefs)
        for b in bm.list_beliefs():
            assert len(b.embedding) > 0

        # Similarity
        bm.calculate_initial_similarity()
        for b in beliefs:
            assert bm.similarity[b.id, b.id] == pytest.approx(1.0)

        # Analyze interesting pairs
        results = bm.analyze_interesting(top_n=50, threshold=0.0)
        for id_a, id_b, result in results:
            assert -1.0 <= result.score <= 1.0
            assert not np.isnan(bm.scores[id_a, id_b])

    @pytest.mark.parametrize("profile", ["philosopher", "economist", "mixed_worldview"])
    def test_visualization_outputs(self, belief_map_factory, tmp_path, profile):
        from visualization import export_heatmap, export_network

        bm = belief_map_factory(profile)
        bm.generate_embeddings()
        bm.calculate_initial_similarity()
        bm.analyze_interesting(top_n=50, threshold=0.0)

        hm = export_heatmap(bm, output=tmp_path / "heatmap.html")
        net = export_network(bm, output=tmp_path / "network.html")
        assert hm.exists() and hm.stat().st_size > 0
        assert net.exists() and net.stat().st_size > 0


class TestIncrementalUpdate:
    """Verify that adding one belief to an existing map is efficient."""

    def test_incremental_embed_uses_cache(self, belief_map_factory, mock_openai):
        bm = belief_map_factory(beliefs=["A", "B", "C"])
        bm.generate_embeddings()
        first_call_count = mock_openai.embeddings.create.call_count

        # Add one more belief and re-embed
        bm.add_belief("D")
        bm.generate_embeddings()
        second_call_count = mock_openai.embeddings.create.call_count

        # Only one new API call for "D" (A/B/C still have in-memory embeddings)
        assert second_call_count - first_call_count == 1

    def test_incremental_embed_from_cold_cache(self, tmp_path, mock_openai, mock_anthropic):
        """Simulate a restart: embeddings come from SQLite, not memory."""
        from cache import RateLimiter, ResultCache
        from engine import BeliefMap

        # --- Run 1: embed 3 beliefs (writes to cache) ----------------
        bm1 = BeliefMap()
        bm1.cache = ResultCache(tmp_path / "cache.db")
        bm1.rate_limiter = RateLimiter(max_rpm=10_000)
        bm1.add_belief("A")
        bm1.add_belief("B")
        bm1.add_belief("C")
        bm1.generate_embeddings()
        calls_run1 = mock_openai.embeddings.create.call_count

        # --- Run 2: new BeliefMap, same cache, add one belief ---------
        bm2 = BeliefMap()
        bm2.cache = ResultCache(tmp_path / "cache.db")
        bm2.rate_limiter = RateLimiter(max_rpm=10_000)
        bm2.add_belief("A")  # no in-memory embedding
        bm2.add_belief("B")
        bm2.add_belief("C")
        bm2.add_belief("D")  # new
        bm2.generate_embeddings()
        calls_run2 = mock_openai.embeddings.create.call_count

        # Run 2 should only call the API once (for "D"); A/B/C from cache
        new_api_calls = calls_run2 - calls_run1
        assert new_api_calls == 1

        # All 4 should have embeddings now
        for b in bm2.list_beliefs():
            assert len(b.embedding) > 0

        bm1.cache.close()
        bm2.cache.close()

    def test_incremental_analysis_skips_scored(
        self, belief_map_factory, mock_anthropic
    ):
        bm = belief_map_factory()
        bm.add_belief("A", tags=["shared"])
        bm.add_belief("B", tags=["shared"])
        bm.add_belief("C", tags=["shared"])

        # First run: analyze all interesting pairs
        bm.analyze_interesting(top_n=50, threshold=0.0)
        calls_first = mock_anthropic.messages.create.call_count

        # Add a fourth belief
        bm.add_belief("D", tags=["shared"])

        # Second run: only new pairs involving "D" should be analyzed
        bm.analyze_interesting(top_n=50, threshold=0.0)
        calls_second = mock_anthropic.messages.create.call_count

        new_calls = calls_second - calls_first
        # D pairs with A, B, C = 3 new pairs (existing pairs are skipped)
        assert new_calls == 3


class TestEdgeCases:
    def test_duplicate_beliefs_rejected(self, tmp_path, belief_map_factory):
        bm = belief_map_factory()
        bm.add_belief("Same belief")
        with pytest.raises(ValueError, match="Duplicate belief"):
            bm.add_belief("Same belief")
        assert len(bm.list_beliefs()) == 1

    def test_empty_map_no_crash(self, belief_map_factory):
        bm = belief_map_factory()
        bm.calculate_initial_similarity()
        assert bm.interesting_pairs() == []
        assert bm.scored_pairs() == []

    def test_single_belief_no_crash(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["Alone"])
        bm.generate_embeddings()
        bm.calculate_initial_similarity()
        assert bm.interesting_pairs() == []
