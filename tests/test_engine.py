"""Unit tests for BeliefMap core operations (no API calls)."""

from __future__ import annotations

import numpy as np
import pytest

from engine import MAX_BELIEFS, BeliefMap


class TestBeliefCRUD:
    def test_add_and_get(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["Alpha", "Beta"])
        assert len(bm.list_beliefs()) == 2
        assert bm.get_belief(0).text == "Alpha"
        assert bm.get_belief(1).text == "Beta"

    def test_add_returns_sequential_ids(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["A", "B", "C"])
        ids = [b.id for b in bm.list_beliefs()]
        assert ids == [0, 1, 2]

    def test_remove_belief(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["A", "B", "C"])
        bm.remove_belief(1)
        assert len(bm.list_beliefs()) == 2
        with pytest.raises(KeyError):
            bm.get_belief(1)

    def test_remove_clears_scores(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["A", "B"])
        bm.set_score(0, 1, 0.5)
        bm.remove_belief(1)
        assert np.isnan(bm.scores[0, 1])

    def test_id_reuse_after_removal(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["A", "B", "C"])
        bm.remove_belief(0)
        new = bm.add_belief("D")
        assert new.id == 0  # lowest available reused

    def test_overflow_at_max(self, belief_map_factory):
        bm = belief_map_factory()
        for i in range(MAX_BELIEFS):
            bm.add_belief(f"Belief {i}")
        with pytest.raises(ValueError, match="Cannot exceed"):
            bm.add_belief("one too many")


class TestScores:
    def test_set_and_get_score(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["A", "B"])
        bm.set_score(0, 1, -0.75)
        assert bm.get_score(0, 1) == -0.75
        assert bm.get_score(1, 0) == -0.75  # symmetric

    def test_score_range_validation(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["A", "B"])
        with pytest.raises(ValueError):
            bm.set_score(0, 1, 1.5)
        with pytest.raises(ValueError):
            bm.set_score(0, 1, -1.1)

    def test_scored_pairs(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["A", "B", "C"])
        bm.set_score(0, 1, 0.5)
        bm.set_score(1, 2, -0.3)
        pairs = bm.scored_pairs()
        assert len(pairs) == 2
        assert (0, 1, 0.5) in pairs


class TestSimilarity:
    def test_cosine_similarity_shape(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["A", "B", "C"])
        bm.generate_embeddings()
        bm.calculate_initial_similarity()
        # Diagonal should be 1.0 for embedded beliefs
        for b in bm.list_beliefs():
            assert bm.similarity[b.id, b.id] == pytest.approx(1.0)

    def test_similarity_is_symmetric(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["A", "B"])
        bm.generate_embeddings()
        bm.calculate_initial_similarity()
        assert bm.similarity[0, 1] == pytest.approx(bm.similarity[1, 0])

    def test_similarity_non_nan_for_embedded(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["X", "Y"])
        bm.generate_embeddings()
        bm.calculate_initial_similarity()
        assert not np.isnan(bm.similarity[0, 1])


class TestInterestingPairs:
    def test_threshold_filter(self, belief_map_factory):
        bm = belief_map_factory(beliefs=["A", "B", "C"])
        # Manually set similarity to control the test
        bm.similarity[0, 1] = bm.similarity[1, 0] = 0.9
        bm.similarity[0, 2] = bm.similarity[2, 0] = 0.3
        bm.similarity[1, 2] = bm.similarity[2, 1] = 0.5
        pairs = bm.interesting_pairs(threshold=0.7)
        pair_ids = {(a, b) for a, b, _ in pairs}
        assert (0, 1) in pair_ids
        assert (0, 2) not in pair_ids

    def test_tag_based_pairing(self, belief_map_factory):
        bm = belief_map_factory()
        bm.add_belief("A", tags=["ethics"])
        bm.add_belief("B", tags=["ethics"])
        # Similarity is NaN (no embeddings), but shared tag qualifies
        pairs = bm.interesting_pairs(threshold=0.99)
        assert len(pairs) == 1
        assert pairs[0][0] == 0 and pairs[0][1] == 1

    def test_top_n_cap(self, belief_map_factory):
        bm = belief_map_factory()
        for i in range(10):
            bm.add_belief(f"B{i}", tags=["shared"])
        pairs = bm.interesting_pairs(top_n=3, threshold=0.0)
        assert len(pairs) <= 3
