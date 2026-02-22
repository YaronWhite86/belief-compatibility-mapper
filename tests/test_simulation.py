"""Tests for BeliefMap.simulate_removal()."""

from __future__ import annotations

import pytest

from engine import BeliefMap


def _star_map() -> tuple[BeliefMap, int, list[int]]:
    """Build a star graph: one hub connected to 4 leaves.

    Hub id=0, leaves id=1..4 — all with score 0.8 to hub.
    Leaves are not connected to each other.
    """
    bm = BeliefMap()
    hub = bm.add_belief("Central hub belief")
    leaves = [bm.add_belief(f"Leaf belief {i}") for i in range(4)]
    for leaf in leaves:
        bm.set_score(hub.id, leaf.id, 0.8)
    return bm, hub.id, [l.id for l in leaves]


def _chain_map() -> tuple[BeliefMap, list[int]]:
    """Build a chain: A — B — C with scores 0.8."""
    bm = BeliefMap()
    a = bm.add_belief("Belief A")
    b = bm.add_belief("Belief B")
    c = bm.add_belief("Belief C")
    bm.set_score(a.id, b.id, 0.8)
    bm.set_score(b.id, c.id, 0.8)
    return bm, [a.id, b.id, c.id]


def _triangle_map() -> tuple[BeliefMap, list[int]]:
    """Build a triangle: A — B — C — A."""
    bm = BeliefMap()
    a = bm.add_belief("Belief P")
    b = bm.add_belief("Belief Q")
    c = bm.add_belief("Belief R")
    bm.set_score(a.id, b.id, 0.8)
    bm.set_score(b.id, c.id, 0.8)
    bm.set_score(a.id, c.id, 0.8)
    return bm, [a.id, b.id, c.id]


def test_hub_removal_orphans_leaves():
    """Removing the hub of a star graph orphans all leaf beliefs."""
    bm, hub_id, leaf_ids = _star_map()
    result = bm.simulate_removal(hub_id)

    assert result.removed_id == hub_id
    assert set(result.orphaned_ids) == set(leaf_ids)
    assert result.stable_ids == []
    assert result.destabilized_ids == []


def test_chain_removal_destabilizes():
    """Removing the middle node of A-B-C splits the chain — endpoints are destabilized."""
    bm, (a_id, b_id, c_id) = _chain_map()
    result = bm.simulate_removal(b_id)

    # A and C lose their only connection to B and each other → destabilized or orphaned
    affected = set(result.destabilized_ids) | set(result.orphaned_ids)
    assert a_id in affected or c_id in affected
    assert b_id not in result.stable_ids
    assert b_id not in result.destabilized_ids
    assert b_id not in result.orphaned_ids


def test_stable_in_triangle():
    """Removing one node of a triangle leaves the other two still connected (stable)."""
    bm, (a_id, b_id, c_id) = _triangle_map()
    result = bm.simulate_removal(a_id)

    # B and C are still connected to each other, so both should be stable
    assert b_id in result.stable_ids
    assert c_id in result.stable_ids
    assert result.orphaned_ids == []


def test_removed_id_not_in_results():
    """The removed_id never appears in stable, destabilized, or orphaned lists."""
    bm, hub_id, _ = _star_map()
    result = bm.simulate_removal(hub_id)

    all_result_ids = (
        result.stable_ids + result.destabilized_ids + result.orphaned_ids
    )
    assert hub_id not in all_result_ids


def test_unknown_belief_raises():
    """simulate_removal(99) raises KeyError for a non-existent belief."""
    bm = BeliefMap()
    bm.add_belief("Only belief")

    with pytest.raises(KeyError, match="99"):
        bm.simulate_removal(99)
