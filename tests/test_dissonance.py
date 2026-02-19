"""Unit tests for BeliefMap.dissonance_report() — no API mocking needed."""

from __future__ import annotations

import pytest

from engine import BeliefMap


def _make_map(*texts: str) -> BeliefMap:
    """Helper: create a BeliefMap with the given belief texts."""
    bmap = BeliefMap()
    for text in texts:
        bmap.add_belief(text)
    return bmap


def test_no_contradictions_returns_empty():
    """All positive/neutral scores → empty alert list."""
    bmap = _make_map("A", "B", "C")
    bmap.set_score(0, 1, 0.8)
    bmap.set_score(0, 2, 0.3)
    bmap.set_score(1, 2, 0.0)

    alerts = bmap.dissonance_report()
    assert alerts == []


def test_single_contradiction_detected():
    """A pair with score <= threshold produces exactly one alert with correct fields."""
    bmap = _make_map("A", "B")
    bmap.set_score(0, 1, -0.7)

    alerts = bmap.dissonance_report(contradiction_threshold=-0.5)

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.belief_id_a == 0
    assert alert.belief_id_b == 1
    assert abs(alert.score - (-0.7)) < 1e-9
    assert alert.dependent_ids == []
    assert abs(alert.severity - 0.7) < 1e-9


def test_dependent_beliefs_collected():
    """Belief C aligned with A (score >= 0.3) appears in dependent_ids; D below → excluded."""
    bmap = _make_map("A", "B", "C", "D")
    # A and B contradict each other
    bmap.set_score(0, 1, -0.8)
    # C is aligned with A → at risk
    bmap.set_score(2, 0, 0.5)
    # D is below alignment threshold with both A and B → not at risk
    bmap.set_score(3, 0, 0.1)
    bmap.set_score(3, 1, 0.2)

    alerts = bmap.dissonance_report(contradiction_threshold=-0.5, alignment_threshold=0.3)

    assert len(alerts) == 1
    alert = alerts[0]
    assert 2 in alert.dependent_ids
    assert 3 not in alert.dependent_ids


def test_nan_pairs_ignored():
    """Unscored (NaN) pairs are never flagged as contradictions."""
    bmap = _make_map("A", "B", "C")
    # Only score one pair — the others remain NaN
    bmap.set_score(0, 1, 0.5)
    # (0,2) and (1,2) are NaN — should never appear in alerts

    alerts = bmap.dissonance_report(contradiction_threshold=-0.5)
    assert alerts == []


def test_alerts_sorted_by_severity_descending():
    """The most severe alert should appear first."""
    bmap = _make_map("A", "B", "C", "D")
    # A-B: score -0.9 (high severity)
    bmap.set_score(0, 1, -0.9)
    # C-D: score -0.6 (lower severity)
    bmap.set_score(2, 3, -0.6)

    alerts = bmap.dissonance_report(contradiction_threshold=-0.5)

    assert len(alerts) == 2
    assert alerts[0].severity >= alerts[1].severity
    # Specifically the A-B pair should come first
    assert {alerts[0].belief_id_a, alerts[0].belief_id_b} == {0, 1}


def test_severity_clamped_to_one():
    """With score -1.0 and 12 dependents, severity should be clamped to 1.0."""
    # Need 14 beliefs: 2 contradictory (ids 0,1) + 12 dependents
    texts = ["belief_" + str(i) for i in range(14)]
    bmap = _make_map(*texts)

    bmap.set_score(0, 1, -1.0)

    # Make beliefs 2–13 aligned with belief 0
    for dep_id in range(2, 14):
        bmap.set_score(dep_id, 0, 0.5)

    alerts = bmap.dissonance_report(contradiction_threshold=-0.5, alignment_threshold=0.3)

    assert len(alerts) == 1
    assert alerts[0].severity == 1.0
