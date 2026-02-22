"""Tests for BeliefRole functionality."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine import BeliefMap
from models import BeliefRole, Belief
from utils import save_map, load_map


def test_belief_role_default():
    """New belief has role == UNTAGGED by default."""
    bm = BeliefMap()
    b = bm.add_belief("I value honesty")
    assert b.role == BeliefRole.UNTAGGED


def test_add_belief_with_role():
    """add_belief(role=SELF_SCHEMA) stores the correct role."""
    bm = BeliefMap()
    b = bm.add_belief("I am a caring person", role=BeliefRole.SELF_SCHEMA)
    assert b.role == BeliefRole.SELF_SCHEMA
    # Verify it's accessible from the map too
    assert bm.get_belief(b.id).role == BeliefRole.SELF_SCHEMA


def test_edit_belief_role(tmp_path):
    """edit_belief(role=ASPIRATION) changes role without clearing scores."""
    bm = BeliefMap()
    a = bm.add_belief("I want to be more patient")
    b = bm.add_belief("I believe growth is possible")
    # Manually set a score so we can verify it's preserved
    bm.set_score(a.id, b.id, 0.7)

    bm.edit_belief(a.id, role=BeliefRole.ASPIRATION)

    assert bm.get_belief(a.id).role == BeliefRole.ASPIRATION
    # Score must not have been cleared
    assert bm.get_score(a.id, b.id) == pytest.approx(0.7)


def test_role_json_roundtrip(tmp_path):
    """save_map / load_map preserves the role field."""
    bm = BeliefMap()
    bm.add_belief("I trust others easily", role=BeliefRole.CORE_VALUE)
    bm.add_belief("Failure is a learning opportunity", role=BeliefRole.BRIDGE_BELIEF)

    save_map(bm, tmp_path)
    loaded = load_map(tmp_path)

    roles = {b.text: b.role for b in loaded.list_beliefs()}
    assert roles["I trust others easily"] == BeliefRole.CORE_VALUE
    assert roles["Failure is a learning opportunity"] == BeliefRole.BRIDGE_BELIEF


def test_untagged_from_legacy_json(tmp_path):
    """JSON without 'role' key loads as UNTAGGED (Pydantic default)."""
    # Simulate a legacy beliefs.json that has no 'role' field
    legacy_data = [
        {"id": 0, "text": "Legacy belief A", "expanded": "", "embedding": [], "tags": []},
        {"id": 1, "text": "Legacy belief B", "expanded": "", "embedding": [], "tags": []},
    ]
    beliefs_path = tmp_path / "beliefs.json"
    beliefs_path.write_text(json.dumps(legacy_data, indent=2), encoding="utf-8")

    loaded = load_map(tmp_path)
    for b in loaded.list_beliefs():
        assert b.role == BeliefRole.UNTAGGED
