"""Tests for snapshot save/load/list functionality."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from engine import BeliefMap
from utils import save_map, save_snapshot, list_snapshots, load_snapshot


def _make_bmap_with_beliefs(*texts: str) -> BeliefMap:
    bm = BeliefMap()
    for t in texts:
        bm.add_belief(t)
    return bm


def test_save_snapshot_creates_files(tmp_path):
    """snapshot_meta.json, beliefs.json, and scores.npy are all created."""
    bm = _make_bmap_with_beliefs("I believe in hard work", "Effort leads to success")
    patient_dir = tmp_path / "case-001"

    snap_dir = save_snapshot(bm, patient_dir, label="Session 1", notes="First session")

    assert (snap_dir / "snapshot_meta.json").exists()
    assert (snap_dir / "beliefs.json").exists()
    assert (snap_dir / "scores.npy").exists()


def test_load_snapshot_restores_map(tmp_path):
    """Loaded beliefs and scores match the saved map."""
    bm = _make_bmap_with_beliefs("Kindness matters", "Effort is rewarded")
    bm.set_score(0, 1, 0.6)
    patient_dir = tmp_path / "case-002"

    snap_dir = save_snapshot(bm, patient_dir, label="Session A")
    loaded = load_snapshot(snap_dir)

    assert {b.text for b in loaded.list_beliefs()} == {
        "Kindness matters", "Effort is rewarded"
    }
    assert loaded.get_score(0, 1) == pytest.approx(0.6)


def test_list_snapshots_sorted(tmp_path):
    """Two snapshots are returned in chronological order."""
    patient_dir = tmp_path / "case-003"
    bm1 = _make_bmap_with_beliefs("Belief X")
    save_snapshot(bm1, patient_dir, label="Session 1")

    # Small sleep to ensure distinct timestamps
    time.sleep(0.05)

    bm2 = _make_bmap_with_beliefs("Belief X", "Belief Y")
    save_snapshot(bm2, patient_dir, label="Session 2")

    snaps = list_snapshots(patient_dir)
    assert len(snaps) == 2
    assert snaps[0]["label"] == "Session 1"
    assert snaps[1]["label"] == "Session 2"
    assert snaps[0]["created_at"] < snaps[1]["created_at"]


def test_snapshot_meta_fields(tmp_path):
    """label, notes, created_at, and belief_count are all present in meta."""
    patient_dir = tmp_path / "case-004"
    bm = _make_bmap_with_beliefs("Trust is foundational", "People are generally good")

    snap_dir = save_snapshot(bm, patient_dir, label="Intake", notes="Initial session notes")

    meta_raw = json.loads((snap_dir / "snapshot_meta.json").read_text(encoding="utf-8"))
    assert meta_raw["label"] == "Intake"
    assert meta_raw["notes"] == "Initial session notes"
    assert "created_at" in meta_raw
    assert meta_raw["belief_count"] == 2


def test_snapshots_are_independent(tmp_path):
    """Saving a second snapshot does not overwrite the first."""
    patient_dir = tmp_path / "case-005"
    bm1 = _make_bmap_with_beliefs("I am resilient")
    snap1 = save_snapshot(bm1, patient_dir, label="Session 1")

    time.sleep(0.05)

    bm2 = _make_bmap_with_beliefs("I am resilient", "I can change")
    snap2 = save_snapshot(bm2, patient_dir, label="Session 2")

    assert snap1 != snap2
    loaded1 = load_snapshot(snap1)
    loaded2 = load_snapshot(snap2)

    assert len(loaded1.list_beliefs()) == 1
    assert len(loaded2.list_beliefs()) == 2
