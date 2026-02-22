"""Utility helpers for the Belief Compatibility Mapper."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import tempfile

import numpy as np

from engine import MAX_BELIEFS, BeliefMap
from models import Belief

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Persistence (simple JSON + .npy pair) with atomic writes
# ------------------------------------------------------------------


def _atomic_write_text(path: pathlib.Path, content: str) -> None:
    """Write *content* to *path* atomically via temp file + rename."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def _atomic_save_npy(path: pathlib.Path, arr: np.ndarray) -> None:
    """Write a numpy array to *path* atomically via temp file + rename."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        np.save(tmp, arr)
        # np.save appends .npy if missing; handle both cases
        tmp_npy = tmp if tmp.endswith(".npy") else tmp + ".npy"
        if os.path.exists(tmp_npy) and tmp_npy != tmp:
            os.replace(tmp_npy, path)
            if os.path.exists(tmp):
                os.unlink(tmp)
        else:
            os.replace(tmp, path)
    except BaseException:
        for p in (tmp, tmp + ".npy"):
            if os.path.exists(p):
                os.unlink(p)
        raise


def save_map(bmap: BeliefMap, directory: str | pathlib.Path) -> None:
    """Persist a BeliefMap to *directory*/beliefs.json and scores.npy.

    Uses atomic writes (temp file + os.replace) to prevent corruption
    if the process is interrupted mid-save.
    """
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    beliefs_data = [b.model_dump() for b in bmap.list_beliefs()]
    _atomic_write_text(
        directory / "beliefs.json",
        json.dumps(beliefs_data, indent=2),
    )
    _atomic_save_npy(directory / "scores.npy", bmap.scores)
    _atomic_save_npy(directory / "similarity.npy", bmap.similarity)


def load_map(directory: str | pathlib.Path) -> BeliefMap:
    """Load a BeliefMap from a previously saved directory.

    Validates that loaded matrices have the expected shape; logs a
    warning and re-initializes if they do not.
    """
    directory = pathlib.Path(directory)
    expected_shape = (MAX_BELIEFS, MAX_BELIEFS)

    bmap = BeliefMap()
    beliefs_data = json.loads(
        (directory / "beliefs.json").read_text(encoding="utf-8")
    )
    for entry in beliefs_data:
        belief = Belief(**entry)
        bmap.beliefs[belief.id] = belief

    scores_path = directory / "scores.npy"
    if scores_path.exists():
        loaded = np.load(scores_path)
        if loaded.shape == expected_shape:
            bmap.scores = loaded
        else:
            logger.warning(
                "scores.npy has shape %s, expected %s; re-initializing matrix",
                loaded.shape, expected_shape,
            )

    similarity_path = directory / "similarity.npy"
    if similarity_path.exists():
        loaded = np.load(similarity_path)
        if loaded.shape == expected_shape:
            bmap.similarity = loaded
        else:
            logger.warning(
                "similarity.npy has shape %s, expected %s; re-initializing matrix",
                loaded.shape, expected_shape,
            )

    return bmap


def save_snapshot(
    bmap: BeliefMap,
    patient_dir: pathlib.Path,
    label: str,
    notes: str = "",
) -> pathlib.Path:
    """Save a named, timestamped snapshot of the current belief map."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = label.strip().replace(" ", "_")[:40]
    snap_dir = patient_dir / "snapshots" / f"{ts}_{slug}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    save_map(bmap, snap_dir)

    meta = {
        "label": label,
        "notes": notes,
        "created_at": datetime.now().isoformat(),
        "belief_count": len(bmap.beliefs),
    }
    _atomic_write_text(snap_dir / "snapshot_meta.json", json.dumps(meta, indent=2))
    return snap_dir


def list_snapshots(patient_dir: pathlib.Path) -> list[dict]:
    """Return snapshot metadata dicts sorted by created_at ascending."""
    snaps_dir = patient_dir / "snapshots"
    if not snaps_dir.exists():
        return []
    results = []
    for snap_dir in sorted(snaps_dir.iterdir()):
        meta_path = snap_dir / "snapshot_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["path"] = snap_dir
            results.append(meta)
    return sorted(results, key=lambda m: m["created_at"])


def load_snapshot(snapshot_path: pathlib.Path) -> BeliefMap:
    """Load a snapshot as a BeliefMap (read-only by convention)."""
    return load_map(snapshot_path)


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------


def format_belief(belief: Belief, verbose: bool = False) -> str:
    """Return a human-readable one-liner (or multi-line if verbose)."""
    header = f"[{belief.id:>2}] {belief.text}"
    if not verbose:
        return header
    lines = [header]
    if belief.expanded:
        lines.append(f"     Expanded: {belief.expanded}")
    if belief.embedding:
        lines.append(f"     Embedding dim: {len(belief.embedding)}")
    return "\n".join(lines)
