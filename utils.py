"""Utility helpers for the Belief Compatibility Mapper."""

from __future__ import annotations

import json
import pathlib

import numpy as np

from engine import BeliefMap
from models import Belief

# ------------------------------------------------------------------
# Persistence (simple JSON + .npy pair)
# ------------------------------------------------------------------


def save_map(bmap: BeliefMap, directory: str | pathlib.Path) -> None:
    """Persist a BeliefMap to *directory*/beliefs.json and scores.npy."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    beliefs_data = [b.model_dump() for b in bmap.list_beliefs()]
    (directory / "beliefs.json").write_text(
        json.dumps(beliefs_data, indent=2), encoding="utf-8"
    )
    np.save(directory / "scores.npy", bmap.scores)
    np.save(directory / "similarity.npy", bmap.similarity)


def load_map(directory: str | pathlib.Path) -> BeliefMap:
    """Load a BeliefMap from a previously saved directory."""
    directory = pathlib.Path(directory)

    bmap = BeliefMap()
    beliefs_data = json.loads(
        (directory / "beliefs.json").read_text(encoding="utf-8")
    )
    for entry in beliefs_data:
        belief = Belief(**entry)
        bmap.beliefs[belief.id] = belief

    scores_path = directory / "scores.npy"
    if scores_path.exists():
        bmap.scores = np.load(scores_path)

    similarity_path = directory / "similarity.npy"
    if similarity_path.exists():
        bmap.similarity = np.load(similarity_path)

    return bmap


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
