"""Shared fixtures for the Belief Compatibility Mapper test suite."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Ensure openai and anthropic are importable even if not installed,
# so that unittest.mock.patch can target their attributes.
if "openai" not in sys.modules:
    sys.modules["openai"] = MagicMock()
if "anthropic" not in sys.modules:
    sys.modules["anthropic"] = MagicMock()

from cache import RateLimiter, ResultCache
from engine import BeliefMap

PROFILES_DIR = Path(__file__).parent / "profiles"


# ------------------------------------------------------------------
# Deterministic fake API responses
# ------------------------------------------------------------------


def _fake_embedding(text: str, dims: int = 16) -> list[float]:
    """Return a deterministic unit-ish vector derived from *text*."""
    h = hashlib.sha256(text.encode()).digest()
    raw = [b / 255.0 for b in h[:dims]]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


def _fake_tension_json(user_msg: str) -> str:
    """Return a deterministic TensionResult JSON derived from *user_msg*."""
    h = hashlib.sha256(user_msg.encode()).digest()
    score_raw = (h[0] / 255.0) * 2 - 1  # map byte to [-1, 1]
    score = round(max(-1.0, min(1.0, score_raw)), 2)

    if score >= 0.5:
        cat = "compatible_harmonious"
    elif score >= 0.0:
        cat = "neutral"
    elif score >= -0.5:
        cat = "tensioned"
    else:
        cat = "contradictory"

    return json.dumps(
        {"score": score, "category": cat, "justification": "Mock analysis result."}
    )


def _fake_recommend_json(count: int) -> str:
    """Return a deterministic recommendations JSON with *count* items."""
    recs = [
        {"text": f"Mock recommended belief {i + 1}", "justification": "Mock justification."}
        for i in range(count)
    ]
    return json.dumps({"recommendations": recs})


# ------------------------------------------------------------------
# Mock fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def mock_openai():
    """Patch ``openai.OpenAI`` so embedding calls return deterministic vectors."""

    def _embeddings_create(*, input, model, **kw):  # noqa: A002
        data = [
            SimpleNamespace(embedding=_fake_embedding(text)) for text in input
        ]
        return SimpleNamespace(data=data)

    with patch("openai.OpenAI") as cls:
        instance = cls.return_value
        instance.embeddings.create.side_effect = _embeddings_create
        yield instance


@pytest.fixture()
def mock_anthropic():
    """Patch ``anthropic.Anthropic`` so tension calls return canned JSON."""

    def _messages_create(*, model, max_tokens, system, messages, **kw):
        import re
        user_msg = messages[0]["content"]
        if "recommendations" in system:
            m = re.search(r"Number of recommendations requested:\s*(\d+)", user_msg)
            count = int(m.group(1)) if m else 1
            return SimpleNamespace(
                content=[SimpleNamespace(text=_fake_recommend_json(count))],
                stop_reason="end_turn",
            )
        return SimpleNamespace(
            content=[SimpleNamespace(text=_fake_tension_json(user_msg))],
            stop_reason="end_turn",
        )

    with patch("anthropic.Anthropic") as cls:
        instance = cls.return_value
        instance.messages.create.side_effect = _messages_create
        yield instance


# ------------------------------------------------------------------
# Profile loader
# ------------------------------------------------------------------


def load_profile(name: str) -> list[dict]:
    """Read a profile JSON from tests/profiles/."""
    path = PROFILES_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------
# BeliefMap factory
# ------------------------------------------------------------------


@pytest.fixture()
def belief_map_factory(tmp_path, mock_openai, mock_anthropic):
    """Return a factory that builds a fully-wired BeliefMap.

    Usage in tests::

        bm = belief_map_factory("philosopher")       # from profile
        bm = belief_map_factory(beliefs=["A", "B"])   # ad-hoc
    """
    created: list[BeliefMap] = []

    def _factory(
        profile: str | None = None,
        beliefs: list[str] | None = None,
    ) -> BeliefMap:
        bm = BeliefMap()
        bm.cache = ResultCache(tmp_path / "cache.db")
        bm.rate_limiter = RateLimiter(max_rpm=10_000)

        if profile:
            for entry in load_profile(profile):
                bm.add_belief(entry["text"], tags=entry.get("tags", []))
        elif beliefs:
            for text in beliefs:
                bm.add_belief(text)

        created.append(bm)
        return bm

    yield _factory

    for bm in created:
        if bm.cache:
            bm.cache.close()
