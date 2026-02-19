"""Tests for the identify_bedrock_principles feature."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from engine import BEDROCK_SYSTEM_PROMPT, BeliefMap
from models import BedrockPrinciple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_map(belief_map_factory, texts: list[str]) -> BeliefMap:
    return belief_map_factory(beliefs=texts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_valid_principles(belief_map_factory):
    """Happy path: returns a list of BedrockPrinciple with correct types."""
    bm = _make_map(belief_map_factory, ["Free markets allocate resources efficiently.", "Government intervention causes market distortions."])
    results = bm.identify_bedrock_principles()
    assert isinstance(results, list)
    assert len(results) >= 1
    for p in results:
        assert isinstance(p, BedrockPrinciple)
        assert len(p.principle) > 0
        assert len(p.belief_ids) >= 2
        assert 0.0 <= p.coherence <= 1.0
        assert len(p.explanation) > 0


def test_requires_two_beliefs_empty(belief_map_factory):
    """Raises ValueError when map has 0 beliefs."""
    bm = belief_map_factory()
    with pytest.raises(ValueError, match="At least 2"):
        bm.identify_bedrock_principles()


def test_requires_two_beliefs_single(belief_map_factory):
    """Raises ValueError when map has only 1 belief."""
    bm = belief_map_factory(beliefs=["Only one belief here."])
    with pytest.raises(ValueError, match="At least 2"):
        bm.identify_bedrock_principles()


def test_filters_invalid_belief_ids(belief_map_factory, mock_anthropic):
    """Non-existent IDs in principle are dropped; principle survives if ≥2 valid IDs remain."""
    bm = _make_map(belief_map_factory, ["A", "B", "C"])
    valid_ids = set(bm.beliefs.keys())

    # Override mock to return a principle with one invalid ID
    def _override(*, model, max_tokens, system, messages, **kw):
        ids = sorted(valid_ids)[:2]
        payload = {"principles": [{
            "principle": "Overridden principle",
            "belief_ids": ids + [999],  # 999 is invalid
            "coherence": 0.7,
            "explanation": "Mock explanation.",
        }]}
        return SimpleNamespace(
            content=[SimpleNamespace(text=json.dumps(payload))],
            stop_reason="end_turn",
        )

    mock_anthropic.messages.create.side_effect = _override
    results = bm.identify_bedrock_principles()
    assert len(results) == 1
    assert 999 not in results[0].belief_ids
    assert len(results[0].belief_ids) >= 2


def test_single_api_call(belief_map_factory, mock_anthropic):
    """Exactly one messages.create call per unique map state."""
    bm = _make_map(belief_map_factory, ["Belief alpha.", "Belief beta."])
    bm.identify_bedrock_principles()
    assert mock_anthropic.messages.create.call_count == 1


def test_uses_expanded_text(belief_map_factory, mock_anthropic):
    """expanded definition is included in the prompt when set; raw text is absent."""
    bm = belief_map_factory()
    b0 = bm.add_belief("Short text.", expanded="This is the expanded definition.")
    b1 = bm.add_belief("Another belief.")

    bm.identify_bedrock_principles()

    call_args = mock_anthropic.messages.create.call_args
    user_content = call_args.kwargs["messages"][0]["content"]
    assert "expanded definition" in user_content
    assert "Short text." not in user_content


def test_caching(belief_map_factory, mock_anthropic):
    """Second call with same beliefs returns cached result; no extra API calls."""
    bm = _make_map(belief_map_factory, ["Belief X.", "Belief Y."])
    r1 = bm.identify_bedrock_principles()
    call_count_after_first = mock_anthropic.messages.create.call_count

    r2 = bm.identify_bedrock_principles()
    assert mock_anthropic.messages.create.call_count == call_count_after_first
    assert r1[0].principle == r2[0].principle


def test_cache_invalidated_after_add(belief_map_factory, mock_anthropic):
    """Adding a belief forces a new API call (cache key changes)."""
    bm = _make_map(belief_map_factory, ["Belief one.", "Belief two."])
    bm.identify_bedrock_principles()
    count_after_first = mock_anthropic.messages.create.call_count

    bm.add_belief("A third belief changes the map hash.")
    bm.identify_bedrock_principles()
    assert mock_anthropic.messages.create.call_count > count_after_first


def test_model_kwarg(belief_map_factory, mock_anthropic):
    """The model kwarg is forwarded to messages.create."""
    bm = _make_map(belief_map_factory, ["P", "Q"])
    bm.identify_bedrock_principles(model="claude-haiku-4-5-20251001")
    call_args = mock_anthropic.messages.create.call_args
    assert call_args.kwargs["model"] == "claude-haiku-4-5-20251001"


def test_uses_bedrock_system_prompt(belief_map_factory, mock_anthropic):
    """BEDROCK_SYSTEM_PROMPT (and not other prompts) is used."""
    from engine import RECOMMEND_SYSTEM_PROMPT, TENSION_SYSTEM_PROMPT

    bm = _make_map(belief_map_factory, ["A", "B"])
    bm.identify_bedrock_principles()
    call_args = mock_anthropic.messages.create.call_args
    system_used = call_args.kwargs["system"]
    assert system_used == BEDROCK_SYSTEM_PROMPT
    assert system_used != TENSION_SYSTEM_PROMPT
    assert system_used != RECOMMEND_SYSTEM_PROMPT


def test_cli_no_api_key(tmp_path):
    """CLI errors with a clear message when ANTHROPIC_API_KEY is unset."""
    from typer.testing import CliRunner
    from main import app

    runner = CliRunner()
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    result = runner.invoke(app, ["identify-bedrock"], env=env, catch_exceptions=False)
    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY" in result.output


def test_cli_output_format(tmp_path, mock_anthropic):
    """CLI output contains principle text, coherence, and supporting beliefs."""
    from typer.testing import CliRunner
    from main import app, _get_map
    import main

    runner = CliRunner()

    # Patch _get_map to return a pre-populated in-memory map
    bm = BeliefMap()
    from cache import RateLimiter, ResultCache
    bm.cache = ResultCache(tmp_path / "cache.db")
    bm.rate_limiter = RateLimiter(max_rpm=10_000)
    bm.add_belief("Markets are efficient.")
    bm.add_belief("Competition drives innovation.")

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
        with patch.object(main, "_get_map", return_value=bm):
            result = runner.invoke(app, ["identify-bedrock"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "bedrock principle" in result.output.lower()
    assert "Coherence:" in result.output
    assert "Supporting beliefs:" in result.output
