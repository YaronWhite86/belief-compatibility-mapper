"""Tests for BeliefMap.recommend_belief() and the recommend CLI command."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from engine import RECOMMEND_SYSTEM_PROMPT, BeliefMap
from models import BeliefRecommendation
from main import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TWO_BELIEFS = ["I value individual freedom above collective welfare.", "Free markets allocate resources efficiently."]
FIVE_BELIEFS = [
    "I value individual freedom above collective welfare.",
    "Free markets allocate resources efficiently.",
    "Climate change is the defining challenge of our era.",
    "Education should be universally accessible and free.",
    "Science is the most reliable path to truth.",
]


def _make_map(belief_map_factory, texts: list[str]) -> BeliefMap:
    return belief_map_factory(beliefs=texts)


# ---------------------------------------------------------------------------
# 1. Default call returns exactly 1 recommendation
# ---------------------------------------------------------------------------

def test_recommend_default_count(belief_map_factory):
    bm = _make_map(belief_map_factory, TWO_BELIEFS)
    recs = bm.recommend_belief()
    assert len(recs) == 1
    assert isinstance(recs[0], BeliefRecommendation)


# ---------------------------------------------------------------------------
# 2. count=N returns exactly N recommendations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 5])
def test_recommend_count_n(belief_map_factory, n):
    bm = _make_map(belief_map_factory, FIVE_BELIEFS)
    recs = bm.recommend_belief(count=n)
    assert len(recs) == n
    for rec in recs:
        assert isinstance(rec, BeliefRecommendation)
        assert rec.text
        assert rec.justification


# ---------------------------------------------------------------------------
# 3. All valid styles are accepted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", ["harmonious", "complementary", "challenging"])
def test_recommend_valid_styles(belief_map_factory, style):
    bm = _make_map(belief_map_factory, TWO_BELIEFS)
    recs = bm.recommend_belief(style=style)
    assert len(recs) == 1


# ---------------------------------------------------------------------------
# 4. Invalid style raises ValueError
# ---------------------------------------------------------------------------

def test_recommend_invalid_style(belief_map_factory):
    bm = _make_map(belief_map_factory, TWO_BELIEFS)
    with pytest.raises(ValueError, match="Invalid style"):
        bm.recommend_belief(style="bad_style")


# ---------------------------------------------------------------------------
# 5. Empty map raises ValueError
# ---------------------------------------------------------------------------

def test_recommend_empty_map(belief_map_factory):
    bm = belief_map_factory()  # no beliefs
    with pytest.raises(ValueError, match="At least 2 beliefs"):
        bm.recommend_belief()


# ---------------------------------------------------------------------------
# 6. Single-belief map raises ValueError
# ---------------------------------------------------------------------------

def test_recommend_single_belief(belief_map_factory):
    bm = _make_map(belief_map_factory, ["Only one belief."])
    with pytest.raises(ValueError, match="At least 2 beliefs"):
        bm.recommend_belief()


# ---------------------------------------------------------------------------
# 7. Exactly one API call is made per recommend_belief() call
# ---------------------------------------------------------------------------

def test_recommend_single_api_call(belief_map_factory, mock_anthropic):
    bm = _make_map(belief_map_factory, TWO_BELIEFS)
    bm._anthropic_client = mock_anthropic
    bm.recommend_belief(count=2)
    assert mock_anthropic.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# 8. Scored pairs appear in the user prompt when present
# ---------------------------------------------------------------------------

def test_recommend_includes_scored_pairs(belief_map_factory, mock_anthropic):
    bm = _make_map(belief_map_factory, TWO_BELIEFS)
    bm.set_score(0, 1, 0.75)
    bm._anthropic_client = mock_anthropic
    bm.recommend_belief()

    call_kwargs = mock_anthropic.messages.create.call_args.kwargs
    user_content = call_kwargs["messages"][0]["content"]
    assert "compatibility pairs" in user_content.lower() or "score=" in user_content


# ---------------------------------------------------------------------------
# 9. expanded text is used in the prompt (expanded-or-text pattern)
# ---------------------------------------------------------------------------

def test_recommend_uses_expanded_text(belief_map_factory, mock_anthropic):
    bm = belief_map_factory()
    bm.add_belief("Raw text A", expanded="Expanded definition A")
    bm.add_belief("Raw text B", expanded="Expanded definition B")
    bm._anthropic_client = mock_anthropic
    bm.recommend_belief()

    call_kwargs = mock_anthropic.messages.create.call_args.kwargs
    user_content = call_kwargs["messages"][0]["content"]
    assert "Expanded definition A" in user_content
    assert "Expanded definition B" in user_content


# ---------------------------------------------------------------------------
# 10. model kwarg is passed through to the API
# ---------------------------------------------------------------------------

def test_recommend_model_kwarg(belief_map_factory, mock_anthropic):
    bm = _make_map(belief_map_factory, TWO_BELIEFS)
    bm._anthropic_client = mock_anthropic
    custom_model = "claude-opus-4-6"
    bm.recommend_belief(model=custom_model)

    call_kwargs = mock_anthropic.messages.create.call_args.kwargs
    assert call_kwargs["model"] == custom_model


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_cli_recommend_no_api_key(monkeypatch):
    """recommend command exits with error when ANTHROPIC_API_KEY is unset."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["recommend"])
    assert result.exit_code != 0


def test_cli_recommend_invalid_style(monkeypatch):
    """recommend command exits with error for unknown --style."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    result = runner.invoke(app, ["recommend", "--style", "unknown"])
    assert result.exit_code != 0
    assert "style" in result.output.lower() or "style" in (result.stderr or "").lower()
