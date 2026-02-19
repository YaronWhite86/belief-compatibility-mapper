"""Tests for the quick-add preset beliefs CLI command."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from engine import BeliefMap


SAMPLE_DATA = {
    "categories": [
        {"name": "Ethics", "beliefs": ["Duty matters.", "Outcomes matter.", "Character matters."]},
        {"name": "Politics", "beliefs": ["Liberty first.", "Equality matters."]},
    ]
}


@pytest.fixture()
def common_beliefs_file(tmp_path):
    p = tmp_path / "common_beliefs.json"
    p.write_text(json.dumps(SAMPLE_DATA), encoding="utf-8")
    return p


def test_quick_add_list_all(common_beliefs_file):
    from typer.testing import CliRunner
    from main import app
    import main

    runner = CliRunner()
    with patch.object(main, "COMMON_BELIEFS_PATH", common_beliefs_file):
        result = runner.invoke(app, ["quick-add"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "Ethics" in result.output
    assert "Politics" in result.output
    assert "Duty matters." in result.output
    assert "5 preset belief" in result.output


def test_quick_add_filter_category(common_beliefs_file):
    from typer.testing import CliRunner
    from main import app
    import main

    runner = CliRunner()
    with patch.object(main, "COMMON_BELIEFS_PATH", common_beliefs_file):
        result = runner.invoke(app, ["quick-add", "--category", "ethics"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "Ethics" in result.output
    assert "Politics" not in result.output


def test_quick_add_filter_unknown_category(common_beliefs_file):
    from typer.testing import CliRunner
    from main import app
    import main

    runner = CliRunner()
    with patch.object(main, "COMMON_BELIEFS_PATH", common_beliefs_file):
        result = runner.invoke(app, ["quick-add", "--category", "zzz"], catch_exceptions=False)

    assert result.exit_code != 0


def test_quick_add_add_belief(common_beliefs_file, tmp_path):
    from typer.testing import CliRunner
    from main import app
    import main
    from cache import RateLimiter, ResultCache

    runner = CliRunner()
    bm = BeliefMap()
    bm.cache = ResultCache(tmp_path / "cache.db")
    bm.rate_limiter = RateLimiter(max_rpm=10_000)

    with patch.object(main, "COMMON_BELIEFS_PATH", common_beliefs_file):
        with patch.object(main, "_get_map", return_value=bm):
            with patch.object(main, "_persist"):
                result = runner.invoke(app, ["quick-add", "--add", "1"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "Added belief" in result.output
    assert len(bm.beliefs) == 1
    assert bm.beliefs[0].text == "Duty matters."


def test_quick_add_add_out_of_range(common_beliefs_file):
    from typer.testing import CliRunner
    from main import app
    import main

    runner = CliRunner()
    with patch.object(main, "COMMON_BELIEFS_PATH", common_beliefs_file):
        result = runner.invoke(app, ["quick-add", "--add", "99"], catch_exceptions=False)

    assert result.exit_code != 0


def test_quick_add_missing_file(tmp_path):
    from typer.testing import CliRunner
    from main import app
    import main

    runner = CliRunner()
    missing = tmp_path / "nonexistent.json"
    with patch.object(main, "COMMON_BELIEFS_PATH", missing):
        result = runner.invoke(app, ["quick-add"], catch_exceptions=False)

    assert result.exit_code != 0
    assert "common_beliefs.json" in result.output
