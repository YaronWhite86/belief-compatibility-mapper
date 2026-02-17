# CLAUDE.md

## Project Overview
Belief Compatibility Mapper — a CLI tool that maps logical relationships between
personal beliefs using vector embeddings and LLM-powered tension analysis.

Pipeline: add beliefs → embed → cosine similarity → filter interesting pairs → Claude analysis → visualize (heatmap + network graph)

## Quick Reference

```bash
# Run tests (46 tests, no API keys needed)
python -m pytest tests/ -v

# Run offline demo (no API keys needed)
python main.py demo mixed_debate --verbose

# List available demo profiles
python main.py demo --list

# Full pipeline (requires ANTHROPIC_API_KEY)
python main.py analyze-map beliefs.txt --open
```

## Architecture

```
main.py           CLI (Typer). All user-facing commands. Module-level _bmap singleton.
engine.py         BeliefMap class: CRUD, embeddings, similarity, LLM tension analysis.
models.py         Pydantic models: Belief, TensionCategory, TensionResult.
cache.py          SQLite result cache (content-addressed) + sliding-window rate limiter.
visualization.py  Plotly heatmap + NetworkX force-directed graph exports.
utils.py          Persistence (JSON + .npy) with atomic writes, display helpers.
profiles/         Pre-scored demo profiles (JSON). No API keys needed.
tests/            pytest suite with mocked API responses. All tests run offline.
```

## Key Constraints

- **MAX_BELIEFS = 15** (engine.py). Belief IDs are 0-14 and double as numpy matrix indices.
  Changing this requires updating engine.py, models.py (Pydantic `lt=` validator),
  and main.py (truncation message in analyze-map). Existing .npy files will fail
  shape validation on load if this changes.
- **Belief IDs are reused after deletion.** ID = matrix row/col index. Be careful.
- **Duplicate belief texts are rejected** by add_belief().
- **scikit-learn is optional.** The default embedding model (local-tfidf) falls back
  to local-bow automatically if sklearn is not installed. Never make it a hard dependency.

## API & Models

- Tension analysis defaults to `claude-sonnet-4-5-latest` (not a dated snapshot).
- Embeddings default to `local-tfidf` (offline). OpenAI models also supported.
- API keys are read from env vars: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.
- The `_check_anthropic_key()` helper in main.py validates before starting LLM work.

## Error Handling Patterns

- LLM responses: strip markdown fences → json.loads → Pydantic validation.
  3 retries with exponential backoff. Failed pairs are skipped, not fatal.
- CLI commands: wrap in try/except → `_handle_error()` for user-friendly messages.
- Persistence: atomic writes (temp file + os.replace). Incremental save every 10 pairs
  during batch analysis.

## Testing

- Tests use mocked API clients (conftest.py). No real API calls ever.
- `mock_anthropic` returns deterministic JSON with `stop_reason="end_turn"`.
- `mock_openai` returns deterministic embeddings via SHA-256 hashing.
- `belief_map_factory` fixture wires up cache + rate limiter automatically.
- Test profiles live in `tests/profiles/` (separate from demo profiles in `profiles/`).

## Style & Conventions

- Tags are passed via `add_belief(text, tags=[...])`, never by mutating the object after.
- Use `belief.expanded or belief.text` when you need the best available description
  (this pattern appears in several places — keep it consistent).
- Demo HTML output goes to `offline_output/` with timestamps. The folder is gitignored.
- Profile JSON files have optional `"justifications"` dict — new fields must be optional
  to avoid breaking existing profiles.
- Commit messages follow conventional commits (feat:, fix:, chore:, docs:).

## Things to Avoid

- Don't add required dependencies. New deps should be optional with graceful fallback.
- Don't generate or modify files in `data/` during tests — use `tmp_path`.
- Don't hardcode dated model snapshots (e.g. `claude-sonnet-4-5-20250929`). Use aliases.
- Don't break offline demo functionality. The tool must work fully without API keys
  via the `demo` command and `local-bow`/`local-tfidf` embeddings.
