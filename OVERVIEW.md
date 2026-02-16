# Belief Compatibility Mapper

A CLI tool that maps relationships between personal beliefs using vector
embeddings and LLM-powered logical analysis.

## What it does

You feed it belief statements. It tells you which ones harmonize, which ones
create tension, and which ones flat-out contradict each other — backed by
formal logic reasoning from Claude.

## Pipeline

```
 Add beliefs ──> Embed (OpenAI) ──> Cosine similarity ──> Filter interesting ──> Deep analysis (Claude) ──> Visualize
                                                           pairs                  with scored results       (heatmap / network)
```

### Step 1 — Manage beliefs

Add plain-language belief statements, optionally with an expanded definition
(for richer LLM context) and thematic tags (for cluster-based pairing).

```bash
python main.py add "Free will exists" --expanded "Humans possess libertarian free will" --tags "agency,philosophy"
python main.py add "Determinism is true" --tags "agency,philosophy"
python main.py add "I like apples"
python main.py list -v
python main.py remove 2
```

Supports up to **50 beliefs** tracked by integer ID.

### Step 2 — Generate embeddings

Call OpenAI to produce vector embeddings for each belief. Uses the `expanded`
text when available for richer semantic representation.

```bash
export OPENAI_API_KEY=sk-...
python main.py embed
```

### Step 3 — Compute cosine similarity

Build a 50x50 similarity matrix from the embeddings. This is a fast,
token-free way to measure semantic overlap between every pair.

```bash
python main.py similarity
```

### Step 4 — Identify interesting pairs

Filter down to pairs that are actually worth comparing via an LLM. A pair
qualifies if:

- **Cosine similarity >= 0.7** (semantic overlap), _or_
- **Shared thematic tag** (e.g. both tagged `"ethics"`)

Everything else (e.g. "I like apples" vs. "I believe in the gold standard")
is skipped to conserve LLM tokens.

```bash
python main.py interesting --threshold 0.7 --top 20
```

### Step 5 — LLM logical tension analysis

Send each interesting pair to **Claude Sonnet** with a formal-logic system
prompt. The model classifies the relationship and returns:

| Category              | Score range | Meaning                                      |
|-----------------------|-------------|----------------------------------------------|
| Mutually Entailed     | +1.0        | One belief necessitates the other             |
| Compatible/Harmonious | +0.5 to +0.9 | They support the same worldview             |
| Neutral               |  0.0        | Unrelated                                     |
| Tensioned             | -0.1 to -0.5 | Hard to hold both without cognitive dissonance |
| Contradictory         | -0.6 to -1.0 | Logically impossible to hold both            |

Each result includes a one-sentence justification.

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Single pair
python main.py analyze 0 1

# Batch all interesting pairs
python main.py analyze-all --threshold 0.7 --top 20
```

### Step 6 — Visualize

Generate interactive HTML files you can open in any browser.

**Heatmap** — the full compatibility matrix as a color-coded grid. Hover any
cell to see both belief texts and the exact score. NaN (unscored) cells render
as gaps.

```bash
python main.py heatmap                          # default: heatmap.html
python main.py heatmap --matrix similarity      # cosine similarity instead
python main.py heatmap -o my_heatmap.html       # custom filename
```

**Network graph** — beliefs as nodes, compatibility as edges. Only edges with
`|score| > 0.5` are drawn to keep things readable. Uses a **force-directed
layout** (NetworkX spring layout) where positive edges act as attraction
springs so that compatible beliefs naturally cluster together. Negative
(tensioned/contradictory) edges are drawn as dashed red lines but do not
pull nodes together.

```bash
python main.py network                          # default: network.html
python main.py network --threshold 0.3          # lower bar for edges
python main.py network -o my_graph.html         # custom filename
```

Both outputs use Plotly with CDN-loaded JS, so the HTML files are lightweight
and fully interactive (zoom, pan, hover tooltips).

### One-shot pipeline: `analyze-map`

Feed a plain text file (one belief per line) and get the full pipeline in a
single command. Only new beliefs are added, only missing embeddings are
generated, and only unscored pairs are sent to Claude. Everything else is
served from the SQLite cache.

```bash
python main.py analyze-map beliefs.txt
python main.py analyze-map beliefs.txt --output my_report --threshold 0.6 --top 30
```

Outputs: `<prefix>_heatmap.html` and `<prefix>_network.html`.

## Caching

A **SQLite database** (`data/cache.db`) stores all API results keyed by
content hash so that re-runs are nearly free:

| What is cached           | Key                                         | Effect                                     |
|--------------------------|---------------------------------------------|--------------------------------------------|
| OpenAI embeddings        | SHA-256 of belief text + model name         | Adding 1 belief to 99 re-embeds only the 1 |
| Claude tension results   | SHA-256 of sorted text pair + model name    | Already-judged pairs are never re-sent     |

Cache keys are **content-based**: editing a belief's text automatically
invalidates its entries. The pair hash is order-independent
(`(A, B) == (B, A)`).

## Rate limiting

A **sliding-window rate limiter** (default 50 RPM) wraps every Claude API
call. When the budget is exhausted the process sleeps until a slot opens.
This handles the 50-belief worst case (up to 1,225 pairs) gracefully
without hitting API errors.

## Project structure

```
belief-compatibility-mapper/
  main.py           CLI entry-point (Typer). All user-facing commands.
  models.py         Pydantic models: Belief, TensionCategory, TensionResult.
  engine.py         BeliefMap class: CRUD, embeddings, similarity, tension analysis.
  cache.py          SQLite result cache + sliding-window rate limiter.
  visualization.py  Plotly heatmap + NetworkX force-directed graph exports.
  utils.py          Persistence (JSON + .npy) and display helpers.
  pyproject.toml    Dependencies and project metadata.
```

## Data storage

All state persists to a `./data/` directory:

- `beliefs.json` — serialized belief objects
- `scores.npy` — 50x50 compatibility score matrix (LLM-judged)
- `similarity.npy` — 50x50 cosine similarity matrix (embedding-based)
- `cache.db` — SQLite cache for embeddings and tension results

## Requirements

- Python >= 3.11
- `OPENAI_API_KEY` for embeddings
- `ANTHROPIC_API_KEY` for tension analysis

```bash
pip install numpy openai anthropic pydantic typer plotly networkx
```

## Full CLI reference

| Command       | Description                                        |
|---------------|----------------------------------------------------|
| `add`         | Add a new belief (with optional tags / expansion)  |
| `list`        | List all beliefs (`-v` for details)                |
| `remove`      | Remove a belief by ID                              |
| `score`       | Manually set a compatibility score between two IDs |
| `show-score`  | Display the score for a pair                       |
| `pairs`       | Show all scored pairs (with optional threshold)    |
| `embed`       | Generate OpenAI embeddings                         |
| `similarity`  | Compute cosine-similarity matrix                   |
| `interesting` | Show pairs worth deep-analyzing                    |
| `analyze`     | Run Claude tension analysis on a single pair       |
| `analyze-all` | Batch-analyze all interesting pairs via Claude     |
| `heatmap`     | Export interactive HTML heatmap (scores/similarity) |
| `network`     | Export force-directed network graph as HTML        |
| `analyze-map` | One-shot pipeline: text file in, HTML vis out      |
