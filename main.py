"""CLI entry-point for the Belief Compatibility Mapper (Typer-based)."""

from __future__ import annotations

import os
import pathlib
import webbrowser
from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from cache import RateLimiter, ResultCache
from engine import BeliefMap
from utils import format_belief, load_map, save_map
from visualization import export_heatmap, export_network

app = typer.Typer(help="Belief Compatibility Mapper CLI")

DATA_DIR = pathlib.Path("./data")

# Module-level state so sub-commands share the same map.
_bmap: BeliefMap | None = None


def _get_map() -> BeliefMap:
    global _bmap
    if _bmap is None:
        if (DATA_DIR / "beliefs.json").exists():
            _bmap = load_map(DATA_DIR)
        else:
            _bmap = BeliefMap()
        _bmap.cache = ResultCache(DATA_DIR / "cache.db")
        _bmap.rate_limiter = RateLimiter(max_rpm=50)
    return _bmap


def _persist() -> None:
    if _bmap is not None:
        save_map(_bmap, DATA_DIR)


def _handle_error(exc: Exception) -> None:
    """Print a user-friendly error message and exit."""
    if isinstance(exc, KeyError):
        typer.echo(f"Error: {exc}", err=True)
    elif isinstance(exc, ValueError):
        typer.echo(f"Error: {exc}", err=True)
    elif "AuthenticationError" in type(exc).__name__:
        typer.echo(
            "Error: API authentication failed. Check your API key "
            "(ANTHROPIC_API_KEY or OPENAI_API_KEY).",
            err=True,
        )
    elif "APIConnectionError" in type(exc).__name__ or "ConnectionError" in type(exc).__name__:
        typer.echo(
            "Error: Could not connect to the API. Check your network connection.",
            err=True,
        )
    else:
        typer.echo(f"Error: {type(exc).__name__}: {exc}", err=True)
    raise typer.Exit(code=1)


def _check_anthropic_key() -> None:
    """Validate that ANTHROPIC_API_KEY is set before starting LLM analysis."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        typer.echo(
            "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Set it with: export ANTHROPIC_API_KEY=sk-ant-...",
            err=True,
        )
        raise typer.Exit(code=1)


def _maybe_open(path: pathlib.Path, should_open: bool) -> None:
    """Open an HTML file in the default browser if requested."""
    if should_open:
        webbrowser.open(path.resolve().as_uri())


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------


@app.command()
def add(
    text: str = typer.Argument(..., help="Raw belief statement"),
    expanded: str = typer.Option("", "--expanded", "-e", help="Expanded definition"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated thematic tags"),
) -> None:
    """Add a new belief to the map."""
    try:
        bmap = _get_map()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        belief = bmap.add_belief(text, expanded=expanded, tags=tag_list)
        _persist()
        typer.echo(f"Added belief {belief.id}: {belief.text}")
    except (KeyError, ValueError) as exc:
        _handle_error(exc)


@app.command("list")
def list_beliefs(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List all beliefs."""
    bmap = _get_map()
    beliefs = bmap.list_beliefs()
    if not beliefs:
        typer.echo("No beliefs yet. Use 'add' to create one.")
        return
    for b in beliefs:
        typer.echo(format_belief(b, verbose=verbose))


@app.command()
def remove(belief_id: int = typer.Argument(..., help="ID of the belief to remove")) -> None:
    """Remove a belief by ID."""
    try:
        bmap = _get_map()
        bmap.remove_belief(belief_id)
        _persist()
        typer.echo(f"Removed belief {belief_id}")
    except KeyError as exc:
        _handle_error(exc)


@app.command()
def score(
    id_a: int = typer.Argument(..., help="First belief ID"),
    id_b: int = typer.Argument(..., help="Second belief ID"),
    value: float = typer.Argument(..., help="Compatibility score (-1 to 1)"),
) -> None:
    """Set the compatibility score between two beliefs."""
    try:
        bmap = _get_map()
        bmap.set_score(id_a, id_b, value)
        _persist()
        typer.echo(f"Score [{id_a} <-> {id_b}] = {value}")
    except (KeyError, ValueError) as exc:
        _handle_error(exc)


@app.command()
def show_score(
    id_a: int = typer.Argument(..., help="First belief ID"),
    id_b: int = typer.Argument(..., help="Second belief ID"),
) -> None:
    """Show the compatibility score between two beliefs."""
    try:
        bmap = _get_map()
        val = bmap.get_score(id_a, id_b)
        typer.echo(f"Score [{id_a} <-> {id_b}] = {val}")
    except KeyError as exc:
        _handle_error(exc)


@app.command()
def pairs(
    threshold: Optional[float] = typer.Option(None, "--threshold", help="Min absolute score to display"),
) -> None:
    """Show all scored belief pairs."""
    bmap = _get_map()
    all_pairs = bmap.scored_pairs()
    if threshold is not None:
        all_pairs = [(a, b, s) for a, b, s in all_pairs if abs(s) >= threshold]
    if not all_pairs:
        typer.echo("No scored pairs found.")
        return
    for a, b, s in all_pairs:
        typer.echo(f"  [{a} <-> {b}] = {s:+.3f}")


@app.command()
def embed(
    model: str = typer.Option("local-tfidf", "--model", "-m"),
) -> None:
    """Generate embeddings for all beliefs that lack one."""
    try:
        bmap = _get_map()
        count = bmap.generate_embeddings(model=model)
        _persist()
        typer.echo(f"Embedded {count} belief(s) using {model}")
    except Exception as exc:
        _handle_error(exc)


@app.command()
def similarity() -> None:
    """Compute cosine-similarity matrix from embeddings and show stats."""
    bmap = _get_map()
    bmap.calculate_initial_similarity()
    _persist()

    beliefs = [b for b in bmap.list_beliefs() if b.embedding]
    n = len(beliefs)
    n_pairs = n * (n - 1) // 2
    typer.echo(f"Computed similarity for {n} embedded beliefs ({n_pairs} pairs)")


@app.command()
def interesting(
    top_n: int = typer.Option(20, "--top", "-n", help="Max pairs to return"),
    threshold: float = typer.Option(0.7, "--threshold", help="Min cosine similarity"),
) -> None:
    """Show belief pairs worth sending to an LLM for deep comparison."""
    bmap = _get_map()
    results = bmap.interesting_pairs(top_n=top_n, threshold=threshold)
    if not results:
        typer.echo("No interesting pairs found. Lower --threshold or add tags.")
        return
    typer.echo(f"Top-{len(results)} interesting pairs (threshold={threshold}):\n")
    for a, b, sim in results:
        ba = bmap.get_belief(a)
        bb = bmap.get_belief(b)
        typer.echo(f"  [{a:>2} <-> {b:>2}]  sim={sim:+.4f}")
        typer.echo(f"      A: {ba.text}")
        typer.echo(f"      B: {bb.text}")


@app.command()
def analyze(
    id_a: int = typer.Argument(..., help="First belief ID"),
    id_b: int = typer.Argument(..., help="Second belief ID"),
    model: str = typer.Option("claude-sonnet-4-5-latest", "--model", "-m"),
) -> None:
    """Analyze logical tension between two beliefs via Claude."""
    try:
        _check_anthropic_key()
        bmap = _get_map()
        result = bmap.analyze_pair(id_a, id_b, model=model)
        _persist()
        ba = bmap.get_belief(id_a)
        bb = bmap.get_belief(id_b)
        typer.echo(f"  A: {ba.text}")
        typer.echo(f"  B: {bb.text}")
        typer.echo(f"  Category:      {result.category.value}")
        typer.echo(f"  Score:         {result.score:+.2f}")
        typer.echo(f"  Justification: {result.justification}")
    except Exception as exc:
        _handle_error(exc)


@app.command()
def analyze_all(
    top_n: int = typer.Option(20, "--top", "-n", help="Max pairs to analyze"),
    threshold: float = typer.Option(0.7, "--threshold", help="Min cosine similarity"),
    model: str = typer.Option("claude-sonnet-4-5-latest", "--model", "-m"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-analyze pairs that already have scores"),
) -> None:
    """Run tension analysis on all interesting pairs via Claude."""
    try:
        _check_anthropic_key()
        bmap = _get_map()

        # Count how many pairs will actually be analyzed
        import numpy as np
        candidates = bmap.interesting_pairs(top_n=top_n, threshold=threshold)
        to_analyze = [
            (a, b) for a, b, _ in candidates
            if force or np.isnan(bmap.scores[a, b])
        ]
        total = len(to_analyze)

        if total == 0:
            typer.echo("No interesting pairs to analyze.")
            return

        typer.echo(f"Analyzing {total} pair(s) (threshold={threshold}, top={top_n})...\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing pairs", total=total)
            results = bmap.analyze_interesting(
                top_n=top_n, threshold=threshold, model=model, force=force,
                on_progress=lambda: progress.advance(task),
                persist_fn=_persist,
            )

        _persist()
        for id_a, id_b, result in results:
            ba = bmap.get_belief(id_a)
            bb = bmap.get_belief(id_b)
            typer.echo(f"  [{id_a:>2} <-> {id_b:>2}]  {result.score:+.2f}  {result.category.value}")
            typer.echo(f"      A: {ba.text}")
            typer.echo(f"      B: {bb.text}")
            typer.echo(f"      {result.justification}\n")
        typer.echo(f"Analyzed {len(results)} pair(s). Scores saved to matrix.")
    except Exception as exc:
        _handle_error(exc)


@app.command()
def heatmap(
    output: str = typer.Option("heatmap.html", "--output", "-o"),
    matrix: str = typer.Option("scores", "--matrix", help="'scores' or 'similarity'"),
    open_browser: bool = typer.Option(False, "--open", help="Open in default browser"),
) -> None:
    """Export the compatibility matrix as an interactive HTML heatmap."""
    try:
        if matrix not in ("scores", "similarity"):
            typer.echo(f"Error: --matrix must be 'scores' or 'similarity', got '{matrix}'", err=True)
            raise typer.Exit(code=1)
        bmap = _get_map()
        path = export_heatmap(bmap, output=output, matrix=matrix)
        typer.echo(f"Heatmap written to {path}")
        _maybe_open(path, open_browser)
    except Exception as exc:
        _handle_error(exc)


@app.command()
def network(
    output: str = typer.Option("network.html", "--output", "-o"),
    threshold: float = typer.Option(0.5, "--threshold", help="Min |score| to draw an edge"),
    open_browser: bool = typer.Option(False, "--open", help="Open in default browser"),
) -> None:
    """Export a force-directed network graph as interactive HTML."""
    try:
        bmap = _get_map()
        path = export_network(bmap, output=output, edge_threshold=threshold)
        typer.echo(f"Network graph written to {path}")
        _maybe_open(path, open_browser)
    except Exception as exc:
        _handle_error(exc)


@app.command()
def analyze_map(
    beliefs_file: str = typer.Argument(..., help="Text file with one belief per line"),
    output: str = typer.Option("belief_map", "--output", "-o", help="Output prefix for HTML files"),
    threshold: float = typer.Option(0.7, "--threshold", help="Min cosine similarity for interesting pairs"),
    top_n: int = typer.Option(50, "--top", "-n", help="Max interesting pairs to analyze"),
    model: str = typer.Option("claude-sonnet-4-5-latest", "--model", "-m"),
    embed_model: str = typer.Option("local-tfidf", "--embed-model", help="Embedding model ('local-tfidf', 'local-bow', or OpenAI model name)"),
    open_browser: bool = typer.Option(False, "--open", help="Open visualizations in default browser"),
) -> None:
    """End-to-end pipeline: load beliefs from a text file and output HTML visualizations.

    Reads one belief per line (blank lines and '#' comments are skipped).
    Runs the full pipeline: add -> embed -> similarity -> analyze -> visualize.
    Only new beliefs and unscored pairs are processed; cached results are reused.
    """
    try:
        src = pathlib.Path(beliefs_file)
        if not src.exists():
            typer.echo(f"Error: File not found: {src}", err=True)
            raise typer.Exit(code=1)

        lines = [
            ln.strip()
            for ln in src.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if not lines:
            typer.echo("Error: No beliefs found in file.", err=True)
            raise typer.Exit(code=1)
        if len(lines) > 50:
            typer.echo(f"File has {len(lines)} beliefs; max is 50. Truncating.")
            lines = lines[:50]

        # Validate API key before starting the pipeline (if using LLM model)
        if not model.startswith("local"):
            _check_anthropic_key()

        bmap = _get_map()

        # --- Step 1: add beliefs (skip duplicates) -------------------
        existing_texts = {b.text for b in bmap.list_beliefs()}
        added = 0
        for text in lines:
            if text not in existing_texts:
                bmap.add_belief(text)
                existing_texts.add(text)
                added += 1
        typer.echo(f"[1/5] Beliefs: {added} new, {len(bmap.list_beliefs())} total")

        # --- Step 2: embed -------------------------------------------
        count = bmap.generate_embeddings(model=embed_model)
        typer.echo(f"[2/5] Embeddings: {count} generated ({embed_model})")

        # --- Step 3: cosine similarity -------------------------------
        bmap.calculate_initial_similarity()
        typer.echo("[3/5] Similarity matrix computed")

        # --- Step 4: interesting pairs + LLM analysis ----------------
        import numpy as np
        candidates = bmap.interesting_pairs(top_n=top_n, threshold=threshold)
        to_analyze = [
            (a, b) for a, b, _ in candidates
            if np.isnan(bmap.scores[a, b])
        ]
        total = len(to_analyze)

        if total > 0:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("[4/5] Analyzing tensions", total=total)
                results = bmap.analyze_interesting(
                    top_n=top_n, threshold=threshold, model=model,
                    on_progress=lambda: progress.advance(task),
                    persist_fn=_persist,
                )
            typer.echo(f"[4/5] Tension analysis: {len(results)} pair(s) scored via LLM")
        else:
            typer.echo("[4/5] Tension analysis: all pairs already scored (cached)")
        _persist()

        # --- Step 5: visualize ---------------------------------------
        n_beliefs = len(bmap.list_beliefs())
        hm_path = net_path = None
        if n_beliefs >= 2:
            hm_path = export_heatmap(bmap, output=f"{output}_heatmap.html")
            typer.echo(f"[5/5] Heatmap  -> {hm_path}")
        if n_beliefs >= 1:
            net_path = export_network(bmap, output=f"{output}_network.html")
            typer.echo(f"[5/5] Network  -> {net_path}")

        if open_browser:
            if hm_path:
                _maybe_open(hm_path, True)
            if net_path:
                _maybe_open(net_path, True)

        typer.echo("Done.")
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc)


PROFILES_DIR = pathlib.Path(__file__).parent / "profiles"


@app.command()
def demo(
    profile: str = typer.Argument(
        "", help="Profile name (e.g. 'mixed_debate'). Leave empty with --list to see options."
    ),
    list_profiles: bool = typer.Option(False, "--list", "-l", help="List available profiles"),
    output: str = typer.Option("", "--output", "-o", help="Output prefix for HTML files (defaults to profile name)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full justification for each pair"),
    open_browser: bool = typer.Option(False, "--open", help="Open visualizations in default browser"),
) -> None:
    """Load a predefined belief profile with pre-scored compatibility and generate visualizations instantly (no API calls)."""
    import json

    available = sorted(p.stem for p in PROFILES_DIR.glob("*.json"))

    if list_profiles or not profile:
        if not available:
            typer.echo("No profiles found in profiles/ directory.")
            raise typer.Exit(code=1)
        typer.echo("Available profiles:\n")
        for name in available:
            data = json.loads((PROFILES_DIR / f"{name}.json").read_text(encoding="utf-8"))
            typer.echo(f"  {name:30s} {data.get('description', '')[:70]}")
        typer.echo(f"\nUsage: python main.py demo <profile_name>")
        return

    profile_path = PROFILES_DIR / f"{profile}.json"
    if not profile_path.exists():
        typer.echo(f"Profile '{profile}' not found. Available: {', '.join(available)}")
        raise typer.Exit(code=1)

    data = json.loads(profile_path.read_text(encoding="utf-8"))
    beliefs_list = data["beliefs"]
    score_matrix = data["scores"]
    justifications = data.get("justifications", {})
    prefix = output or profile

    typer.echo(f"Profile: {data.get('name', profile)}")
    typer.echo(f"  {data.get('description', '')}\n")

    # Build the BeliefMap
    bmap = BeliefMap()
    for text in beliefs_list:
        bmap.add_belief(text)
    typer.echo(f"Loaded {len(beliefs_list)} beliefs")

    # Load pre-computed scores into the matrix
    n = len(beliefs_list)
    scored = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = score_matrix[i][j]
            bmap.set_score(i, j, s)
            scored += 1
    typer.echo(f"Loaded {scored} pre-scored pairs")

    # Generate local embeddings for the similarity matrix / heatmap labels
    bmap.generate_embeddings(model="local-tfidf")
    bmap.calculate_initial_similarity()

    # Print scored pairs summary
    typer.echo(f"\n{'='*60}")
    typer.echo("Scored pairs:")
    typer.echo(f"{'='*60}")
    for id_a, id_b, sc in bmap.scored_pairs():
        ba = bmap.get_belief(id_a)
        bb = bmap.get_belief(id_b)
        label = _score_label(sc)
        typer.echo(f"  [{id_a:>2} <-> {id_b:>2}] {sc:+.2f}  {label:22s} {ba.text[:30]:30s} | {bb.text[:30]}")
        if verbose:
            key = f"{id_a}-{id_b}"
            justification = justifications.get(key, "")
            if justification:
                typer.echo(f"      Justification: {justification}")

    # Visualize
    hm = export_heatmap(bmap, output=f"{prefix}_heatmap.html")
    net = export_network(bmap, output=f"{prefix}_network.html", edge_threshold=0.3)
    typer.echo(f"\nHeatmap  -> {hm}")
    typer.echo(f"Network  -> {net}")
    if open_browser:
        _maybe_open(hm, True)
        _maybe_open(net, True)
    typer.echo("Done.")


def _score_label(score: float) -> str:
    """Human-readable label for a compatibility score."""
    if score >= 0.5:
        return "compatible/harmonious"
    elif score >= 0.0:
        return "neutral"
    elif score >= -0.5:
        return "tensioned"
    else:
        return "contradictory"


if __name__ == "__main__":
    app()
