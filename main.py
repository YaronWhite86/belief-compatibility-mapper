"""CLI entry-point for the Belief Compatibility Mapper (Typer-based)."""

from __future__ import annotations

import pathlib
from typing import Optional

import typer

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


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------


@app.command()
def add(
    text: str = typer.Argument(..., help="Raw belief statement"),
    expanded: str = typer.Option("", "--expanded", "-e", help="Expanded definition"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated thematic tags"),
) -> None:
    """Add a new belief to the map."""
    bmap = _get_map()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    belief = bmap.add_belief(text, expanded=expanded)
    belief.tags = tag_list
    _persist()
    typer.echo(f"Added belief {belief.id}: {belief.text}")


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
    bmap = _get_map()
    bmap.remove_belief(belief_id)
    _persist()
    typer.echo(f"Removed belief {belief_id}")


@app.command()
def score(
    id_a: int = typer.Argument(..., help="First belief ID"),
    id_b: int = typer.Argument(..., help="Second belief ID"),
    value: float = typer.Argument(..., help="Compatibility score (-1 to 1)"),
) -> None:
    """Set the compatibility score between two beliefs."""
    bmap = _get_map()
    bmap.set_score(id_a, id_b, value)
    _persist()
    typer.echo(f"Score [{id_a} <-> {id_b}] = {value}")


@app.command()
def show_score(
    id_a: int = typer.Argument(..., help="First belief ID"),
    id_b: int = typer.Argument(..., help="Second belief ID"),
) -> None:
    """Show the compatibility score between two beliefs."""
    bmap = _get_map()
    val = bmap.get_score(id_a, id_b)
    typer.echo(f"Score [{id_a} <-> {id_b}] = {val}")


@app.command()
def pairs(
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Min absolute score to display"),
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
    model: str = typer.Option("text-embedding-3-small", "--model", "-m"),
) -> None:
    """Generate OpenAI embeddings for all beliefs that lack one."""
    bmap = _get_map()
    count = bmap.generate_embeddings(model=model)
    _persist()
    typer.echo(f"Embedded {count} belief(s) using {model}")


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
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Min cosine similarity"),
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
    model: str = typer.Option("claude-sonnet-4-5-20250929", "--model", "-m"),
) -> None:
    """Analyze logical tension between two beliefs via Claude."""
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


@app.command()
def analyze_all(
    top_n: int = typer.Option(20, "--top", "-n", help="Max pairs to analyze"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Min cosine similarity"),
    model: str = typer.Option("claude-sonnet-4-5-20250929", "--model", "-m"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-analyze pairs that already have scores"),
) -> None:
    """Run tension analysis on all interesting pairs via Claude."""
    bmap = _get_map()
    typer.echo(f"Analyzing interesting pairs (threshold={threshold}, top={top_n})...\n")
    results = bmap.analyze_interesting(
        top_n=top_n, threshold=threshold, model=model, force=force,
    )
    _persist()
    if not results:
        typer.echo("No interesting pairs to analyze.")
        return
    for id_a, id_b, result in results:
        ba = bmap.get_belief(id_a)
        bb = bmap.get_belief(id_b)
        typer.echo(f"  [{id_a:>2} <-> {id_b:>2}]  {result.score:+.2f}  {result.category.value}")
        typer.echo(f"      A: {ba.text}")
        typer.echo(f"      B: {bb.text}")
        typer.echo(f"      {result.justification}\n")
    typer.echo(f"Analyzed {len(results)} pair(s). Scores saved to matrix.")


@app.command()
def heatmap(
    output: str = typer.Option("heatmap.html", "--output", "-o"),
    matrix: str = typer.Option("scores", "--matrix", help="'scores' or 'similarity'"),
) -> None:
    """Export the compatibility matrix as an interactive HTML heatmap."""
    bmap = _get_map()
    path = export_heatmap(bmap, output=output, matrix=matrix)
    typer.echo(f"Heatmap written to {path}")


@app.command()
def network(
    output: str = typer.Option("network.html", "--output", "-o"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Min |score| to draw an edge"),
) -> None:
    """Export a force-directed network graph as interactive HTML."""
    bmap = _get_map()
    path = export_network(bmap, output=output, edge_threshold=threshold)
    typer.echo(f"Network graph written to {path}")


@app.command()
def analyze_map(
    beliefs_file: str = typer.Argument(..., help="Text file with one belief per line"),
    output: str = typer.Option("belief_map", "--output", "-o", help="Output prefix for HTML files"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Min cosine similarity for interesting pairs"),
    top_n: int = typer.Option(50, "--top", "-n", help="Max interesting pairs to analyze"),
    model: str = typer.Option("claude-sonnet-4-5-20250929", "--model", "-m"),
) -> None:
    """End-to-end pipeline: load beliefs from a text file and output HTML visualizations.

    Reads one belief per line (blank lines and '#' comments are skipped).
    Runs the full pipeline: add -> embed -> similarity -> analyze -> visualize.
    Only new beliefs and unscored pairs are processed; cached results are reused.
    """
    import pathlib as _pathlib

    src = _pathlib.Path(beliefs_file)
    if not src.exists():
        typer.echo(f"File not found: {src}")
        raise typer.Exit(code=1)

    lines = [
        ln.strip()
        for ln in src.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        typer.echo("No beliefs found in file.")
        raise typer.Exit(code=1)
    if len(lines) > 50:
        typer.echo(f"File has {len(lines)} beliefs; max is 50. Truncating.")
        lines = lines[:50]

    bmap = _get_map()

    # --- Step 1: add beliefs (skip duplicates) -----------------------
    existing_texts = {b.text for b in bmap.list_beliefs()}
    added = 0
    for text in lines:
        if text not in existing_texts:
            bmap.add_belief(text)
            existing_texts.add(text)
            added += 1
    typer.echo(f"Beliefs: {added} new, {len(bmap.list_beliefs())} total")

    # --- Step 2: embed -----------------------------------------------
    count = bmap.generate_embeddings()
    typer.echo(f"Embeddings: {count} generated (cache + API)")

    # --- Step 3: cosine similarity -----------------------------------
    bmap.calculate_initial_similarity()
    typer.echo("Similarity matrix computed")

    # --- Step 4: interesting pairs + LLM analysis --------------------
    results = bmap.analyze_interesting(
        top_n=top_n, threshold=threshold, model=model,
    )
    typer.echo(f"Tension analysis: {len(results)} new pair(s) scored via LLM")
    _persist()

    # --- Step 5: visualize -------------------------------------------
    n_beliefs = len(bmap.list_beliefs())
    if n_beliefs >= 2:
        hm = export_heatmap(bmap, output=f"{output}_heatmap.html")
        typer.echo(f"Heatmap  -> {hm}")
    if n_beliefs >= 1:
        net = export_network(bmap, output=f"{output}_network.html")
        typer.echo(f"Network  -> {net}")

    typer.echo("Done.")


if __name__ == "__main__":
    app()
