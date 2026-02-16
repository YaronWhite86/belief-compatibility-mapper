"""CLI entry-point for the Belief Compatibility Mapper (Typer-based)."""

from __future__ import annotations

import pathlib
from typing import Optional

import typer

from engine import BeliefMap
from utils import format_belief, load_map, save_map

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
) -> None:
    """Run tension analysis on all interesting pairs via Claude."""
    bmap = _get_map()
    typer.echo(f"Analyzing interesting pairs (threshold={threshold}, top={top_n})...\n")
    results = bmap.analyze_interesting(top_n=top_n, threshold=threshold, model=model)
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


if __name__ == "__main__":
    app()
