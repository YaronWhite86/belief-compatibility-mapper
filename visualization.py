"""Visualization exports: interactive heatmap and force-directed network graph."""

from __future__ import annotations

import pathlib

import numpy as np

from engine import BeliefMap


# ------------------------------------------------------------------
# 1. Interactive HTML heatmap (Plotly)
# ------------------------------------------------------------------


def build_heatmap_figure(
    beliefs: list,
    scores: np.ndarray,
    title: str = "Belief Compatibility Matrix",
):
    """Build a Plotly heatmap figure from beliefs and a score matrix.

    Parameters
    ----------
    beliefs : list of Belief objects (must have .id and .text)
    scores : 2-D numpy array indexed by belief ID
    title : heading shown above the heatmap

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    if len(beliefs) < 2:
        raise ValueError("Need at least 2 beliefs to draw a heatmap")

    ids = [b.id for b in beliefs]
    labels = [f"[{b.id}] {b.text[:50]}" for b in beliefs]

    sub = scores[np.ix_(ids, ids)].copy()

    # Build hover text with full belief names and score.
    hover: list[list[str]] = []
    for i, row_b in enumerate(beliefs):
        row_hover: list[str] = []
        for j, col_b in enumerate(beliefs):
            val = sub[i, j]
            score_str = f"{val:+.3f}" if not np.isnan(val) else "n/a"
            row_hover.append(
                f"<b>{row_b.text}</b><br>vs.<br><b>{col_b.text}</b>"
                f"<br>Score: {score_str}"
            )
        hover.append(row_hover)

    # Replace NaN with None so Plotly renders it as a gap.
    z_clean = np.where(np.isnan(sub), None, sub)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_clean.tolist(),
            x=labels,
            y=labels,
            hovertext=hover,
            hoverinfo="text",
            colorscale=[
                [0.0, "#d73027"],   # -1  contradictory (red)
                [0.25, "#fc8d59"],  # -0.5
                [0.5, "#f7f7f7"],   #  0  neutral (white)
                [0.75, "#91bfdb"],  # +0.5
                [1.0, "#4575b4"],   # +1  entailed (blue)
            ],
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title="Score",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["Contradictory", "Tensioned", "Neutral", "Harmonious", "Entailed"],
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-45, side="bottom"),
        yaxis=dict(autorange="reversed"),
        width=max(600, 80 * len(beliefs)),
        height=max(600, 80 * len(beliefs)),
        template="plotly_white",
    )

    return fig


def export_heatmap(
    bmap: BeliefMap,
    output: str | pathlib.Path = "heatmap.html",
    matrix: str = "scores",
    title: str = "Belief Compatibility Matrix",
) -> pathlib.Path:
    """Render the score (or similarity) matrix as an interactive HTML heatmap.

    Parameters
    ----------
    bmap : BeliefMap
    output : path for the generated HTML file
    matrix : ``"scores"`` (LLM-judged) or ``"similarity"`` (cosine)
    title : heading shown above the heatmap

    Returns
    -------
    pathlib.Path to the written file.
    """
    beliefs = bmap.list_beliefs()
    source = bmap.scores if matrix == "scores" else bmap.similarity
    fig = build_heatmap_figure(beliefs, source, title=title)

    output = pathlib.Path(output)
    fig.write_html(str(output), include_plotlyjs="cdn")
    return output


# ------------------------------------------------------------------
# 2. Force-directed network graph (NetworkX + Plotly)
# ------------------------------------------------------------------


def build_network_figure(
    beliefs: list,
    beliefs_dict: dict,
    scores: np.ndarray,
    edge_threshold: float = 0.5,
    title: str = "Belief Compatibility Network",
    node_overrides: dict[int, dict] | None = None,
):
    """Build a Plotly force-directed network graph figure.

    Parameters
    ----------
    beliefs : list of Belief objects (sorted)
    beliefs_dict : dict mapping belief ID -> Belief (for text lookup)
    scores : 2-D numpy array indexed by belief ID
    edge_threshold : minimum ``|score|`` to draw an edge (default 0.5)
    title : heading shown above the graph

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import networkx as nx
    import plotly.graph_objects as go

    if not beliefs:
        raise ValueError("No beliefs to graph")

    # ---- build graph ------------------------------------------------
    G = nx.Graph()
    for b in beliefs:
        G.add_node(b.id, text=b.text)

    edges: list[tuple[int, int, float]] = []
    ids = sorted(beliefs_dict)
    for i, a in enumerate(ids):
        for b_id in ids[i + 1 :]:
            val = float(scores[a, b_id])
            if np.isnan(val):
                continue
            if abs(val) >= edge_threshold:
                edges.append((a, b_id, val))
                G.add_edge(a, b_id, weight=val)

    # ---- force-directed layout --------------------------------------
    # Build a layout graph that only contains positive edges so that
    # compatible beliefs attract each other.  Negative-score pairs have
    # no spring, so they drift apart naturally.
    L = nx.Graph()
    L.add_nodes_from(G.nodes)
    for a, b_id, val in edges:
        if val > 0:
            L.add_edge(a, b_id, weight=val)

    pos = nx.spring_layout(L, k=2.0, iterations=100, seed=42, weight="weight")

    # ---- edge traces ------------------------------------------------
    pos_edge_x, pos_edge_y = [], []
    neg_edge_x, neg_edge_y = [], []
    pos_hover, neg_hover = [], []

    for a, b_id, val in edges:
        x0, y0 = pos[a]
        x1, y1 = pos[b_id]
        target_x = pos_edge_x if val > 0 else neg_edge_x
        target_y = pos_edge_y if val > 0 else neg_edge_y
        target_x += [x0, x1, None]
        target_y += [y0, y1, None]

        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        label = (
            f"{beliefs_dict[a].text}<br>vs.<br>"
            f"{beliefs_dict[b_id].text}<br>Score: {val:+.3f}"
        )
        target_hover = pos_hover if val > 0 else neg_hover
        target_hover.append(dict(x=mid_x, y=mid_y, text=label))

    traces = []

    # Positive edges (blue)
    if pos_edge_x:
        traces.append(go.Scatter(
            x=pos_edge_x, y=pos_edge_y,
            mode="lines", line=dict(width=1.5, color="#4575b4"),
            hoverinfo="none", showlegend=True, name="Compatible",
        ))
        traces.append(go.Scatter(
            x=[h["x"] for h in pos_hover],
            y=[h["y"] for h in pos_hover],
            mode="markers", marker=dict(size=1, opacity=0),
            text=[h["text"] for h in pos_hover],
            hoverinfo="text", showlegend=False,
        ))

    # Negative edges (red)
    if neg_edge_x:
        traces.append(go.Scatter(
            x=neg_edge_x, y=neg_edge_y,
            mode="lines", line=dict(width=1.5, color="#d73027", dash="dash"),
            hoverinfo="none", showlegend=True, name="Tensioned",
        ))
        traces.append(go.Scatter(
            x=[h["x"] for h in neg_hover],
            y=[h["y"] for h in neg_hover],
            mode="markers", marker=dict(size=1, opacity=0),
            text=[h["text"] for h in neg_hover],
            hoverinfo="text", showlegend=False,
        ))

    # ---- node trace -------------------------------------------------
    node_x = [pos[n][0] for n in G.nodes]
    node_y = [pos[n][1] for n in G.nodes]

    # Color nodes by their average score (warmer = more tension overall).
    node_color: list[float] = []
    for n in G.nodes:
        neighbours = [d["weight"] for _, _, d in G.edges(n, data=True)]
        node_color.append(float(np.mean(neighbours)) if neighbours else 0.0)

    node_text = [
        f"<b>[{n}] {beliefs_dict[n].text}</b><br>"
        f"Edges: {G.degree(n)}<br>"
        f"Avg score: {c:+.2f}"
        for n, c in zip(G.nodes, node_color)
    ]

    # Build per-node marker overrides
    _overrides = node_overrides or {}
    node_sizes: list[float] = []
    node_line_colors: list[str] = []
    node_line_widths: list[float] = []
    for n in G.nodes:
        ov = _overrides.get(n, {})
        node_sizes.append(ov.get("size", 20))
        node_line_colors.append(ov.get("line_color", "#333333"))
        node_line_widths.append(ov.get("line_width", 1))

    traces.append(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_color,
            colorscale="RdBu",
            cmin=-1, cmax=1,
            line=dict(
                width=node_line_widths,
                color=node_line_colors,
            ),
            colorbar=dict(
                title="Avg Score",
                tickvals=[-1, 0, 1],
                ticktext=["Contradiction", "Neutral", "Entailed"],
            ),
        ),
        text=[f"[{n}]" for n in G.nodes],
        textposition="top center",
        textfont=dict(size=10),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=900,
        height=700,
        template="plotly_white",
    )

    return fig


def build_delta_heatmap_figure(
    beliefs_a: list,
    beliefs_b: list,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    title: str = "Score Delta (Later − Earlier)",
):
    """Heatmap showing how scores changed between two snapshots.

    Only beliefs present in BOTH snapshots (matched by text) are shown.
    Cell value = scores_b[i,j] − scores_a[i,j].
    Red = moved toward contradiction, blue = toward entailment.
    NaN where either snapshot has no score for that pair.
    """
    import plotly.graph_objects as go

    # Match beliefs by text to handle ID drift between sessions
    text_to_a = {b.text: b for b in beliefs_a}
    text_to_b = {b.text: b for b in beliefs_b}
    shared_texts = [t for t in text_to_a if t in text_to_b]

    if len(shared_texts) < 2:
        raise ValueError("Need at least 2 beliefs in common between snapshots")

    beliefs_shared_a = [text_to_a[t] for t in shared_texts]
    beliefs_shared_b = [text_to_b[t] for t in shared_texts]

    ids_a = [b.id for b in beliefs_shared_a]
    ids_b = [b.id for b in beliefs_shared_b]
    n = len(shared_texts)

    labels = [f"[{b.id}] {b.text[:50]}" for b in beliefs_shared_a]

    # Build delta sub-matrix
    delta = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            val_a = scores_a[ids_a[i], ids_a[j]]
            val_b = scores_b[ids_b[i], ids_b[j]]
            if not np.isnan(val_a) and not np.isnan(val_b):
                delta[i, j] = val_b - val_a

    # Build hover text
    hover: list[list[str]] = []
    for i, t_row in enumerate(shared_texts):
        row_hover: list[str] = []
        for j, t_col in enumerate(shared_texts):
            val = delta[i, j]
            delta_str = f"{val:+.3f}" if not np.isnan(val) else "n/a"
            row_hover.append(
                f"<b>{t_row}</b><br>vs.<br><b>{t_col}</b>"
                f"<br>Delta: {delta_str}"
            )
        hover.append(row_hover)

    z_clean = np.where(np.isnan(delta), None, delta)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_clean.tolist(),
            x=labels,
            y=labels,
            hovertext=hover,
            hoverinfo="text",
            colorscale="RdBu",
            zmin=-2,
            zmax=2,
            colorbar=dict(
                title="Score Change",
                tickvals=[-2, -1, 0, 1, 2],
                ticktext=["-2 (more contradictory)", "-1", "0 (no change)", "+1", "+2 (more aligned)"],
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-45, side="bottom"),
        yaxis=dict(autorange="reversed"),
        width=max(600, 80 * n),
        height=max(600, 80 * n),
        template="plotly_white",
    )

    return fig


def export_network(
    bmap: BeliefMap,
    output: str | pathlib.Path = "network.html",
    edge_threshold: float = 0.5,
    title: str = "Belief Compatibility Network",
) -> pathlib.Path:
    """Render beliefs as a force-directed network graph in HTML.

    Nodes are beliefs.  Edges are drawn only when ``|score| > edge_threshold``
    to keep the graph readable.  The spring layout uses **positive** edges as
    attraction springs so that compatible beliefs naturally cluster together.

    Parameters
    ----------
    bmap : BeliefMap
    output : path for the generated HTML file
    edge_threshold : minimum ``|score|`` to draw an edge (default 0.5)
    title : heading shown above the graph

    Returns
    -------
    pathlib.Path to the written file.
    """
    beliefs = bmap.list_beliefs()
    fig = build_network_figure(
        beliefs, bmap.beliefs, bmap.scores,
        edge_threshold=edge_threshold, title=title,
    )

    output = pathlib.Path(output)
    fig.write_html(str(output), include_plotlyjs="cdn")
    return output
