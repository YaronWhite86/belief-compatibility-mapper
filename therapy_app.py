"""Therapy-focused Streamlit app for the Belief Compatibility Mapper.

Entry point: streamlit run therapy_app.py

Patient data is stored under therapy_data/<pseudonym>/ (gitignored).
This app shares the same engine, models, and visualization code as the
general tool but uses patient-safe language and clinical disclaimers.
"""

from __future__ import annotations

import os
import pathlib

import streamlit as st

from cache import RateLimiter, ResultCache
from engine import BeliefMap
from models import BeliefRole
from utils import load_map, save_map, save_snapshot, list_snapshots, load_snapshot
from visualization import (
    build_heatmap_figure,
    build_network_figure,
    build_delta_heatmap_figure,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THERAPY_DATA_DIR = pathlib.Path("./therapy_data")

ROLE_COLORS: dict[BeliefRole, str] = {
    BeliefRole.SELF_SCHEMA:     "#E74C3C",  # red
    BeliefRole.ASPIRATION:      "#2ECC71",  # green
    BeliefRole.SAFETY_STRATEGY: "#F39C12",  # amber
    BeliefRole.LIMITING_BELIEF: "#9B59B6",  # purple
    BeliefRole.BRIDGE_BELIEF:   "#1ABC9C",  # teal
    BeliefRole.CORE_VALUE:      "#3498DB",  # blue
    BeliefRole.UNTAGGED:        "#95A5A6",  # grey
}

ROLE_LABELS: dict[BeliefRole, str] = {
    BeliefRole.SELF_SCHEMA:     "Self-Schema",
    BeliefRole.ASPIRATION:      "Aspiration",
    BeliefRole.SAFETY_STRATEGY: "Safety Strategy",
    BeliefRole.LIMITING_BELIEF: "Limiting Belief",
    BeliefRole.BRIDGE_BELIEF:   "Bridge Belief",
    BeliefRole.CORE_VALUE:      "Core Value",
    BeliefRole.UNTAGGED:        "Untagged",
}

SEVERITY_LABELS = {
    "high":     "High Structural Tension",
    "moderate": "Moderate Structural Tension",
    "low":      "Low Structural Tension",
}


# ---------------------------------------------------------------------------
# Patient helpers
# ---------------------------------------------------------------------------


def list_patients() -> list[str]:
    if not THERAPY_DATA_DIR.exists():
        return []
    return sorted(p.name for p in THERAPY_DATA_DIR.iterdir() if p.is_dir())


def _get_patient_dir(pseudonym: str) -> pathlib.Path:
    return THERAPY_DATA_DIR / pseudonym


def _get_engine(pseudonym: str) -> BeliefMap:
    patient_dir = _get_patient_dir(pseudonym)
    patient_dir.mkdir(parents=True, exist_ok=True)
    if (patient_dir / "beliefs.json").exists():
        bmap = load_map(patient_dir)
    else:
        bmap = BeliefMap()
    bmap.cache = ResultCache(patient_dir / "cache.db")
    bmap.rate_limiter = RateLimiter(max_rpm=50)
    return bmap


def _save_current(pseudonym: str, bmap: BeliefMap) -> None:
    save_map(bmap, _get_patient_dir(pseudonym))


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _role_chip(role: BeliefRole) -> str:
    color = ROLE_COLORS[role]
    label = ROLE_LABELS[role]
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:12px;font-size:0.78rem;">{label}</span>'
    )


def _severity_label(score: float) -> str:
    abs_score = abs(score)
    if abs_score >= 0.7:
        return SEVERITY_LABELS["high"]
    if abs_score >= 0.4:
        return SEVERITY_LABELS["moderate"]
    return SEVERITY_LABELS["low"]


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Belief Mapper — Therapy Tool",
    layout="wide",
)

# Clinical disclaimer — always shown
st.warning(
    "⚕️ **Clinical Use Notice** — This tool is a structural analysis aid for "
    "trained therapists. It is **not** a diagnostic instrument. All clinical "
    "interpretations are the clinician's responsibility. When running analysis, "
    "belief text is sent to the Anthropic API — use pseudonyms only, never real names."
)

st.title("Belief Compatibility Mapper — Therapy Tool")

# ---------------------------------------------------------------------------
# Sidebar — patient management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Patient")

    patients = list_patients()
    patient_options = ["+ New patient"] + patients
    selected = st.selectbox("Patient (pseudonym)", patient_options)

    if selected == "+ New patient":
        new_pseudonym = st.text_input("Enter pseudonym (e.g. case-001)")
        if st.button("Create patient") and new_pseudonym.strip():
            pseudonym = new_pseudonym.strip()
            _get_patient_dir(pseudonym).mkdir(parents=True, exist_ok=True)
            st.session_state["pseudonym"] = pseudonym
            st.session_state["bmap"] = _get_engine(pseudonym)
            st.rerun()
        st.stop()
    else:
        pseudonym = selected
        if st.session_state.get("pseudonym") != pseudonym:
            st.session_state["pseudonym"] = pseudonym
            st.session_state["bmap"] = _get_engine(pseudonym)

    bmap: BeliefMap = st.session_state.get("bmap")
    if bmap is None:
        st.info("Select or create a patient to begin.")
        st.stop()

    st.markdown(f"**Active patient:** `{pseudonym}`")
    st.markdown(f"Beliefs loaded: **{len(bmap.beliefs)}**")

    st.divider()

    # ------------------------------------------------------------------
    # Add belief
    # ------------------------------------------------------------------
    st.subheader("Add Belief")
    new_text = st.text_input("Belief statement")
    new_role = st.selectbox(
        "Role",
        options=list(BeliefRole),
        format_func=lambda r: ROLE_LABELS[r],
    )
    new_tags_raw = st.text_input("Tags (comma-separated, optional)")
    if st.button("Add belief") and new_text.strip():
        tags = [t.strip() for t in new_tags_raw.split(",") if t.strip()]
        try:
            bmap.add_belief(new_text.strip(), tags=tags, role=new_role)
            _save_current(pseudonym, bmap)
            st.success("Belief added.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.divider()

    # ------------------------------------------------------------------
    # Run analysis
    # ------------------------------------------------------------------
    st.subheader("Analysis")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning("Set ANTHROPIC_API_KEY to enable LLM analysis.")

    if st.button("Run Analysis", disabled=not api_key):
        if len(bmap.beliefs) < 2:
            st.error("Add at least 2 beliefs before running analysis.")
        else:
            with st.spinner("Embedding beliefs…"):
                bmap.generate_embeddings(model="local-tfidf")
                bmap.calculate_initial_similarity()

            pairs = bmap.interesting_pairs()
            to_do = [(a, b) for a, b, _ in pairs if __import__("numpy").isnan(bmap.scores[a, b])]
            if to_do:
                bar = st.progress(0, text="Analysing pairs…")
                done = [0]

                def _on_progress():
                    done[0] += 1
                    bar.progress(min(done[0] / max(len(to_do), 1), 1.0))

                bmap.analyze_interesting(on_progress=_on_progress)
                _save_current(pseudonym, bmap)
                st.success(f"Analysed {done[0]} pair(s).")
            else:
                st.info("All interesting pairs already scored.")

    st.divider()

    # ------------------------------------------------------------------
    # Save snapshot
    # ------------------------------------------------------------------
    st.subheader("Save Session Snapshot")
    snap_label = st.text_input("Session label", value="Session")
    snap_notes = st.text_area("Notes", height=80)
    if st.button("Save Snapshot"):
        snap_path = save_snapshot(bmap, _get_patient_dir(pseudonym), snap_label, snap_notes)
        st.success(f"Snapshot saved: `{snap_path.name}`")

    st.divider()

    if st.button("Reset session state"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

beliefs = bmap.list_beliefs()

tab_map, tab_tensions, tab_roots, tab_stress, tab_history, tab_network = st.tabs([
    "Belief Map",
    "Structural Tensions",
    "Root Commitments",
    "Stress-Test",
    "Session History",
    "Compatibility Network",
])

# ---- Tab 1: Belief Map ---------------------------------------------------
with tab_map:
    st.subheader("Belief Map")
    st.caption(
        "The model shows these structural relationships — not judgments about "
        "which beliefs are correct."
    )

    if not beliefs:
        st.info("No beliefs added yet. Use the sidebar to add beliefs.")
    else:
        rows = []
        for b in beliefs:
            rows.append({
                "ID": b.id,
                "Belief": b.text,
                "Role": ROLE_LABELS[b.role],
                "Tags": ", ".join(b.tags) if b.tags else "—",
                "Embedded": "Yes" if b.embedding else "No",
            })

        # Render table with role color chips using HTML
        header_html = (
            "<table style='width:100%;border-collapse:collapse'>"
            "<tr style='background:#f0f0f0'>"
            "<th style='padding:6px;text-align:left'>ID</th>"
            "<th style='padding:6px;text-align:left'>Belief</th>"
            "<th style='padding:6px;text-align:left'>Role</th>"
            "<th style='padding:6px;text-align:left'>Tags</th>"
            "<th style='padding:6px;text-align:left'>Embedded</th>"
            "</tr>"
        )
        body_rows = []
        for b in beliefs:
            body_rows.append(
                f"<tr>"
                f"<td style='padding:6px'>{b.id}</td>"
                f"<td style='padding:6px'>{b.text}</td>"
                f"<td style='padding:6px'>{_role_chip(b.role)}</td>"
                f"<td style='padding:6px'>{', '.join(b.tags) if b.tags else '—'}</td>"
                f"<td style='padding:6px'>{'Yes' if b.embedding else 'No'}</td>"
                f"</tr>"
            )
        st.markdown(header_html + "".join(body_rows) + "</table>", unsafe_allow_html=True)

        st.divider()

        # Heatmap
        import numpy as np
        has_scores = any(
            not np.isnan(bmap.scores[a.id, b.id])
            for i, a in enumerate(beliefs)
            for b in beliefs[i + 1:]
        )
        if has_scores and len(beliefs) >= 2:
            st.subheader("Compatibility Heatmap")
            fig = build_heatmap_figure(beliefs, bmap.scores)
            st.plotly_chart(fig, use_container_width=True)

# ---- Tab 2: Structural Tensions ------------------------------------------
with tab_tensions:
    st.subheader("Structural Tensions")
    st.caption(
        "The model shows these structural relationships — not judgments about "
        "which beliefs are correct."
    )

    alerts = bmap.dissonance_report()
    if not alerts:
        st.info("No structural tensions detected. Run analysis or add more beliefs.")
    else:
        for alert in alerts:
            sev_label = _severity_label(alert.score)
            b_a = bmap.beliefs.get(alert.belief_id_a)
            b_b = bmap.beliefs.get(alert.belief_id_b)
            if not b_a or not b_b:
                continue

            with st.expander(
                f"{sev_label} (score: {alert.score:+.2f}) — "
                f"[{alert.belief_id_a}] vs [{alert.belief_id_b}]"
            ):
                st.write(f"**Belief {alert.belief_id_a}:** {b_a.text}")
                st.write(f"**Belief {alert.belief_id_b}:** {b_b.text}")
                st.write(f"**Structural tension score:** {alert.score:+.3f}")
                if alert.dependent_ids:
                    dep_texts = [
                        f"[{d}] {bmap.beliefs[d].text}"
                        for d in alert.dependent_ids
                        if d in bmap.beliefs
                    ]
                    st.write("**Potentially affected beliefs:**")
                    for dt in dep_texts:
                        st.write(f"  • {dt}")

# ---- Tab 3: Root Commitments ---------------------------------------------
with tab_roots:
    st.subheader("Root Commitments")
    st.caption(
        "Root commitments are the implicit foundational values that appear to unify "
        "two or more surface-level beliefs. They are inferred from structural patterns, "
        "not from the content of specific beliefs."
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning("Set ANTHROPIC_API_KEY to identify root commitments.")
    elif len(beliefs) < 2:
        st.info("Add at least 2 beliefs to identify root commitments.")
    else:
        if st.button("Identify Root Commitments"):
            with st.spinner("Analysing…"):
                try:
                    principles = bmap.identify_bedrock_principles()
                    st.session_state["principles"] = principles
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

        principles = st.session_state.get("principles", [])
        if principles:
            for p in principles:
                belief_texts = [
                    f"[{bid}] {bmap.beliefs[bid].text}"
                    for bid in p.belief_ids
                    if bid in bmap.beliefs
                ]
                with st.expander(
                    f"{p.principle} (coherence: {p.coherence:.2f})"
                ):
                    st.write(f"**Explanation:** {p.explanation}")
                    st.write("**Related beliefs:**")
                    for bt in belief_texts:
                        st.write(f"  • {bt}")

# ---- Tab 4: Stress-Test --------------------------------------------------
with tab_stress:
    st.subheader("Stress-Test Simulation")
    st.caption(
        "Virtually remove a belief to see which other beliefs may become structurally "
        "isolated or destabilized. This does **not** modify the belief map."
    )

    if len(beliefs) < 2:
        st.info("Add at least 2 beliefs and run analysis to use the stress-test.")
    else:
        belief_options = {f"[{b.id}] {b.text}": b.id for b in beliefs}
        selected_label = st.selectbox("Select belief to remove (virtually)", list(belief_options))
        selected_id = belief_options[selected_label]

        threshold = st.slider(
            "Edge threshold (|score| ≥ threshold counts as a connection)",
            min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        )

        if st.button("Run Simulation"):
            try:
                result = bmap.simulate_removal(selected_id, stability_threshold=threshold)
                st.session_state["sim_result"] = result
            except Exception as exc:
                st.error(str(exc))

        sim = st.session_state.get("sim_result")
        if sim and sim.removed_id == selected_id:
            st.markdown(f"**Removed:** [{sim.removed_id}] {sim.removed_text}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### Stable")
                st.caption("Still well-connected after removal")
                for bid in sim.stable_ids:
                    b = bmap.beliefs.get(bid)
                    if b:
                        st.markdown(
                            f'<span style="color:#27ae60">✓ [{bid}] {b.text}</span>',
                            unsafe_allow_html=True,
                        )
                if not sim.stable_ids:
                    st.write("—")

            with col2:
                st.markdown("#### Destabilized")
                st.caption("Lost some connections due to removal")
                for bid in sim.destabilized_ids:
                    b = bmap.beliefs.get(bid)
                    if b:
                        st.markdown(
                            f'<span style="color:#e67e22">⚠ [{bid}] {b.text}</span>',
                            unsafe_allow_html=True,
                        )
                if not sim.destabilized_ids:
                    st.write("—")

            with col3:
                st.markdown("#### Orphaned")
                st.caption("Now structurally isolated (no connections)")
                for bid in sim.orphaned_ids:
                    b = bmap.beliefs.get(bid)
                    if b:
                        st.markdown(
                            f'<span style="color:#e74c3c">✗ [{bid}] {b.text}</span>',
                            unsafe_allow_html=True,
                        )
                if not sim.orphaned_ids:
                    st.write("—")

            # Network graph with removed node greyed out
            import numpy as np
            has_scores = any(
                not np.isnan(bmap.scores[a.id, b2.id])
                for i, a in enumerate(beliefs)
                for b2 in beliefs[i + 1:]
            )
            if has_scores:
                node_overrides = {}
                for b in beliefs:
                    if b.id == selected_id:
                        node_overrides[b.id] = {
                            "size": 20,
                            "line_color": "#aaaaaa",
                            "line_width": 3,
                        }
                    else:
                        node_overrides[b.id] = {
                            "size": 20,
                            "line_color": "#333333",
                            "line_width": 1,
                        }
                try:
                    fig = build_network_figure(
                        beliefs, bmap.beliefs, bmap.scores,
                        title="Stress-Test Network (greyed node = removed)",
                        node_overrides=node_overrides,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ValueError:
                    pass  # Not enough data for graph yet

# ---- Tab 5: Session History ----------------------------------------------
with tab_history:
    st.subheader("Session History")

    patient_dir = _get_patient_dir(pseudonym)
    snaps = list_snapshots(patient_dir)

    if not snaps:
        st.info("No snapshots saved yet. Use the sidebar to save session snapshots.")
    else:
        # Display snapshot table
        snap_rows = [
            {
                "Label": s["label"],
                "Notes": s.get("notes", ""),
                "Created": s["created_at"][:19].replace("T", " "),
                "Beliefs": s["belief_count"],
            }
            for s in snaps
        ]
        st.dataframe(snap_rows, use_container_width=True)

        st.divider()
        st.subheader("Compare Two Snapshots")

        snap_labels = [f"{s['label']} ({s['created_at'][:10]})" for s in snaps]
        col_a, col_b = st.columns(2)
        with col_a:
            idx_a = st.selectbox("Earlier snapshot", range(len(snaps)),
                                 format_func=lambda i: snap_labels[i], key="snap_a")
        with col_b:
            idx_b = st.selectbox("Later snapshot", range(len(snaps)),
                                 format_func=lambda i: snap_labels[i],
                                 index=min(1, len(snaps) - 1), key="snap_b")

        if st.button("Compare snapshots"):
            if idx_a == idx_b:
                st.warning("Please select two different snapshots.")
            else:
                snap_a = snaps[idx_a]
                snap_b = snaps[idx_b]
                bmap_a = load_snapshot(snap_a["path"])
                bmap_b = load_snapshot(snap_b["path"])
                try:
                    fig = build_delta_heatmap_figure(
                        bmap_a.list_beliefs(), bmap_b.list_beliefs(),
                        bmap_a.scores, bmap_b.scores,
                        title=f"Score Delta: '{snap_b['label']}' − '{snap_a['label']}'",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "Blue = beliefs became more compatible. "
                        "Red = beliefs became more contradictory. "
                        "Grey = no change or no shared scored pair."
                    )
                except ValueError as exc:
                    st.warning(str(exc))

# ---- Tab 6: Compatibility Network ----------------------------------------
with tab_network:
    st.subheader("Compatibility Network")
    st.caption(
        "Node color reflects the average compatibility score with connected beliefs. "
        "Node border color reflects the belief's therapeutic role."
    )

    import numpy as np
    has_scores = any(
        not np.isnan(bmap.scores[a.id, b2.id])
        for i, a in enumerate(beliefs)
        for b2 in beliefs[i + 1:]
    )

    if not beliefs:
        st.info("No beliefs to display.")
    elif not has_scores:
        st.info("Run analysis first to compute compatibility scores.")
    else:
        # Build node overrides for role-based border colours
        node_overrides = {}
        for b in beliefs:
            node_overrides[b.id] = {
                "size": 20,
                "line_color": ROLE_COLORS.get(b.role, "#333333"),
                "line_width": 3,
            }

        edge_threshold = st.slider(
            "Edge threshold",
            min_value=0.1, max_value=1.0, value=0.5, step=0.05,
            key="network_threshold",
        )

        try:
            fig = build_network_figure(
                beliefs, bmap.beliefs, bmap.scores,
                edge_threshold=edge_threshold,
                title="Belief Compatibility Network (role-coloured borders)",
                node_overrides=node_overrides,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ValueError as exc:
            st.warning(str(exc))

        # Role legend
        st.subheader("Role Legend")
        legend_html = "<div style='display:flex;flex-wrap:wrap;gap:8px'>"
        for role, color in ROLE_COLORS.items():
            legend_html += (
                f'<span style="background:{color};color:#fff;padding:4px 12px;'
                f'border-radius:12px;font-size:0.85rem;">{ROLE_LABELS[role]}</span>'
            )
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)
