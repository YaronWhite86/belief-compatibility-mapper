"""Streamlit web UI for the Belief Compatibility Mapper."""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import streamlit as st

from cache import RateLimiter, ResultCache
from engine import MAX_BELIEFS, BeliefMap
from utils import load_map, save_map
from visualization import build_heatmap_figure, build_network_figure

DATA_DIR = pathlib.Path("./data")
PROFILES_DIR = pathlib.Path(__file__).parent / "profiles"
COMMON_BELIEFS_PATH = pathlib.Path(__file__).parent / "common_beliefs.json"


def _load_common_beliefs() -> list[dict]:
    """Return list of {name, beliefs} dicts, or [] if file missing."""
    if not COMMON_BELIEFS_PATH.exists():
        return []
    return json.loads(COMMON_BELIEFS_PATH.read_text(encoding="utf-8"))["categories"]

# ------------------------------------------------------------------
# Session state helpers
# ------------------------------------------------------------------


def _get_engine() -> BeliefMap:
    """Return the BeliefMap stored in session state, loading from disk if needed."""
    if "engine" not in st.session_state:
        if (DATA_DIR / "beliefs.json").exists():
            bmap = load_map(DATA_DIR)
        else:
            bmap = BeliefMap()
        bmap.cache = ResultCache(DATA_DIR / "cache.db")
        bmap.rate_limiter = RateLimiter(max_rpm=50)
        st.session_state.engine = bmap
    return st.session_state.engine


def _persist() -> None:
    """Save current engine state to disk."""
    if "engine" in st.session_state:
        save_map(st.session_state.engine, DATA_DIR)


def _reset_engine() -> None:
    """Replace the engine with a fresh BeliefMap."""
    bmap = BeliefMap()
    bmap.cache = ResultCache(DATA_DIR / "cache.db")
    bmap.rate_limiter = RateLimiter(max_rpm=50)
    st.session_state.engine = bmap
    _persist()


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Belief Compatibility Mapper",
    page_icon=":compass:",
    layout="wide",
)
st.title("Belief Compatibility Mapper")

engine = _get_engine()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.header("Controls")

    # -- Embedding model selector --
    embed_model = st.radio(
        "Embedding model",
        ["local-tfidf", "local-bow"],
        index=0,
        help="local-tfidf gives better results but needs scikit-learn.",
    )

    # -- Analysis model selector --
    analysis_model = st.selectbox(
        "Analysis model",
        [
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
        ],
        index=0,
        help="Claude model used for tension analysis.",
        key="analysis_model",
    )

    st.divider()

    # -- Add belief --
    st.subheader("Add Belief")
    new_belief_text = st.text_input("Belief text", key="add_belief_text")
    new_belief_tags = st.text_input(
        "Tags (comma-separated)", key="add_belief_tags",
        help="Optional thematic tags like 'ethics, economics'",
    )
    if st.button("Add", key="add_btn"):
        if not new_belief_text.strip():
            st.warning("Please enter a belief.")
        elif len(engine.beliefs) >= MAX_BELIEFS:
            st.warning(f"Maximum of {MAX_BELIEFS} beliefs reached. Remove one first.")
        else:
            tags = [t.strip() for t in new_belief_tags.split(",") if t.strip()] if new_belief_tags else []
            try:
                b = engine.add_belief(new_belief_text.strip(), tags=tags)
                _persist()
                st.success(f"Added belief [{b.id}]: {b.text}")
            except ValueError as exc:
                st.error(str(exc))

    st.divider()

    # -- Quick Add preset beliefs --
    _common = _load_common_beliefs()
    if _common:
        with st.expander("Quick Add", expanded=False):
            cat_names = [c["name"] for c in _common]
            sel_cat = st.selectbox("Category", cat_names, key="quick_add_cat")
            cat_beliefs = next(c["beliefs"] for c in _common if c["name"] == sel_cat)
            existing_texts = {b.text for b in engine.beliefs.values()}
            for j, belief_text in enumerate(cat_beliefs):
                already = belief_text in existing_texts
                col_t, col_b = st.columns([4, 1])
                with col_t:
                    st.caption(belief_text)
                with col_b:
                    if already:
                        st.caption("✓")
                    elif st.button("Add", key=f"qa_sb_{sel_cat}_{j}"):
                        if len(engine.beliefs) >= MAX_BELIEFS:
                            st.warning(f"Maximum of {MAX_BELIEFS} beliefs reached.")
                        else:
                            try:
                                b = engine.add_belief(belief_text)
                                _persist()
                                st.success(f"Added [{b.id}].")
                                st.rerun()
                            except ValueError as exc:
                                st.error(str(exc))

    st.divider()

    # -- Remove belief --
    st.subheader("Remove Belief")
    beliefs_list = engine.list_beliefs()
    if beliefs_list:
        options = {f"[{b.id}] {b.text[:60]}": b.id for b in beliefs_list}
        selected = st.selectbox("Select belief", list(options.keys()), key="remove_select")
        if st.button("Remove", key="remove_btn"):
            engine.remove_belief(options[selected])
            _persist()
            st.success(f"Removed belief {options[selected]}")
            st.rerun()
    else:
        st.caption("No beliefs to remove.")

    st.divider()

    # -- Load demo profile --
    st.subheader("Load Demo Profile")
    available_profiles = sorted(p.stem for p in PROFILES_DIR.glob("*.json"))
    if available_profiles:
        profile_name = st.selectbox("Profile", available_profiles, key="profile_select")
        if st.button("Load Profile", key="load_profile_btn"):
            profile_path = PROFILES_DIR / f"{profile_name}.json"
            data = json.loads(profile_path.read_text(encoding="utf-8"))

            _reset_engine()
            engine = _get_engine()

            for text in data["beliefs"]:
                engine.add_belief(text)

            score_matrix = data["scores"]
            n = len(data["beliefs"])
            for i in range(n):
                for j in range(i + 1, n):
                    engine.set_score(i, j, score_matrix[i][j])

            # Store justifications in session state for the Scored Pairs tab
            st.session_state.justifications = data.get("justifications", {})

            engine.generate_embeddings(model=embed_model)
            engine.calculate_initial_similarity()
            _persist()
            st.success(f"Loaded profile: {profile_name}")
            st.rerun()

    st.divider()

    # -- Run Analysis --
    st.subheader("Run Analysis")
    st.caption("Requires ANTHROPIC_API_KEY")
    if st.button("Run Analysis", key="analyze_btn"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error(
                "ANTHROPIC_API_KEY is not set. "
                "Set it before launching Streamlit:\n\n"
                "`export ANTHROPIC_API_KEY=sk-ant-...`"
            )
        elif len(engine.beliefs) < 2:
            st.warning("Add at least 2 beliefs before running analysis.")
        else:
            # Re-embed all beliefs together (TF-IDF dimensions depend
            # on corpus size). Bypass cache to avoid stale dimensions.
            for b in engine.beliefs.values():
                b.embedding = []
            saved_cache = engine.cache
            engine.cache = None
            engine.generate_embeddings(model=embed_model)
            engine.cache = saved_cache
            engine.calculate_initial_similarity()

            # Use threshold=0.0 so all pairs are analyzed when the
            # user explicitly clicks Run Analysis in the web UI.
            candidates = engine.interesting_pairs(threshold=0.0)
            to_analyze = [
                (a, b) for a, b, _ in candidates
                if np.isnan(engine.scores[a, b])
            ]
            total = len(to_analyze)

            if total == 0:
                st.info("All interesting pairs are already scored.")
            else:
                status = st.status(f"Analyzing {total} pair(s)...", expanded=True)
                progress = st.progress(0.0)
                completed = [0]

                def _on_progress():
                    completed[0] += 1
                    progress.progress(completed[0] / total)
                    status.update(label=f"Analyzed {completed[0]}/{total} pairs...")

                try:
                    results = engine.analyze_interesting(
                        threshold=0.0,
                        model=analysis_model,
                        on_progress=_on_progress,
                        persist_fn=_persist,
                    )
                    _persist()
                    status.update(label=f"Done! Analyzed {len(results)} pair(s).", state="complete")
                except Exception as exc:
                    status.update(label="Analysis failed.", state="error")
                    st.error(f"Error during analysis: {exc}")

            st.rerun()

    st.divider()

    # -- Recommend Belief --
    st.subheader("Recommend Belief")
    st.caption("Requires ANTHROPIC_API_KEY")
    rec_style = st.selectbox(
        "Style",
        ["complementary", "harmonious", "challenging"],
        key="rec_style",
        help="How the suggested belief should relate to your existing ones.",
    )
    rec_count = st.number_input("Count", min_value=1, max_value=5, value=1, key="rec_count")
    if st.button("Get Recommendations", key="rec_btn"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("ANTHROPIC_API_KEY is not set. Set it before launching Streamlit.")
        elif len(engine.beliefs) < 2:
            st.warning("Add at least 2 beliefs before requesting recommendations.")
        else:
            with st.spinner("Asking Claude..."):
                try:
                    recs = engine.recommend_belief(
                        count=int(rec_count),
                        style=rec_style,
                        model=analysis_model,
                    )
                    st.session_state.recommendations = recs
                    st.success(f"Got {len(recs)} recommendation(s). See the Recommendations tab.")
                except Exception as exc:
                    st.error(f"Error: {exc}")

    st.divider()

    # -- Bedrock Principles --
    st.subheader("Bedrock Principles")
    st.caption("Requires ANTHROPIC_API_KEY")
    if st.button("Find Bedrock Principles", key="bedrock_btn"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("ANTHROPIC_API_KEY is not set. Set it before launching Streamlit.")
        elif len(engine.beliefs) < 2:
            st.warning("Add at least 2 beliefs before identifying bedrock principles.")
        else:
            with st.spinner("Asking Claude for bedrock principles..."):
                try:
                    principles = engine.identify_bedrock_principles(model=analysis_model)
                    st.session_state.bedrock_principles = principles
                    st.success(f"Found {len(principles)} principle(s). See the Bedrock tab.")
                except Exception as exc:
                    st.error(f"Error: {exc}")

    st.divider()

    # -- Reset --
    if st.button("Reset All", key="reset_btn", type="secondary"):
        _reset_engine()
        st.session_state.pop("justifications", None)
        st.session_state.pop("bedrock_principles", None)
        st.rerun()


# ------------------------------------------------------------------
# Main area — 4 tabs
# ------------------------------------------------------------------

beliefs = engine.list_beliefs()

tab_beliefs, tab_heatmap, tab_network, tab_pairs, tab_recs, tab_bedrock, tab_quick, tab_dissonance = st.tabs(
    ["Beliefs", "Heatmap", "Network Graph", "Scored Pairs", "Recommendations",
     "Bedrock", "Quick Add", "Dissonance"]
)

# -- Tab 1: Beliefs table --
with tab_beliefs:
    if not beliefs:
        st.info("No beliefs yet. Use the sidebar to add beliefs or load a demo profile.")
    else:
        rows = []
        for b in beliefs:
            rows.append({
                "ID": b.id,
                "Text": b.text,
                "Tags": ", ".join(b.tags) if b.tags else "",
                "Embedded": bool(b.embedding),
            })
        st.dataframe(rows, width="stretch", hide_index=True)

# -- Tab 2: Heatmap --
with tab_heatmap:
    scored = engine.scored_pairs()
    if len(beliefs) < 2:
        st.info("Add at least 2 beliefs to see a heatmap.")
    elif not scored:
        st.info("Run analysis or load a demo profile to see the heatmap.")
    else:
        fig = build_heatmap_figure(beliefs, engine.scores)
        st.plotly_chart(fig, width="stretch")

# -- Tab 3: Network graph --
with tab_network:
    if not beliefs:
        st.info("Add beliefs to see the network graph.")
    elif not scored:
        st.info("Run analysis or load a demo profile to see the network graph.")
    else:
        threshold = st.slider(
            "Edge threshold (min |score| to draw an edge)",
            min_value=0.0, max_value=1.0, value=0.3, step=0.05,
            key="net_threshold",
        )
        highlight_contradictions = st.checkbox(
            "Highlight contradiction nodes",
            value=False,
            key="net_highlight_contradictions",
            help="Nodes in contradictory pairs (score ≤ -0.5) get a red border and larger size.",
        )

        node_overrides: dict[int, dict] = {}
        if highlight_contradictions:
            _alerts = engine.dissonance_report()
            _contra_ids = {a.belief_id_a for a in _alerts} | {a.belief_id_b for a in _alerts}
            node_overrides = {
                nid: {"size": 32, "line_color": "#FF0000", "line_width": 3}
                for nid in _contra_ids
            }

        fig = build_network_figure(
            beliefs, engine.beliefs, engine.scores,
            edge_threshold=threshold,
            node_overrides=node_overrides or None,
        )
        st.plotly_chart(fig, width="stretch")

# -- Tab 4: Scored pairs --
with tab_pairs:
    scored = engine.scored_pairs()
    if not scored:
        st.info("No scored pairs yet. Run analysis or load a demo profile.")
    else:
        justifications = st.session_state.get("justifications", {})
        sorted_pairs = sorted(scored, key=lambda t: t[2])
        rows = []
        for a, b, s in sorted_pairs:
            ba = engine.get_belief(a)
            bb = engine.get_belief(b)
            # Score label
            if s >= 0.5:
                cat = "Compatible"
            elif s >= 0.0:
                cat = "Neutral"
            elif s >= -0.5:
                cat = "Tensioned"
            else:
                cat = "Contradictory"
            key = f"{a}-{b}"
            just = justifications.get(key, "")
            rows.append({
                "Belief A": f"[{a}] {ba.text}",
                "Belief B": f"[{b}] {bb.text}",
                "Score": round(s, 3),
                "Category": cat,
                "Justification": just,
            })
        st.dataframe(rows, width="stretch", hide_index=True)

# -- Tab 5: Recommendations --
with tab_recs:
    recs = st.session_state.get("recommendations", [])
    if not recs:
        st.info(
            "No recommendations yet. Use 'Recommend Belief' in the sidebar "
            "to ask Claude for belief suggestions."
        )
    else:
        st.caption(
            f"{len(recs)} recommendation(s) — click 'Add to map' to include a belief."
        )
        for i, rec in enumerate(recs):
            with st.container(border=True):
                st.markdown(f"**{rec.text}**")
                st.caption(rec.justification)
                if st.button("Add to map", key=f"add_rec_{i}"):
                    if len(engine.beliefs) >= MAX_BELIEFS:
                        st.warning(f"Maximum of {MAX_BELIEFS} beliefs reached.")
                    else:
                        try:
                            b = engine.add_belief(rec.text)
                            _persist()
                            st.success(f"Added as belief [{b.id}].")
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))

# -- Tab 6: Bedrock Principles --
with tab_bedrock:
    principles = st.session_state.get("bedrock_principles", [])
    if not principles:
        st.info(
            "No bedrock principles yet. Use 'Find Bedrock Principles' in the sidebar "
            "to ask Claude to identify implicit foundational commitments in your belief map."
        )
    else:
        st.caption(
            f"{len(principles)} bedrock principle(s) — implicit upstream commitments "
            "that unify your surface beliefs."
        )
        for i, p in enumerate(principles):
            with st.container(border=True):
                st.markdown(f"### {p.principle}")
                coh = p.coherence
                color = "green" if coh >= 0.7 else ("orange" if coh >= 0.4 else "red")
                label = "Strong" if coh >= 0.7 else ("Moderate" if coh >= 0.4 else "Tentative")
                st.markdown(f"**Coherence:** :{color}[{label} ({coh:.2f})]")
                st.caption(p.explanation)
                with st.expander(f"Supporting beliefs ({len(p.belief_ids)})"):
                    for bid in p.belief_ids:
                        b = engine.beliefs.get(bid)
                        if b:
                            st.markdown(f"- **[{bid}]** {b.text}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Add to map", key=f"add_bedrock_{i}"):
                        if len(engine.beliefs) >= MAX_BELIEFS:
                            st.warning(f"Maximum of {MAX_BELIEFS} beliefs reached.")
                        else:
                            try:
                                b = engine.add_belief(p.principle)
                                _persist()
                                st.success(f"Added as belief [{b.id}].")
                                st.rerun()
                            except ValueError as exc:
                                st.error(str(exc))
                with col2:
                    n = len(p.belief_ids)
                    if st.button(f"Replace {n} supporting belief(s)", key=f"replace_bedrock_{i}"):
                        try:
                            removed = []
                            for bid in p.belief_ids:
                                if bid in engine.beliefs:
                                    engine.remove_belief(bid)
                                    removed.append(bid)
                            b = engine.add_belief(p.principle)
                            _persist()
                            st.session_state.pop("bedrock_principles", None)
                            st.success(
                                f"Replaced belief(s) {removed} with bedrock principle [{b.id}]."
                            )
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))

# -- Tab 7: Quick Add --
with tab_quick:
    common_categories = _load_common_beliefs()
    if not common_categories:
        st.warning("common_beliefs.json not found in project root.")
    else:
        existing_texts = {b.text for b in engine.beliefs.values()}
        st.caption(
            "Browse preset beliefs by category and add them to your map. "
            "Already-added entries are greyed out."
        )
        for cat in common_categories:
            with st.expander(f"**{cat['name']}** ({len(cat['beliefs'])} beliefs)", expanded=False):
                for j, belief_text in enumerate(cat["beliefs"]):
                    already = belief_text in existing_texts
                    col_t, col_b = st.columns([5, 1])
                    with col_t:
                        if already:
                            st.markdown(f":gray[{belief_text}]")
                        else:
                            st.markdown(belief_text)
                    with col_b:
                        if already:
                            st.caption("✓")
                        elif st.button("Add", key=f"qa_tab_{cat['name']}_{j}"):
                            if len(engine.beliefs) >= MAX_BELIEFS:
                                st.warning(f"Maximum of {MAX_BELIEFS} beliefs reached.")
                            else:
                                try:
                                    b = engine.add_belief(belief_text)
                                    _persist()
                                    st.success(f"Added [{b.id}]: {belief_text[:60]}")
                                    st.rerun()
                                except ValueError as exc:
                                    st.error(str(exc))

# -- Tab 8: Dissonance --
with tab_dissonance:
    scored = engine.scored_pairs()
    if not scored:
        st.info("No scored pairs yet. Run analysis or load a demo profile.")
    else:
        st.subheader("Cognitive Dissonance Detector")
        st.caption(
            "Contradictory pairs are ranked by severity. "
            "Dependent beliefs are those that positively align with one side "
            "of a contradiction and may be on shaky ground."
        )

        col_thresh, col_align = st.columns(2)
        with col_thresh:
            contra_thresh = st.slider(
                "Contradiction threshold",
                min_value=-1.0, max_value=0.0,
                value=-0.5, step=0.05,
                key="dissonance_contra_thresh",
                help="Pairs at or below this score are flagged.",
            )
        with col_align:
            align_thresh = st.slider(
                "Alignment threshold",
                min_value=0.0, max_value=1.0,
                value=0.3, step=0.05,
                key="dissonance_align_thresh",
                help="A belief C is 'at risk' if its score with A or B ≥ this.",
            )

        alerts = engine.dissonance_report(
            contradiction_threshold=contra_thresh,
            alignment_threshold=align_thresh,
        )

        if not alerts:
            st.success("No contradictory pairs detected at the current threshold.")
        else:
            st.warning(f"{len(alerts)} contradictory pair(s) detected.")

            for alert in alerts:
                ba = engine.beliefs.get(alert.belief_id_a)
                bb = engine.beliefs.get(alert.belief_id_b)
                if ba is None or bb is None:
                    continue

                if alert.severity >= 0.8:
                    badge_color, badge_label = "red", "CRITICAL"
                elif alert.severity >= 0.5:
                    badge_color, badge_label = "orange", "HIGH"
                else:
                    badge_color, badge_label = "yellow", "MODERATE"

                with st.container(border=True):
                    col_badge, col_score = st.columns([1, 3])
                    with col_badge:
                        st.markdown(f":{badge_color}[**{badge_label}**]")
                    with col_score:
                        st.markdown(
                            f"Score: `{alert.score:+.3f}` &nbsp; Severity: `{alert.severity:.2f}`"
                        )
                    st.markdown(f"**[{alert.belief_id_a}]** {ba.text}")
                    st.markdown(f"**[{alert.belief_id_b}]** {bb.text}")

                    if alert.dependent_ids:
                        with st.expander(
                            f"{len(alert.dependent_ids)} belief(s) at risk", expanded=False
                        ):
                            for dep_id in alert.dependent_ids:
                                dep = engine.beliefs.get(dep_id)
                                if dep:
                                    st.markdown(f"- **[{dep_id}]** {dep.text}")
                    else:
                        st.caption("No other beliefs at risk from this contradiction.")
