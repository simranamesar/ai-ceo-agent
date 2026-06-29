"""Executive Intelligence Dashboard (Streamlit).

Sidebar navigation (so every section is always reachable), bordered cards, and
colour-coded badges. Reads the JSON artifacts in data/outputs/ + data/processed/
and auto-matches the theme set in .streamlit/config.toml.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import time

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "outputs"
PROC = ROOT / "data" / "processed"

st.set_page_config(page_title="AI CEO - Strategic Intelligence", layout="wide")

# --- theme-aware chart styling ---
def _theme_base() -> str:
    try:
        with open(ROOT / ".streamlit" / "config.toml", "rb") as fh:
            return (tomllib.load(fh).get("theme", {}) or {}).get("base", "dark")
    except Exception:
        return "dark"


PLOTLY_TMPL = "plotly_dark" if _theme_base() == "dark" else "plotly_white"
SENT_COLORS = {"positive": "#2BB673", "neutral": "#9AA0A6", "negative": "#E0524B"}


def _load(path: Path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


@st.cache_data(show_spinner=False)
def load_all() -> dict:
    cfg = {}
    try:
        with open(ROOT / "config" / "config.yaml", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        cfg = {}
    return {
        "cfg": cfg,
        "intel": _load(OUT / "intelligence.json", {}),
        "recs": _load(OUT / "recommendations.json", []),
        "brief": _load(OUT / "briefing.json", {}),
        "sentiment": _load(OUT / "sentiment.json", {}),
        "verification": _load(OUT / "verification.json", {}),
        "docs": _load(PROC / "clean.json", []),
        "chunks": _load(PROC / "chunks.json", []),
    }


data = load_all()
cfg = data["cfg"]
company_cfg = cfg.get("company", {}) or {}
company = company_cfg.get("name", "Company")
industry = company_cfg.get("industry", "-")
competitors = company_cfg.get("competitors", []) or []
docs, intel = data["docs"], data["intel"]
recs, brief, sent = data["recs"], data["brief"], data["sentiment"]
verification = data["verification"]

st.markdown(
    """<style>
    .block-container {padding-top: 2.2rem; max-width: 1200px;}
    h1 {font-weight: 700; letter-spacing: -0.4px;}
    [data-testid="stMetricValue"] {font-weight: 700;}
    </style>""",
    unsafe_allow_html=True,
)


# --- shared renderers ---
def badge(value, kind: str) -> str:
    v = str(value or "").lower()
    palette = {
        "priority": {"high": "red", "medium": "orange", "low": "green"},
        "risk": {"high": "red", "medium": "orange", "low": "green"},
        "severity": {"high": "red", "medium": "orange", "low": "green"},
        "impact": {"high": "green", "medium": "orange", "low": "gray"},
        "direction": {"rising": "green", "declining": "red", "stable": "gray"},
    }.get(kind, {})
    color = palette.get(v, "gray")
    return f":{color}[**{value or '-'}**]"


def conf_bar(value) -> None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0.0
    st.progress(min(max(v, 0.0), 1.0), text=f"confidence {v:.0%}")


def render_evidence(evidence) -> None:
    if not evidence:
        st.caption("No linked evidence.")
        return
    with st.expander(f"Evidence ({len(evidence)})"):
        for e in evidence:
            title, url, src = e.get("title", "(untitled)"), e.get("url", ""), e.get("source", "")
            st.markdown(f"- **[{title}]({url})** · _{src}_" if url else f"- **{title}** · _{src}_")
            if e.get("snippet"):
                st.caption(e["snippet"])


# --- sidebar navigation ---
@st.cache_resource(show_spinner=False)
def _engines():
    """Load the embedder, vector store and LLM once for live Q&A."""
    from src.config import load_config
    from src.llm.client import LLMClient
    from src.store.embeddings import Embedder
    from src.store.vector_store import VectorStore

    rcfg = load_config()
    return rcfg, Embedder(rcfg["store"]["embedding_model"]), VectorStore(rcfg), LLMClient(rcfg)


with st.sidebar:
    st.markdown(f"### {company}")
    st.caption(industry)
    if intel.get("generated_at"):
        st.caption(f"Updated {intel['generated_at'][:10]}")
    st.divider()
    section = st.radio(
        "View",
        ["Strategic Advisor", "Overview", "Market Intelligence", "Opportunities", "Risks",
         "Trends", "Sentiment", "Recommendations", "CEO Briefing"],
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("Reload data"):
        st.cache_data.clear()
        st.rerun()
    auto_refresh = st.toggle("Auto-refresh", value=False)
    if auto_refresh:
        st.caption("Refreshing every 60 s")
        time.sleep(60)
        st.cache_data.clear()
        st.rerun()

# --- persistent header ---
st.title(f"Strategic Intelligence — {company}")
ov = sent.get("overall", {})
n_find = len(intel.get("opportunities", [])) + len(intel.get("risks", [])) + len(intel.get("trends", []))
h = st.columns(4)
h[0].metric("Documents", len(docs))
h[1].metric("Sources", len({d.get("source", "") for d in docs if d.get("source")}))
h[2].metric("Findings", n_find)
h[3].metric("Net sentiment", ov.get("net_polarity", "-"))
st.divider()

if not docs:
    st.warning("No data yet. Run `python -m pipeline.graph` first.")

# ===== Strategic Advisor (ReAct agent — Graph 2) =====
if section == "Strategic Advisor":
    st.subheader("Strategic Advisor")
    st.caption(
        "The LLM autonomously decides which "
        "tools to call — knowledge-base search, live news, sentiment, or metrics — "
        "before producing a grounded answer."
    )
    q = st.text_input(
        "Your question",
        placeholder="e.g. Should NVIDIA prioritise data-centre GPUs over gaming this quarter?",
    )

    if st.button("Ask", type="primary") and q.strip():
        with st.spinner("Agent reasoning…"):
            try:
                rcfg, emb, store, _ = _engines()
                from src.agent.react_agent import run_react_agent
                res = run_react_agent(q.strip(), cfg=rcfg, store=store, embedder=emb)
                st.markdown(res["answer"] or "_No answer produced._")
                if res.get("tool_trace"):
                    with st.expander(f"Agent trace ({res['steps']} tool call(s))", expanded=False):
                        for i, step in enumerate(res["tool_trace"], 1):
                            inp = step.get("input", {})
                            inp_str = next(iter(inp.values()), "") if inp else ""
                            st.markdown(f"**Step {i}** — `{step['tool']}`")
                            if inp_str:
                                st.caption(f"Input: {inp_str}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Agent error: {e}")

# ===== Overview (Section 1: Company Overview) =====
elif section == "Overview":
    st.subheader("Company overview")
    st.markdown(f"**Company:** {company}  ·  **Industry:** {industry}")
    _last = intel.get("generated_at", "")
    _last_disp = _last.replace("T", " ")[:16] if _last else "not built yet"
    g = st.columns(3)
    g[0].metric("Documents collected", len(docs))
    g[1].metric("Data sources", len({d.get("source", "") for d in docs if d.get("source")}))
    g[2].metric("Last updated", _last_disp)
    nxt = (brief or {}).get("what_to_do_next", "")
    if nxt:
        st.markdown("**Executive snapshot**")
        st.info(nxt)
    if docs:
        counts = pd.Series([d.get("source", "") for d in docs]).value_counts().reset_index()
        counts.columns = ["source", "documents"]
        st.plotly_chart(px.bar(counts, x="source", y="documents", template=PLOTLY_TMPL,
                               title="Documents per source"), width="stretch")

# ===== Market Intelligence =====
elif section == "Market Intelligence":
    st.subheader("Recent news")
    news = sorted([d for d in docs if d.get("source") == "news"],
                  key=lambda x: x.get("published", ""), reverse=True)
    for d in news[:8]:
        st.markdown(f"- [{d.get('title','')}]({d.get('url','')})")
    st.subheader("Company announcements")
    for d in [d for d in docs if d.get("source") == "company"][:6]:
        st.markdown(f"- [{d.get('title','')}]({d.get('url','')})")
    st.subheader("Emerging technologies & trends")
    _mtr = intel.get("trends", []) or []
    if _mtr:
        for t in _mtr[:6]:
            st.markdown(f"- **{t.get('title','')}** {badge(t.get('direction'), 'direction')}")
    else:
        st.caption("No trends yet — run `python -m pipeline.graph` with the LLM endpoint active.")
    st.subheader("Competitor activity")
    rows = [{"competitor": c,
             "mentions": sum(1 for d in docs
                             if c.lower() in (d.get("title", "") + d.get("text", "")).lower())}
            for c in competitors]
    if rows:
        mdf = pd.DataFrame(rows).sort_values("mentions", ascending=False)
        st.plotly_chart(px.bar(mdf, x="competitor", y="mentions", template=PLOTLY_TMPL,
                               title="Competitor mentions in corpus"), width="stretch")

# ===== Opportunities =====
elif section == "Opportunities":
    opps = intel.get("opportunities", []) or []
    st.subheader(f"Opportunities ({len(opps)})")
    if not opps:
        st.info("No opportunities yet — run `python -m pipeline.graph` with the LLM endpoint active.")
    for o in opps:
        with st.container(border=True):
            st.markdown(f"#### {o.get('title','')}")
            if o.get("description"):
                st.write(o["description"])
            st.markdown(f"Impact: {badge(o.get('impact'), 'impact')}")
            conf_bar(o.get("confidence", 0))
            render_evidence(o.get("evidence", []))

# ===== Risks =====
elif section == "Risks":
    risks = intel.get("risks", []) or []
    st.subheader(f"Risks ({len(risks)})")
    if not risks:
        st.info("No risks yet — run `python -m pipeline.graph` with the LLM endpoint active.")
    for r in risks:
        with st.container(border=True):
            st.markdown(f"#### {r.get('title','')}")
            st.markdown(f"Category: **{r.get('category','-')}**  ·  "
                        f"Severity: {badge(r.get('severity'), 'severity')}")
            conf_bar(r.get("confidence", 0))
            render_evidence(r.get("evidence", []))

# ===== Trends =====
elif section == "Trends":
    trends = intel.get("trends", []) or []
    st.subheader(f"Trends ({len(trends)})")
    if not trends:
        st.info("No trends yet — run `python -m pipeline.graph` with the LLM endpoint active.")
    for t in trends:
        with st.container(border=True):
            st.markdown(f"#### {t.get('title','')}")
            if t.get("description"):
                st.write(t["description"])
            st.markdown(f"Direction: {badge(t.get('direction'), 'direction')}")
            conf_bar(t.get("confidence", 0))
            render_evidence(t.get("evidence", []))

# ===== Sentiment =====
elif section == "Sentiment":
    if not sent:
        st.info("Run the pipeline to generate sentiment.")
    else:
        c = st.columns(4)
        c[0].metric("Positive", ov.get("positive", 0))
        c[1].metric("Neutral", ov.get("neutral", 0))
        c[2].metric("Negative", ov.get("negative", 0))
        c[3].metric("Net polarity", ov.get("net_polarity", 0))
        dist = [{"sentiment": k, "count": ov.get(k, 0)} for k in ("positive", "neutral", "negative")]
        if any(d["count"] for d in dist):
            st.plotly_chart(
                px.pie(pd.DataFrame(dist), names="sentiment", values="count", hole=0.5,
                       color="sentiment", color_discrete_map=SENT_COLORS, template=PLOTLY_TMPL,
                       title="Overall sentiment distribution"),
                width="stretch")
        cats = sent.get("by_category", {})
        rows = [{"category": cat, "sentiment": lab, "count": cats.get(cat, {}).get(lab, 0)}
                for cat in ("news", "public") for lab in ("positive", "neutral", "negative")]
        if any(r["count"] for r in rows):
            st.plotly_chart(
                px.bar(pd.DataFrame(rows), x="category", y="count", color="sentiment",
                       color_discrete_map=SENT_COLORS, template=PLOTLY_TMPL,
                       title="News vs public sentiment"),
                width="stretch")
        by_src = sent.get("by_source", {})
        if by_src:
            srows = [{"source": s, "sentiment": lab, "count": by_src.get(s, {}).get(lab, 0)}
                     for s in by_src for lab in ("positive", "neutral", "negative")]
            if any(r2["count"] for r2 in srows):
                st.plotly_chart(
                    px.bar(pd.DataFrame(srows), x="source", y="count", color="sentiment",
                           color_discrete_map=SENT_COLORS, template=PLOTLY_TMPL,
                           title="Sentiment by source"),
                    width="stretch")
        trend = sent.get("trend", [])
        if trend:
            st.plotly_chart(
                px.line(pd.DataFrame(trend), x="date", y="net_polarity", markers=True,
                        template=PLOTLY_TMPL, title="Sentiment trend (net polarity over time)"),
                width="stretch")

# ===== Recommendations =====
elif section == "Recommendations":
    st.subheader(f"Strategic recommendations ({len(recs)})")

    # --- Verification report (Graph 1 verify node output) ---
    if verification:
        total = verification.get("total", 0)
        passed = verification.get("passed", 0)
        failed = verification.get("failed", 0)
        vcols = st.columns(3)
        vcols[0].metric("Total drafted", total)
        vcols[1].metric("Passed validation", passed, delta=None)
        vcols[2].metric("Rejected (< 3 evidence)", failed)
        if failed and verification.get("rejected"):
            with st.expander(f"{failed} rejected recommendation(s)", expanded=False):
                for rej in verification["rejected"]:
                    st.markdown(f"- **{rej.get('recommendation','')}**  \n"
                                f"  _{rej.get('rejection_reason','')}_")
        st.divider()

    if not recs:
        st.info("No recommendations yet — run `python -m pipeline.graph` with the LLM endpoint active.")
    order = {"high": 0, "medium": 1, "low": 2}
    for r in sorted(recs, key=lambda x: order.get(str(x.get("priority", "")).lower(), 3)):
        with st.container(border=True):
            st.markdown(f"#### {r.get('recommendation','')}")
            st.markdown(f"Priority: {badge(r.get('priority'), 'priority')}  ·  "
                        f"Risk: {badge(r.get('risk_level'), 'risk')}")
            if r.get("rationale"):
                st.write(f"**Rationale.** {r['rationale']}")
            if r.get("expected_impact"):
                st.write(f"**Expected impact.** {r['expected_impact']}")
            render_evidence(r.get("evidence", []))

# ===== CEO Briefing =====
elif section == "CEO Briefing":
    st.subheader("CEO Briefing")
    if not brief or not any(brief.values()):
        st.info("Run the pipeline to generate the briefing.")
    else:
        for key, label in (("what_happened", "What happened"),
                           ("why_it_matters", "Why it matters"),
                           ("what_to_do_next", "What to do next")):
            with st.container(border=True):
                st.markdown(f"##### {label}")
                st.write(brief.get(key, "") or "-")
