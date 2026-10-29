"""
StudioX - Local AI Document Studio
A privacy-first document intelligence and creative studio.
Optimized for Windows, CPU-only execution, and 16 GB RAM.

UI: soft lilac / cream dashboard with sidebar navigation, greeting header,
rounded cards and donut-ring stat cards.
"""

import html
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd

# Core Modules
from src.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    UPLOAD_DIR,
)
from src.ollama_client import check_ollama_status
from src.pdf_processor import validate_pdf, process_pdf, compute_file_hash
from src.document_manager import document_manager
from src.vector_store import (
    index_chunks,
    get_collection_stats,
    get_indexed_sources,
    delete_document_vectors,
)
from src.rag_pipeline import answer_question
from src.content_generator import (
    GENERATION_TEMPLATES,
    generate_content_asset,
)
from src.visual_builder import (
    STUDIO_PALETTES,
    generate_poster_image,
    generate_brochure_image,
    generate_chart_image,
    parse_sections_from_markdown,
    parse_panels_from_markdown,
    parse_chart_data,
)
from src.export_manager import (
    export_to_pdf,
    export_to_docx,
    export_to_markdown,
    export_to_txt,
    export_to_csv,
    export_image_to_png,
)
from src.evaluation import (
    evaluate_question,
    evaluate_indexing_performance,
    run_benchmark_suite,
    DEFAULT_BENCHMARK_SUITE,
)

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StudioX • Local AI Document Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _h(s: str) -> str:
    """Flatten HTML/CSS so Markdown never treats indented lines as code blocks."""
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


# NOTE: the font import below fetches Nunito from Google Fonts. For a fully
# offline setup, delete that @import line; the Segoe UI fallback looks fine.
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #CDB5CB;
  --panel: #F7EEE4;
  --side: #E6D7D5;
  --card: #DCC8D4;
  --card-2: #E9DCE1;
  --deep: #957A93;
  --ink: #5A3D5B;
  --ink-soft: #8A7189;
  --accent: #7A5A7B;
  --line: rgba(90, 61, 91, 0.14);
  --pink: #E5738B;
  --teal: #86C5BE;
  --gold: #F0B05A;
}

/* ---------- Base ---------- */
.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp li {
  font-family: 'Nunito', 'Segoe UI', 'Trebuchet MS', -apple-system, sans-serif;
}
.stApp { background: linear-gradient(160deg, #D3BCD0 0%, #C6ADC4 100%); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4, h5, h6 { color: var(--ink) !important; font-weight: 700; letter-spacing: -0.01em; }
p, li, label, .stMarkdown { color: var(--ink); }
hr { border-color: var(--line) !important; }

/* ---------- Main panel (cream) ---------- */
.block-container {
  max-width: none !important;
  width: calc(100% - 24px) !important;
  min-height: calc(100vh - 48px);
  margin: 24px 24px 24px 0 !important;
  padding: 2rem 2.5rem 2.5rem 2.5rem !important;
  background: var(--panel);
  border-radius: 0 28px 28px 0;
  box-shadow: 0 18px 50px rgba(90, 61, 91, 0.18);
}
[data-testid="stVerticalBlock"] { gap: 0.85rem; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background: transparent !important; border: none !important; }
section[data-testid="stSidebar"][aria-expanded="true"] { width: 18rem !important; min-width: 18rem !important; }
section[data-testid="stSidebar"] > div:first-child {
  background: var(--side);
  margin: 24px 0 24px 24px;
  height: calc(100vh - 48px);
  border-radius: 28px 0 0 28px;
  overflow-y: auto;
}
[data-testid="stSidebarHeader"] { padding: 0.5rem 1rem 0 1rem; }
[data-testid="stSidebarUserContent"] { padding: 0.4rem 1rem 1.2rem 1rem; }
section[data-testid="stSidebar"] * { color: var(--ink); }

.brand { font-size: 1.55rem; font-weight: 600; letter-spacing: 0.01em; color: var(--ink); margin: 4px 0 18px 4px; }
.brand b { color: #C0627A; font-weight: 800; }
.profile { display: flex; align-items: center; gap: 12px; margin: 0 0 18px 4px; }
.avatar {
  width: 52px; height: 52px; border-radius: 50%;
  background: linear-gradient(135deg, #B98FB4, #7A5A7B);
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-size: 1.4rem; border: 3px solid var(--panel);
}
.profile-name { font-weight: 700; font-size: 0.92rem; line-height: 1.2; }
.profile-sub { font-size: 0.74rem; color: var(--ink-soft); }
.side-label { font-size: 0.72rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-soft); margin: 18px 0 6px 4px; }
.side-stats { display: flex; gap: 8px; margin: 10px 0 4px 0; }
.side-stat { flex: 1; background: var(--panel); border-radius: 14px; padding: 8px 10px; text-align: center; }
.side-stat b { display: block; font-size: 1.15rem; font-weight: 800; }
.side-stat span { font-size: 0.68rem; color: var(--ink-soft); font-weight: 700; }
.side-engine { display: flex; align-items: center; gap: 8px; background: var(--panel); border-radius: 14px; padding: 10px 12px; font-size: 0.78rem; font-weight: 700; margin-top: 14px; }
.side-engine small { display: block; font-weight: 600; color: var(--ink-soft); font-size: 0.68rem; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: none; }

/* Sidebar navigation (radio -> pill menu) */
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { gap: 4px; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
  width: 100%; margin: 0; padding: 11px 16px; border-radius: 14px 0 0 14px; cursor: pointer;
  transition: background 0.15s ease;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child { display: none; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size: 0.93rem; font-weight: 600; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(247, 238, 228, 0.55); }
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background: var(--panel); box-shadow: 0 2px 10px rgba(90, 61, 91, 0.08); }
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p { font-weight: 800; }

/* ---------- Page header ---------- */
.page-head { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px 20px; margin: 0 0 22px 0; }
.page-title { font-size: 1.8rem; font-weight: 600; letter-spacing: 0.01em; line-height: 1.15; color: var(--ink); }
.page-sub { font-size: 0.86rem; color: var(--ink-soft); margin-top: 4px; font-weight: 600; }
.head-right { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.head-pill {
  display: inline-flex; align-items: center; gap: 7px; background: var(--card);
  border-radius: 999px; padding: 8px 16px; font-size: 0.76rem; font-weight: 700; color: var(--ink);
}

/* ---------- Cards ---------- */
.card, [data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card);
  border: none !important;
  border-radius: 22px;
}
.card { padding: 18px 20px; margin-bottom: 4px; }
[data-testid="stVerticalBlockBorderWrapper"] { padding: 8px 10px; }
.card-title { display: flex; justify-content: space-between; align-items: center; font-size: 0.88rem; font-weight: 800; color: var(--ink); margin-bottom: 12px; }
.card-link { font-size: 0.72rem; font-weight: 700; color: var(--ink-soft); }
.empty { font-size: 0.82rem; color: var(--ink-soft); font-weight: 600; padding: 8px 2px; }

/* Pill rows */
.pill-row {
  display: flex; align-items: center; gap: 10px; background: var(--panel);
  border-radius: 999px; padding: 6px 8px; margin-bottom: 8px; border: 1px solid var(--line);
}
.pill-avatar {
  width: 32px; height: 32px; flex: none; border-radius: 50%; background: var(--deep);
  color: #fff; display: flex; align-items: center; justify-content: center; font-size: 0.9rem;
}
.pill-text { min-width: 0; flex: 1; }
.pill-name { font-size: 0.8rem; font-weight: 800; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pill-sub { font-size: 0.68rem; color: var(--ink-soft); font-weight: 600; }
.pill-dots {
  width: 26px; height: 26px; flex: none; border-radius: 50%; background: var(--deep);
  color: #fff; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; letter-spacing: 1px;
}
.sched-row {
  display: flex; align-items: center; gap: 12px; background: var(--panel);
  border-radius: 999px; padding: 6px 18px 6px 6px; margin-bottom: 8px;
}
.sched-day {
  width: 38px; height: 38px; flex: none; border-radius: 50%; border: 2px solid var(--deep);
  display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.95rem;
}

/* Right stat panel */
.side-panel {
  border: 1.5px solid rgba(90, 61, 91, 0.28); background: var(--card-2);
  border-radius: 26px; padding: 14px; display: flex; flex-direction: column; gap: 12px;
}
.stat-card { background: var(--deep); border-radius: 22px; padding: 16px 10px 14px 10px; text-align: center; color: #F7EEE4; }
.stat-title { font-size: 0.84rem; font-weight: 700; margin-bottom: 8px; color: #F7EEE4; }
.stat-cap { font-size: 0.7rem; opacity: 0.9; margin-top: 6px; font-weight: 600; color: #F7EEE4; }

/* Citations */
.citation-badge {
  display: inline-flex; align-items: center; gap: 4px; background: rgba(122, 90, 123, 0.12);
  color: var(--accent); border: 1px solid rgba(122, 90, 123, 0.28);
  padding: 3px 10px; border-radius: 999px; font-size: 0.76rem; font-weight: 700; margin: 0 6px 4px 0;
}
.citation-badge-page { background: rgba(240, 176, 90, 0.18); color: #9A5F12; border-color: rgba(240, 176, 90, 0.5); }

/* Executive preview */
.exec-box { background: var(--card); border-radius: 22px; padding: 22px 24px; margin-bottom: 14px; }
.exec-kicker { font-size: 0.74rem; font-weight: 800; color: var(--accent); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.exec-meta { font-size: 0.82rem; color: var(--ink-soft); font-weight: 600; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
  background: var(--panel); color: var(--ink); border: 1px solid var(--line);
  border-radius: 999px; font-weight: 700; font-size: 0.85rem; padding: 0.5rem 1.1rem;
  box-shadow: none; transition: all 0.15s ease;
}
.stButton > button p, .stDownloadButton > button p { color: inherit; font-weight: 700; }
.stButton > button:hover, .stDownloadButton > button:hover {
  background: var(--card-2); border-color: var(--accent); color: var(--accent);
  transform: translateY(-1px); box-shadow: 0 6px 14px rgba(90, 61, 91, 0.12);
}
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: var(--accent) !important; color: #fff !important; border-color: var(--accent) !important;
}
button[kind="primary"] p, [data-testid="stBaseButton-primary"] p { color: #fff !important; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { background: #664766 !important; }

/* ---------- Inputs ---------- */
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
  background: var(--panel) !important; border-radius: 14px !important; border: 1px solid var(--line) !important;
}
[data-baseweb="base-input"] { background: transparent !important; }
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea, [data-baseweb="select"] div { color: var(--ink) !important; }
[data-baseweb="tag"] { background: var(--accent) !important; border-radius: 999px !important; }
[data-baseweb="tag"] span, [data-baseweb="tag"] svg { color: #fff !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--card); border: 2px dashed rgba(90, 61, 91, 0.35); border-radius: 20px; }
[data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }

/* ---------- Metrics, tables, expanders, alerts ---------- */
[data-testid="stMetric"] { background: var(--card); border-radius: 18px; padding: 14px 18px; }
[data-testid="stMetricLabel"] p { color: var(--ink-soft) !important; font-weight: 700; }
[data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 800; }
[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
[data-testid="stExpander"] { background: var(--card-2); border: none !important; border-radius: 16px; }
[data-testid="stExpander"] details { border: none !important; }
[data-testid="stAlert"] { border-radius: 16px; }

/* ---------- Tabs (pill style) ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: var(--card-2); padding: 5px; border-radius: 999px; width: fit-content; }
.stTabs [data-baseweb="tab"] { border-radius: 999px; padding: 6px 18px; height: auto; background: transparent; font-weight: 700; }
.stTabs [data-baseweb="tab"] p { font-size: 0.86rem; font-weight: 700; }
.stTabs [aria-selected="true"] { background: var(--accent) !important; }
.stTabs [aria-selected="true"] p { color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] { background: var(--card); border-radius: 20px; padding: 14px 18px; }
[data-testid="stChatInput"] { background: var(--card); border-radius: 999px; border: 1px solid var(--line); }
[data-testid="stChatInput"] textarea { background: transparent !important; }
[data-testid="stBottom"], [data-testid="stBottom"] > div { background: transparent !important; }

/* ---------- Small screens ---------- */
@media (max-width: 900px) {
  .block-container { margin: 12px !important; width: auto !important; border-radius: 22px; padding: 1.2rem !important; }
  section[data-testid="stSidebar"] > div:first-child { margin: 0; height: 100vh; border-radius: 0; }
  .page-title { font-size: 1.4rem; }
}
"""

st.markdown("<style>\n" + _h(APP_CSS) + "\n</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session state & storage sync
# -----------------------------------------------------------------------------
if "initialized" not in st.session_state:
    known_sources = get_indexed_sources()
    document_manager.sync_with_storage(known_sources)
    st.session_state.initialized = True

st.session_state.setdefault("messages", [])
st.session_state.setdefault("created_asset", None)
st.session_state.setdefault("report_asset", None)
st.session_state.setdefault("nav", "Dashboard")


def go(page_key: str):
    """Button callback: jump to another page."""
    st.session_state.nav = page_key


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def ring_svg(pct, color, size=100, stroke=10):
    pct = max(0, min(100, int(round(pct))))
    arc = (
        f'<circle cx="50" cy="50" r="40" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" pathLength="100" stroke-dasharray="{pct} 100" '
        f'transform="rotate(-90 50 50)"/>'
        if pct > 0
        else ""
    )
    return (
        f'<svg viewBox="0 0 100 100" width="{size}" height="{size}">'
        f'<circle cx="50" cy="50" r="40" fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="{stroke}"/>'
        f"{arc}"
        f'<text x="50" y="50" text-anchor="middle" dominant-baseline="central" '
        f'font-size="19" font-weight="800" fill="#F7EEE4">{pct}%</text></svg>'
    )


def stat_card(title, pct, color, caption):
    return (
        f'<div class="stat-card"><div class="stat-title">{html.escape(title)}</div>'
        f"{ring_svg(pct, color)}"
        f'<div class="stat-cap">{html.escape(caption)}</div></div>'
    )


def card_title(title, right=""):
    link = f'<span class="card-link">{html.escape(right)}</span>' if right else ""
    return f'<div class="card-title"><span>{html.escape(title)}</span>{link}</div>'


def pill_row(icon, name, sub):
    return (
        f'<div class="pill-row"><div class="pill-avatar">{icon}</div>'
        f'<div class="pill-text"><div class="pill-name">{html.escape(name)}</div>'
        f'<div class="pill-sub">{html.escape(sub)}</div></div>'
        f'<div class="pill-dots">···</div></div>'
    )


def sched_row(day, name, sub):
    return (
        f'<div class="sched-row"><div class="sched-day">{html.escape(str(day))}</div>'
        f'<div class="pill-text"><div class="pill-name">{html.escape(name)}</div>'
        f'<div class="pill-sub">{html.escape(sub)}</div></div></div>'
    )


# -----------------------------------------------------------------------------
# Sidebar: brand, profile, navigation, scope, engine status
# -----------------------------------------------------------------------------
NAV_ITEMS = {
    "Dashboard": "🏠  Dashboard",
    "Chat": "💬  Chat Intelligence",
    "Documents": "📁  Document Hub",
    "Studio": "✨  Visual Studio",
    "Reports": "📑  Executive Reports",
    "Benchmark": "📊  Quality & Benchmark",
}

ollama_info = check_ollama_status()
all_docs = document_manager.get_all_documents()
doc_names = [d["filename"] for d in all_docs]
stats = get_collection_stats()

with st.sidebar:
    st.markdown(
        _h(
            """
            <div class="brand">LocalBrain<b></b></div>
            <div class="profile">
              <div class="avatar">⚡</div>
              <div>
                <div class="profile-name">Local Workspace</div>
                <div class="profile-sub">CPU • 16 GB • Private</div>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.radio(
        "Navigation",
        options=list(NAV_ITEMS.keys()),
        format_func=lambda k: NAV_ITEMS[k],
        key="nav",
        label_visibility="collapsed",
    )

    st.markdown('<div class="side-label">Knowledge Scope</div>', unsafe_allow_html=True)
    scope_option = st.selectbox(
        "Scope mode",
        ["All Indexed Documents", "Selected Document(s)"],
        key="scope_mode",
        label_visibility="collapsed",
    )

    selected_docs = None
    if scope_option == "Selected Document(s)":
        if doc_names:
            selected_docs = st.multiselect(
                "Filter queries & generation to:",
                options=doc_names,
                default=doc_names[:1],
                help="Focuses retrieval and creative generation to the chosen documents.",
                label_visibility="collapsed",
            )
        else:
            st.info("No documents yet. Upload a PDF in Document Hub.")

    st.markdown(
        _h(
            f"""
            <div class="side-stats">
              <div class="side-stat"><b>{len(all_docs)}</b><span>Documents</span></div>
              <div class="side-stat"><b>{stats.get("total_chunks", 0)}</b><span>Chunks</span></div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    if ollama_info["connected"]:
        st.markdown(
            _h(
                f"""
                <div class="side-engine"><span class="dot" style="background:#5FA98B"></span>
                <div>Ollama Active<small>{html.escape(DEFAULT_LLM_MODEL)} • {html.escape(DEFAULT_EMBEDDING_MODEL)}</small></div></div>
                """
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="side-engine"><span class="dot" style="background:#D98A4E"></span>'
            "<div>Ollama Offline<small>Start the local engine</small></div></div>",
            unsafe_allow_html=True,
        )
        with st.expander("Setup Instructions"):
            st.code("ollama serve\nollama pull llama3.2:3b\nollama pull nomic-embed-text")

page = st.session_state.nav

scope_label = (
    f"{len(selected_docs)} Selected Document(s)"
    if selected_docs
    else f"All {len(all_docs)} Documents"
)


def page_header(title, subtitle=""):
    connected = ollama_info["connected"]
    dot = "#5FA98B" if connected else "#D98A4E"
    state = "Ollama online" if connected else "Ollama offline"
    st.markdown(
        _h(
            f"""
            <div class="page-head">
              <div>
                <div class="page-title">{html.escape(title)}</div>
                <div class="page-sub">{html.escape(subtitle)}</div>
              </div>
              <div class="head-right">
                <span class="head-pill"><span class="dot" style="background:{dot}"></span>{state}</span>
                <span class="head-pill">📂 {html.escape(scope_label)}</span>
                <span class="head-pill">🔒 100% Private</span>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# DASHBOARD
# =============================================================================
if page == "Dashboard":
    hour = datetime.now().hour
    greet = "GOOD MORNING" if hour < 12 else "GOOD AFTERNOON" if hour < 18 else "GOOD EVENING"
    page_header(f"{greet}, STUDIO!", "Your private document intelligence workspace.")

    left, right = st.columns([2.2, 1], gap="large")

    recent_docs = sorted(all_docs, key=lambda d: d.get("indexed_at", ""), reverse=True)

    with left:
        c1, c2 = st.columns(2, gap="medium")

        with c1:
            body = card_title("Recent Documents", f"{len(all_docs)} total")
            if recent_docs:
                for d in recent_docs[:3]:
                    body += pill_row(
                        "📄",
                        d["filename"],
                        f"{d.get('page_count', '?')} pages • {d.get('chunk_count', '?')} chunks",
                    )
            else:
                body += '<div class="empty">No documents yet. Upload a PDF to begin.</div>'
            st.markdown(f'<div class="card">{body}</div>', unsafe_allow_html=True)

        with c2:
            with st.container(border=True):
                st.markdown(card_title("Quick Actions"), unsafe_allow_html=True)
                st.button("💬  Ask a question", key="qa_chat", on_click=go, args=("Chat",), use_container_width=True)
                st.button("📁  Upload a PDF", key="qa_docs", on_click=go, args=("Documents",), use_container_width=True)
                st.button("✨  Create a poster", key="qa_studio", on_click=go, args=("Studio",), use_container_width=True)
                st.button("📑  Write a report", key="qa_reports", on_click=go, args=("Reports",), use_container_width=True)

        # Library timeline (schedule-style card)
        body = card_title("My Library", "Latest indexed")
        if recent_docs:
            for d in recent_docs[:5]:
                day = (d.get("indexed_at") or "")[8:10] or "–"
                size_kb = round(d.get("file_size_bytes", 0) / 1024, 1)
                body += sched_row(
                    day,
                    d["filename"],
                    f"{(d.get('indexed_at') or '')[:10]} • {d.get('page_count', '?')} pages • {size_kb} KB",
                )
        else:
            body += '<div class="empty">Your indexed documents will appear here.</div>'
        st.markdown(f'<div class="card">{body}</div>', unsafe_allow_html=True)

        # Recent creations (projects-style card)
        body = card_title("My Creations", "This session")
        shown = False
        for key, icon in (("created_asset", "🎨"), ("report_asset", "📑")):
            a = st.session_state.get(key)
            if a:
                shown = True
                body += pill_row(icon, a["title"], f"{a.get('category', 'Report')} • {', '.join(a.get('sources', []))}")
        if not shown:
            body += '<div class="empty">Generate a poster, brochure or report and it will show up here.</div>'
        st.markdown(f'<div class="card">{body}</div>', unsafe_allow_html=True)

    with right:
        try:
            idx_sources = set(get_indexed_sources() or [])
        except Exception:
            idx_sources = set()
        coverage = (len(idx_sources & set(doc_names)) / len(doc_names) * 100) if doc_names else 0
        scope_pct = (len(selected_docs) / len(all_docs) * 100) if (selected_docs and all_docs) else (100 if all_docs else 0)
        n_questions = sum(1 for m in st.session_state.messages if m["role"] == "user")
        activity = min(100, n_questions * 10)

        panel = (
            stat_card("Index Coverage", coverage, "#E5738B", f"{len(idx_sources & set(doc_names))} of {len(doc_names)} docs embedded")
            + stat_card("Scope Coverage", scope_pct, "#86C5BE", "Share of library in active scope")
            + stat_card("Session Activity", activity, "#F0B05A", f"{n_questions} question(s) asked")
        )
        st.markdown(f'<div class="side-panel">{panel}</div>', unsafe_allow_html=True)
        st.button("See quality metrics  →", key="qa_bench", on_click=go, args=("Benchmark",), use_container_width=True)


# =============================================================================
# CHAT INTELLIGENCE
# =============================================================================
elif page == "Chat":
    page_header("ASK YOUR DOCUMENTS", "Grounded answers with page-level citations.")

    if selected_docs:
        chips = " ".join(f"<span class='citation-badge'>📄 {html.escape(d)}</span>" for d in selected_docs)
        st.markdown(f"<div><b>Active Scope:</b> {chips}</div>", unsafe_allow_html=True)

    q_cols = st.columns(4)
    prompt_chips = [
        ("💡 Executive Summary", "Summarize the key results and findings."),
        ("🔬 Core Problem", "What specific problem does this document address?"),
        ("⚙️ Methodology", "Explain the technical methodology and architecture."),
        ("📊 Key Features & Impact", "What are the core features and measurable outcomes?"),
    ]

    for col, (label, query_text) in zip(q_cols, prompt_chips):
        if col.button(label, use_container_width=True, key=f"chip_{label}"):
            st.session_state.messages.append({"role": "user", "content": query_text})
            with st.spinner("Synthesizing answer from document evidence..."):
                res = answer_question(
                    question=query_text,
                    chat_history=st.session_state.messages[:-1],
                    selected_sources=selected_docs,
                    k=6,
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res["answer"],
                    "sources": res["sources"],
                    "stats": res,
                })
            st.rerun()

    def render_sources(sources):
        with st.expander(f"📚 Supporting Citations ({len(sources)})"):
            for src in sources:
                st.markdown(
                    f"<span class='citation-badge'>📄 {html.escape(str(src['file']))}</span> "
                    f"<span class='citation-badge citation-badge-page'>Page {src['page']}</span> "
                    f"<span style='font-size: 0.75rem; color: #8A7189;'>Score: {src.get('score', 'N/A')}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"\"{src['excerpt']}\"")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])
            if msg.get("stats"):
                s = msg["stats"]
                st.caption(
                    f"⏱️ Retrieval: {s.get('retrieval_time_seconds', 0)}s · "
                    f"Generation: {s.get('generation_time_seconds', 0)}s · "
                    f"Chunks: {s.get('retrieved_chunks', 0)}"
                )

    user_query = st.chat_input("Ask anything about your documents...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documents & generating grounded response..."):
                    result = answer_question(
                        question=user_query,
                        chat_history=st.session_state.messages[:-1],
                        selected_sources=selected_docs,
                        k=6,
                    )

                st.markdown(result["answer"])
                st.caption(
                    f"⏱️ Retrieval: {result['retrieval_time_seconds']}s · "
                    f"Generation: {result['generation_time_seconds']}s · "
                    f"Chunks: {result['retrieved_chunks']}"
                )

                if result.get("rephrased_query") and result["rephrased_query"] != user_query:
                    with st.expander("Contextual Search Query"):
                        st.caption(f"Follow-up reformulated into: *{result['rephrased_query']}*")

                if result["sources"]:
                    render_sources(result["sources"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "stats": result,
                })
            except Exception as exc:
                st.error(f"Error generating answer: {exc}")

    if st.session_state.messages:
        st.markdown("---")
        c_clear, c_exp_md, c_exp_txt = st.columns(3)
        with c_clear:
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with c_exp_md:
            chat_md = "\n\n".join(f"### {m['role'].capitalize()}:\n{m['content']}" for m in st.session_state.messages)
            st.download_button(
                "📥 Export Chat (.md)",
                data=export_to_markdown("Chat Transcript", chat_md),
                file_name=f"chat_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with c_exp_txt:
            chat_txt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in st.session_state.messages)
            st.download_button(
                "📥 Export Chat (.txt)",
                data=export_to_txt("Chat Transcript", chat_txt),
                file_name=f"chat_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True,
            )


# =============================================================================
# DOCUMENT HUB
# =============================================================================
elif page == "Documents":
    page_header("DOCUMENT HUB", "Upload, index and manage your PDFs locally.")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs (Max 50MB per file):",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("⚡ Index Uploaded Documents", type="primary", use_container_width=True):
        log = []
        for uploaded_file in uploaded_files:
            safe_name = Path(uploaded_file.name).name
            target_path = UPLOAD_DIR / safe_name
            file_bytes = uploaded_file.getvalue()

            progress_bar = st.progress(0, text=f"Checking {safe_name}...")
            try:
                file_hash = compute_file_hash(file_bytes)
                is_dup, dup_name = document_manager.is_duplicate(file_hash)

                if is_dup:
                    log.append(("warning", f"⚠️ `{safe_name}` matches identical content already indexed as `{dup_name}`. Skipped."))
                    progress_bar.progress(100, text=f"Skipped duplicate {safe_name}.")
                    continue

                target_path.write_bytes(file_bytes)

                progress_bar.progress(35, text="Validating PDF integrity...")
                meta = validate_pdf(target_path)

                progress_bar.progress(65, text="Extracting text & generating chunks...")
                chunks = process_pdf(str(target_path))

                progress_bar.progress(85, text="Creating embeddings in ChromaDB...")
                index_chunks(chunks, doc_name=safe_name)

                document_manager.register_document(
                    filename=safe_name,
                    file_hash=file_hash,
                    file_size_bytes=meta["file_size"],
                    page_count=meta["total_pages"],
                    chunk_count=len(chunks),
                    title=meta["title"],
                    author=meta["author"],
                )

                progress_bar.progress(100, text=f"Successfully indexed {safe_name}!")
                log.append(("success", f"✅ Indexed `{safe_name}`: {meta['total_pages']} pages, {len(chunks)} chunks."))

            except Exception as exc:
                log.append(("error", f"❌ Failed indexing `{safe_name}`: {exc}"))
                if target_path.exists():
                    target_path.unlink()

        st.session_state.upload_log = log
        st.rerun()

    for kind, msg in st.session_state.pop("upload_log", []):
        getattr(st, kind)(msg)

    st.markdown("##### Document Inventory")
    all_docs = document_manager.get_all_documents()

    if not all_docs:
        st.info("No documents indexed yet. Upload a PDF above to begin.")
    else:
        doc_rows = []
        for d in all_docs:
            doc_rows.append({
                "Document Name": d["filename"],
                "Pages": d.get("page_count", "N/A"),
                "Chunks": d.get("chunk_count", "N/A"),
                "Size (KB)": round(d.get("file_size_bytes", 0) / 1024, 1),
                "Indexed Date": d.get("indexed_at", "")[:10],
            })
        st.dataframe(pd.DataFrame(doc_rows), use_container_width=True)

        st.markdown("##### Manage")
        c_sel, c_del, c_prev = st.columns([2, 1, 1])
        with c_sel:
            selected_doc_action = st.selectbox("Choose document:", [d["filename"] for d in all_docs], label_visibility="collapsed")
        with c_del:
            if st.button("🗑️ Delete Document", use_container_width=True):
                deleted = document_manager.delete_document(
                    filename=selected_doc_action,
                    delete_vectors_fn=delete_document_vectors,
                )
                if deleted:
                    st.success(f"Deleted `{selected_doc_action}`.")
                    st.rerun()
        with c_prev:
            preview_toggle = st.checkbox("Preview Text Chunks", key="doc_prev_toggle")

        if preview_toggle:
            preview_path = UPLOAD_DIR / selected_doc_action
            if preview_path.exists():
                try:
                    p_chunks = process_pdf(str(preview_path))
                    st.markdown(f"**Extracted Text Passages for `{selected_doc_action}`:**")
                    for i, ch in enumerate(p_chunks[:3]):
                        with st.expander(f"Chunk {i + 1} • Page {ch.metadata.get('page', 1)}"):
                            st.write(ch.page_content)
                except Exception as exc:
                    st.error(f"Cannot preview: {exc}")


# =============================================================================
# VISUAL STUDIO
# =============================================================================
elif page == "Studio":
    page_header("VISUAL STUDIO", "Posters, brochures, diagrams and data visuals from your documents.")

    c_col1, c_col2 = st.columns([1, 2], gap="large")

    with c_col1:
        with st.container(border=True):
            st.markdown(card_title("1 · Source Scope"), unsafe_allow_html=True)
            doc_opts = [d["filename"] for d in all_docs]

            if not doc_opts:
                st.warning("Please upload a PDF first in the Document Hub.")
                create_source_docs = None
            else:
                create_source_docs = st.multiselect(
                    "Source Document(s):",
                    options=doc_opts,
                    default=doc_opts[:1],
                    key="create_scope_docs",
                    label_visibility="collapsed",
                )

            st.markdown(card_title("2 · Asset Template"), unsafe_allow_html=True)
            category_options = ["Posters", "Brochures", "Presentations & Diagrams", "Data & Tables"]
            sel_cat = st.selectbox("Category:", category_options, label_visibility="collapsed")

            cat_templates = {k: v for k, v in GENERATION_TEMPLATES.items() if v["category"] == sel_cat}
            sel_asset_key = st.selectbox(
                "Format:",
                options=list(cat_templates.keys()),
                format_func=lambda k: f"{cat_templates[k]['icon']} {cat_templates[k]['title']}",
                label_visibility="collapsed",
            )
            st.caption(cat_templates[sel_asset_key]["description"])

            st.markdown(card_title("3 · Customization"), unsafe_allow_html=True)
            custom_title_input = st.text_input("Asset Title:", value=cat_templates[sel_asset_key]["title"])
            sel_tone = st.selectbox("Tone:", ["Professional", "Executive Brief", "Academic", "Engaging & Creative"])
            sel_palette = st.selectbox(
                "Color Palette:",
                list(STUDIO_PALETTES.keys()),
                format_func=lambda k: k.replace("_", " ").title(),
            )
            focus_notes = st.text_area(
                "Key Points to Highlight:",
                placeholder="e.g. Highlight user experience results and key features...",
                height=80,
            )

            generate_visual_btn = st.button(
                "🚀 Generate Studio Asset", type="primary", use_container_width=True, disabled=not doc_opts
            )

    with c_col2:
        if generate_visual_btn and create_source_docs:
            with st.spinner(f"Synthesizing {cat_templates[sel_asset_key]['title']} with visual graphics..."):
                gen_result = generate_content_asset(
                    asset_type=sel_asset_key,
                    selected_sources=create_source_docs,
                    custom_title=custom_title_input,
                    tone=sel_tone,
                    specific_focus=focus_notes,
                    k=8,
                )
                gen_result["palette"] = sel_palette
                st.session_state.created_asset = gen_result
                st.session_state.current_image = None

        if st.session_state.created_asset:
            asset = st.session_state.created_asset
            st.success(f"Generated: **{asset['title']}** ({asset['category']})")

            tab_v_preview, tab_v_doc, tab_v_edit = st.tabs(
                ["🎨 Visual Gallery", "📑 Document View", "📝 Editor"]
            )

            with tab_v_preview:
                if asset["category"] == "Posters":
                    sections = parse_sections_from_markdown(asset["content"])
                    poster_img = generate_poster_image(
                        title=asset["title"],
                        sections=sections,
                        palette_name=asset.get("palette", "luxury_gold"),
                    )
                    st.image(poster_img, caption="Visual Poster (CPU Rendered)", use_container_width=True)
                    st.session_state.current_image = poster_img

                elif asset["category"] == "Brochures":
                    panels = parse_panels_from_markdown(asset["content"])
                    brochure_img = generate_brochure_image(
                        title=asset["title"],
                        panels=panels,
                        palette_name=asset.get("palette", "luxury_gold"),
                    )
                    st.image(brochure_img, caption="Tri-Fold Brochure Layout (CPU Rendered)", use_container_width=True)
                    st.session_state.current_image = brochure_img

                else:
                    chart_dict = parse_chart_data(asset["content"])
                    if chart_dict:
                        chart_img = generate_chart_image(
                            chart_type=chart_dict.get("chart_type", "bar"),
                            labels=chart_dict.get("labels", ["A", "B", "C"]),
                            values=chart_dict.get("values", [10, 20, 30]),
                            title=chart_dict.get("title", asset["title"]),
                            unit=chart_dict.get("unit", ""),
                            palette_name=asset.get("palette", "luxury_gold"),
                        )
                        st.image(chart_img, caption="High-Res Infographic Chart", use_container_width=True)
                        st.session_state.current_image = chart_img
                    else:
                        st.markdown(asset["content"])

            with tab_v_doc:
                st.markdown(
                    _h(
                        f"""
                        <div class="exec-box">
                          <div class="exec-kicker">Executive Briefing</div>
                          <h2 style="margin: 0 0 10px 0;">{html.escape(asset['title'])}</h2>
                          <div class="exec-meta">Source: {html.escape(', '.join(asset.get('sources', [])))} • Verified Local Synthesis</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(asset["content"])

            with tab_v_edit:
                edited_content = st.text_area(
                    "Fine-tune content before download:",
                    value=asset["content"],
                    height=450,
                    key="v_editor",
                )
                asset["content"] = edited_content

            st.markdown("##### 📥 Export Formats")
            d1, d2, d3, d4, d5 = st.columns(5)
            base_name = asset["title"].replace(" ", "_").lower()

            with d1:
                st.download_button(
                    "📄 PDF",
                    data=export_to_pdf(asset["title"], asset["content"]),
                    file_name=f"{base_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with d2:
                st.download_button(
                    "📝 Word",
                    data=export_to_docx(asset["title"], asset["content"]),
                    file_name=f"{base_name}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            with d3:
                st.download_button(
                    "📑 Markdown",
                    data=export_to_markdown(asset["title"], asset["content"]),
                    file_name=f"{base_name}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with d4:
                st.download_button(
                    "📋 Text",
                    data=export_to_txt(asset["title"], asset["content"]),
                    file_name=f"{base_name}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with d5:
                if st.session_state.get("current_image"):
                    st.download_button(
                        "🖼️ PNG",
                        data=export_image_to_png(st.session_state.current_image),
                        file_name=f"{base_name}.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                else:
                    st.download_button(
                        "📊 CSV",
                        data=export_to_csv(asset["content"]),
                        file_name=f"{base_name}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        else:
            st.info("Select a template on the left and click **🚀 Generate Studio Asset**.")


# =============================================================================
# EXECUTIVE REPORTS
# =============================================================================
elif page == "Reports":
    page_header("EXECUTIVE REPORTS", "Turn your documents into polished, grounded reports.")

    rep_col1, rep_col2 = st.columns([1, 2], gap="large")

    with rep_col1:
        with st.container(border=True):
            doc_opts = [d["filename"] for d in all_docs]

            st.markdown(card_title("1 · Source Scope"), unsafe_allow_html=True)
            rep_scope = st.multiselect(
                "Source Document(s):",
                options=doc_opts,
                default=doc_opts[:1] if doc_opts else None,
                key="rep_scope_select",
                label_visibility="collapsed",
            )

            st.markdown(card_title("2 · Report Type"), unsafe_allow_html=True)
            rep_types = {
                "executive_summary": "📊 Executive Summary Brief",
                "research_summary": "🔬 Academic Research Synthesis",
                "project_documentation": "📝 Technical System Documentation",
                "study_notes": "📖 Study Notes & Review Guide",
                "project_proposal": "💼 Business & Project Proposal",
            }
            sel_rep_type = st.selectbox(
                "Format:",
                options=list(rep_types.keys()),
                format_func=lambda k: rep_types[k],
                label_visibility="collapsed",
            )

            st.markdown(card_title("3 · Customization"), unsafe_allow_html=True)
            rep_title = st.text_input("Report Title:", value=rep_types[sel_rep_type].split(" ", 1)[1])
            rep_tone = st.selectbox("Tone:", ["Executive & High-Impact", "Formal & Professional", "Academic & Rigorous"])
            rep_focus_text = st.text_area(
                "Key Focus / Required Sections:",
                placeholder="e.g. Include problem context, technical specifications, and key results...",
                height=90,
            )

            gen_rep_btn = st.button(
                "Generate Executive Report", type="primary", use_container_width=True, disabled=not doc_opts
            )

    with rep_col2:
        if gen_rep_btn and rep_scope:
            with st.spinner("Synthesizing executive report..."):
                rep_res = generate_content_asset(
                    asset_type=sel_rep_type,
                    selected_sources=rep_scope,
                    custom_title=rep_title,
                    tone=rep_tone,
                    specific_focus=rep_focus_text,
                    k=10,
                )
                st.session_state.report_asset = rep_res

        if st.session_state.report_asset:
            r_asset = st.session_state.report_asset
            st.success(f"Generated: **{r_asset['title']}**")

            st.markdown(
                _h(
                    f"""
                    <div class="exec-box">
                      <div class="exec-kicker">Executive Synthesis</div>
                      <h2 style="margin: 0 0 10px 0;">{html.escape(r_asset['title'])}</h2>
                      <span class="citation-badge">📄 {html.escape(', '.join(r_asset.get('sources', [])))}</span>
                      <span class="citation-badge citation-badge-page">✓ 100% Grounded</span>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

            r_tab_view, r_tab_edit = st.tabs(["📑 Report Document", "📝 Report Editor"])

            with r_tab_view:
                st.markdown(r_asset["content"])

            with r_tab_edit:
                r_edit_val = st.text_area("Edit Report Draft:", value=r_asset["content"], height=500, key="r_editor_area")
                r_asset["content"] = r_edit_val

            st.markdown("##### 📥 Export Report")
            rc1, rc2, rc3, rc4 = st.columns(4)
            r_base = r_asset["title"].replace(" ", "_").lower()

            with rc1:
                st.download_button(
                    "📄 PDF",
                    data=export_to_pdf(r_asset["title"], r_asset["content"]),
                    file_name=f"{r_base}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with rc2:
                st.download_button(
                    "📝 Word",
                    data=export_to_docx(r_asset["title"], r_asset["content"]),
                    file_name=f"{r_base}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            with rc3:
                st.download_button(
                    "📑 Markdown",
                    data=export_to_markdown(r_asset["title"], r_asset["content"]),
                    file_name=f"{r_base}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with rc4:
                st.download_button(
                    "📋 Text",
                    data=export_to_txt(r_asset["title"], r_asset["content"]),
                    file_name=f"{r_base}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        else:
            st.info("Configure your report on the left and click **Generate Executive Report**.")


# =============================================================================
# BENCHMARK & QUALITY
# =============================================================================
elif page == "Benchmark":
    page_header("QUALITY & BENCHMARK", "Measure retrieval accuracy, latency and indexing speed.")

    eval_tab_mode = st.radio(
        "Evaluation Mode:",
        ["Automated Benchmark Suite", "Single Query Diagnostic", "PDF Indexing Speed"],
        horizontal=True,
        label_visibility="collapsed",
        key="eval_mode",
    )

    st.markdown("---")

    if eval_tab_mode == "Automated Benchmark Suite":
        st.markdown("##### Automated Benchmark Question Suite")
        st.caption("Runs standard evaluation queries against indexed documents to compute Mean Reciprocal Rank (MRR) and Hit Rates.")

        with st.expander("View Benchmark Test Cases"):
            st.dataframe(pd.DataFrame(DEFAULT_BENCHMARK_SUITE), use_container_width=True)

        if st.button("▶️ Run Automated Benchmark", type="primary", use_container_width=True):
            with st.spinner("Executing benchmark queries..."):
                bench_res = run_benchmark_suite(DEFAULT_BENCHMARK_SUITE, k=4)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Queries Tested", bench_res["total_queries"])
            m2.metric("Hit Rate @ k", f"{bench_res['hit_rate_pct']}%")
            m3.metric("Mean Reciprocal Rank", bench_res["mean_reciprocal_rank"])
            m4.metric("Avg Latency", f"{int(bench_res['avg_retrieval_latency_sec'] * 1000)} ms")

            df_b = pd.DataFrame(bench_res["query_results"])
            st.dataframe(df_b, use_container_width=True)

            st.download_button(
                "📥 Download Evaluation Report (.csv)",
                data=df_b.to_csv(index=False),
                file_name=f"rag_benchmark_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    elif eval_tab_mode == "Single Query Diagnostic":
        st.markdown("##### Live Query Diagnostic")
        test_q = st.text_input("Test Question:", value="What is the design process in the CareConnect study?")
        test_kw = st.text_input("Expected Keyword:", value="design")

        if st.button("Evaluate Query", type="primary"):
            with st.spinner("Testing retrieval and generation..."):
                eval_out = evaluate_question(test_q, expected_keyword=test_kw, k=4)

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Evidence Found", "Yes ✅" if eval_out["evidence_found"] else "No ❌")
            q2.metric("Reciprocal Rank", eval_out["reciprocal_rank"])
            q3.metric("Retrieval Latency", f"{eval_out['retrieval_time_seconds']}s")
            q4.metric("Avg Similarity", eval_out["average_relevance_score"])

            st.markdown("###### Generated Answer")
            st.info(eval_out["answer"])

    elif eval_tab_mode == "PDF Indexing Speed":
        st.markdown("##### PDF Indexing Performance")
        all_docs = document_manager.get_all_documents()
        if not all_docs:
            st.info("No documents available.")
        else:
            bench_file = st.selectbox("Select document:", [d["filename"] for d in all_docs])
            if st.button("Benchmark Speed", type="primary"):
                p_path = UPLOAD_DIR / bench_file
                if p_path.exists():
                    with st.spinner("Benchmarking PDF parse & chunk performance..."):
                        p_perf = evaluate_indexing_performance(str(p_path))

                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Total Pages", p_perf["total_pages"])
                    p2.metric("Total Chunks", p_perf["total_chunks"])
                    p3.metric("Chunking Time", f"{p_perf['chunking_time_sec']}s")
                    p4.metric("Throughput", f"{p_perf['characters_per_second']} chars/sec")