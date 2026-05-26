"""
Multimodal Document Analyzer - Production-Ready Streamlit App
=====================================================================
A professional AI-powered document intelligence platform using Google Gemini.
Supports PDFs, DOCX, TXT, Images, and Scanned Documents.

Author: Generated for deployment on Streamlit Community Cloud
"""

import streamlit as st
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# ─── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="DocuMind AI — Document Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/multimodal-document-analyzer",
        "Report a bug": "https://github.com/your-repo/multimodal-document-analyzer/issues",
        "About": "DocuMind AI — Powered by Google Gemini"
    }
)

# ─── Path Setup ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
UTILS_DIR = ROOT_DIR / "utils"
TEMP_DIR = ROOT_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))

# ─── Utility Imports ──────────────────────────────────────────────────────────
from utils.security import validate_api_key, sanitize_text, validate_file
from utils.pdf_utils import extract_text_from_pdf, extract_tables_from_pdf
from utils.image_utils import extract_text_from_image, caption_image
from utils.ai_utils import (
    get_gemini_client, summarize_document, answer_question,
    extract_entities, classify_document, analyze_sentiment,
    generate_study_notes, generate_flashcards, generate_quiz,
    compare_documents, detect_language
)
from utils.embeddings import DocumentEmbedder
from utils.export_utils import export_to_pdf, export_to_txt
from utils.ocr_utils import ocr_image

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Root Variables ── */
:root {
    --primary: #6C63FF;
    --primary-light: #8B85FF;
    --primary-dark: #4B44CC;
    --accent: #FF6584;
    --accent2: #43E6B8;
    --bg-dark: #0D0F1A;
    --bg-card: #161829;
    --bg-card2: #1E2035;
    --text-primary: #F0F2FF;
    --text-secondary: #9BA3CF;
    --text-muted: #5C638A;
    --border: #2A2D45;
    --success: #43E6B8;
    --warning: #FFB547;
    --error: #FF6584;
    --gradient: linear-gradient(135deg, #6C63FF 0%, #FF6584 100%);
    --gradient2: linear-gradient(135deg, #43E6B8 0%, #6C63FF 100%);
}

/* ── Global Reset ── */
html, body, .stApp {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text-primary);
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0F1A 0%, #111326 100%);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-secondary);
    border-radius: 10px;
    padding: 10px 16px;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    transition: all 0.2s ease;
    margin-bottom: 4px;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--bg-card2);
    border-color: var(--border);
    color: var(--text-primary);
}

/* ── Cards ── */
.doc-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.doc-card:hover {
    border-color: var(--primary);
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.15);
}

/* ── Metric Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.metric-card .metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-card .metric-label {
    color: var(--text-secondary);
    font-size: 13px;
    margin-top: 4px;
}

/* ── Chat Messages ── */
.chat-msg-user {
    background: linear-gradient(135deg, var(--primary-dark), var(--primary));
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    margin: 8px 0;
    margin-left: 20%;
    color: white;
    font-size: 14px;
    line-height: 1.6;
}

.chat-msg-ai {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    margin-right: 20%;
    color: var(--text-primary);
    font-size: 14px;
    line-height: 1.6;
}

.chat-citation {
    background: rgba(108, 99, 255, 0.1);
    border-left: 3px solid var(--primary);
    border-radius: 0 8px 8px 0;
    padding: 8px 12px;
    margin-top: 10px;
    font-size: 12px;
    color: var(--text-secondary);
}

/* ── Upload Area ── */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 48px;
    text-align: center;
    background: rgba(108, 99, 255, 0.03);
    transition: all 0.3s ease;
}

.upload-zone:hover {
    border-color: var(--primary);
    background: rgba(108, 99, 255, 0.07);
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.badge-pdf { background: rgba(255, 101, 132, 0.15); color: #FF6584; }
.badge-docx { background: rgba(67, 230, 184, 0.15); color: #43E6B8; }
.badge-txt { background: rgba(108, 99, 255, 0.15); color: #8B85FF; }
.badge-img { background: rgba(255, 181, 71, 0.15); color: #FFB547; }

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
}

.section-icon {
    font-size: 28px;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    margin: 0;
}

/* ── Insight Cards ── */
.insight-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}

.insight-card .insight-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

/* ── Flashcard ── */
.flashcard {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-card2));
    border: 1px solid var(--primary);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    min-height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    cursor: pointer;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.flashcard:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.25);
}

/* ── Progress Bar ── */
.custom-progress {
    background: var(--bg-card2);
    border-radius: 8px;
    height: 6px;
    overflow: hidden;
    margin: 8px 0;
}

.custom-progress-bar {
    height: 100%;
    border-radius: 8px;
    background: var(--gradient);
    transition: width 0.5s ease;
}

/* ── Logo ── */
.logo-text {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: var(--gradient) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s ease, transform 0.1s ease !important;
}

.stButton > button[kind="primary"]:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 6px;
    gap: 4px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: var(--text-secondary);
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    padding: 8px 20px;
}

.stTabs [aria-selected="true"] {
    background: var(--gradient) !important;
    color: white !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: var(--bg-card);
    border-radius: 10px;
    border: 1px solid var(--border);
    color: var(--text-primary);
    font-family: 'DM Sans', sans-serif;
}

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Typing animation ── */
@keyframes typing {
    0%, 100% { opacity: 0.3; transform: translateY(0px); }
    50% { opacity: 1; transform: translateY(-4px); }
}

.typing-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--primary);
    margin: 0 2px;
    animation: typing 1.2s infinite;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

/* ── Gradient border card ── */
.gradient-card {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 2px;
    background: linear-gradient(var(--bg-card), var(--bg-card)) padding-box,
                var(--gradient) border-box;
    border: 2px solid transparent;
    margin-bottom: 16px;
}

.gradient-card-inner {
    background: var(--bg-card);
    border-radius: 14px;
    padding: 20px;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(108,99,255,0.15) 0%, rgba(255,101,132,0.1) 100%);
    border: 1px solid rgba(108, 99, 255, 0.3);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 50% 50%, rgba(108,99,255,0.08), transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 42px;
    font-weight: 800;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
    line-height: 1.2;
}

.hero-subtitle {
    color: var(--text-secondary);
    font-size: 17px;
    max-width: 540px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Tag pills ── */
.tag-pill {
    display: inline-block;
    background: rgba(108,99,255,0.12);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: var(--primary-light);
    margin: 3px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Alert boxes ── */
.alert-info {
    background: rgba(108,99,255,0.1);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    color: var(--primary-light);
    font-size: 14px;
}

.alert-success {
    background: rgba(67,230,184,0.1);
    border: 1px solid rgba(67,230,184,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    color: var(--accent2);
    font-size: 14px;
}

.alert-warning {
    background: rgba(255,181,71,0.1);
    border: 1px solid rgba(255,181,71,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    color: var(--warning);
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ─────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "page": "Home",
        "uploaded_docs": {},          # {filename: {text, type, size, pages, tables}}
        "chat_history": [],           # [{role, content, citations, timestamp}]
        "embedder": None,
        "processing_history": [],     # [{filename, action, result, timestamp}]
        "api_key": None,
        "use_own_key": False,
        "current_summary": {},
        "flashcards": [],
        "quiz_questions": [],
        "study_notes": {},
        "settings": {
            "model": "gemini-3.1-flash-lite",
            "max_tokens": 2048,
            "temperature": 0.7,
            "chunk_size": 3000,
            "ocr_enabled": True,
            "language": "auto",
        },
        "token_usage": 0,
        "last_analysis": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ─── API Key Resolution ────────────────────────────────────────────────────────
def get_active_api_key():
    """
    Resolve the active API key:
    1. User's own key (if provided)
    2. Streamlit secrets (GEMINI_API_KEY)
    3. Environment variable
    """
    if st.session_state.use_own_key and st.session_state.api_key:
        return st.session_state.api_key
    # Try Streamlit secrets
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    # Try env variable
    return os.environ.get("GEMINI_API_KEY", "")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo & Branding
        st.markdown("""
        <div style='padding: 8px 0 24px 0; border-bottom: 1px solid var(--border); margin-bottom: 24px;'>
            <div class='logo-text'>🧠 DocuMind AI</div>
            <div style='color: var(--text-muted); font-size: 12px; margin-top: 4px;'>
                Document Intelligence Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        pages = [
            ("🏠", "Home", "home"),
            ("📂", "Upload Documents", "upload"),
            ("💬", "AI Chat", "chat"),
            ("⚖️", "Compare Documents", "compare"),
            ("📊", "Analytics", "analytics"),
            ("📚", "Study Tools", "study"),
            ("⚙️", "Settings", "settings"),
        ]

        for icon, label, key in pages:
            active = st.session_state.page == label
            btn_style = "primary" if active else "secondary"
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.page = label
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # Document Count Badge
        doc_count = len(st.session_state.uploaded_docs)
        if doc_count > 0:
            st.markdown(f"""
            <div style='background: rgba(108,99,255,0.1); border: 1px solid rgba(108,99,255,0.25);
                        border-radius: 10px; padding: 12px; margin-bottom: 16px;'>
                <div style='font-size: 11px; color: var(--text-muted); text-transform: uppercase;
                           letter-spacing: 1px; margin-bottom: 4px;'>Loaded Documents</div>
                <div style='font-family: Syne, sans-serif; font-size: 24px; font-weight: 800;
                           color: var(--primary-light);'>{doc_count}</div>
            </div>
            """, unsafe_allow_html=True)

            for fname, doc in list(st.session_state.uploaded_docs.items()):
                ext = doc["type"].upper()
                badge_class = f"badge-{doc['type']}"
                st.markdown(f"""
                <div style='display: flex; align-items: center; gap: 8px; padding: 6px 0;
                           border-bottom: 1px solid rgba(42,45,69,0.5);'>
                    <span class='badge {badge_class}'>{ext}</span>
                    <span style='font-size: 12px; color: var(--text-secondary);
                                 overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
                                 max-width: 140px;' title='{fname}'>{fname}</span>
                </div>
                """, unsafe_allow_html=True)

        # API Key Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size: 11px; color: var(--text-muted); text-transform: uppercase;
                   letter-spacing: 1px; margin-bottom: 8px;'>API Configuration</div>
        """, unsafe_allow_html=True)

        use_own = st.toggle("Use My Own API Key", value=st.session_state.use_own_key)
        st.session_state.use_own_key = use_own

        if use_own:
            key_input = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="AIza...",
                value=st.session_state.api_key or ""
            )
            if key_input:
                if validate_api_key(key_input):
                    st.session_state.api_key = key_input
                    st.markdown('<div class="alert-success" style="font-size:12px;">✓ Key accepted</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-warning" style="font-size:12px;">⚠ Invalid key format</div>',
                                unsafe_allow_html=True)

        # Token Usage
        if st.session_state.token_usage > 0:
            st.markdown(f"""
            <div style='margin-top: 12px; font-size: 12px; color: var(--text-muted);'>
                🔢 Tokens used: <span style='color: var(--primary-light);'>
                {st.session_state.token_usage:,}</span>
            </div>
            """, unsafe_allow_html=True)

        # Version Footer
        st.markdown("""
        <div style='position: absolute; bottom: 20px; left: 16px; right: 16px;
                    font-size: 11px; color: var(--text-muted); text-align: center;'>
            DocuMind AI v1.0 · Powered by Gemini
        </div>
        """, unsafe_allow_html=True)


# ─── Page: Home ───────────────────────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>Document Intelligence, Reimagined</div>
        <div class='hero-subtitle'>
            Upload any document — PDF, DOCX, image, or scanned page —
            and let AI extract insights, answer questions, and generate study materials instantly.
        </div>
        <br>
        <div>
            <span class='tag-pill'>📄 PDFs & DOCX</span>
            <span class='tag-pill'>🖼 Images & Scans</span>
            <span class='tag-pill'>💬 AI Chat</span>
            <span class='tag-pill'>📊 Analytics</span>
            <span class='tag-pill'>📚 Study Tools</span>
            <span class='tag-pill'>🔍 RAG Search</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Grid
    features = [
        ("🔍", "Smart Document Analysis", "Extract text, tables, entities, and key insights from any document format automatically."),
        ("💬", "AI-Powered Q&A", "Ask questions and get cited answers drawn directly from your uploaded documents."),
        ("⚖️", "Multi-Document Comparison", "Compare themes, tone, and content across multiple documents side by side."),
        ("📚", "Study Tools", "Generate flashcards, quizzes, and structured study notes from any document instantly."),
        ("🧠", "RAG Search", "Semantic vector search finds relevant passages even when exact keywords aren't present."),
        ("📤", "Export & Share", "Download analysis results as PDF or TXT reports for easy sharing."),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='doc-card'>
                <div style='font-size: 32px; margin-bottom: 12px;'>{icon}</div>
                <div style='font-family: Syne, sans-serif; font-weight: 700; font-size: 16px;
                           margin-bottom: 8px;'>{title}</div>
                <div style='color: var(--text-secondary); font-size: 14px; line-height: 1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Quick Start
    st.markdown("---")
    st.markdown("""
    <div class='section-header'>
        <span class='section-icon'>⚡</span>
        <div class='section-title'>Quick Start</div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("1", "Upload Documents", "Drag & drop PDFs, DOCX, images, or scanned pages"),
        ("2", "AI Processes Files", "Text extraction, OCR, and vector embedding happen automatically"),
        ("3", "Ask Questions", "Chat with your documents using natural language"),
        ("4", "Export Insights", "Download summaries, flashcards, or full analysis reports"),
    ]

    cols2 = st.columns(4)
    for i, (num, title, desc) in enumerate(steps):
        with cols2[i]:
            st.markdown(f"""
            <div style='text-align: center; padding: 20px 10px;'>
                <div style='width: 48px; height: 48px; border-radius: 50%;
                           background: var(--gradient); display: flex; align-items: center;
                           justify-content: center; font-family: Syne, sans-serif;
                           font-weight: 800; font-size: 18px; color: white;
                           margin: 0 auto 16px auto;'>{num}</div>
                <div style='font-family: Syne, sans-serif; font-weight: 700;
                           font-size: 15px; margin-bottom: 8px;'>{title}</div>
                <div style='color: var(--text-secondary); font-size: 13px;
                           line-height: 1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Recent Processing History
    if st.session_state.processing_history:
        st.markdown("---")
        st.markdown("### 📋 Recent Activity")
        for item in reversed(st.session_state.processing_history[-5:]):
            st.markdown(f"""
            <div style='display: flex; align-items: center; gap: 12px; padding: 10px 0;
                       border-bottom: 1px solid var(--border); font-size: 14px;'>
                <span style='color: var(--text-muted); font-size: 12px; min-width: 80px;'>
                    {item.get('timestamp', '')[:16]}
                </span>
                <span style='color: var(--primary-light);'>{item.get('action', '')}</span>
                <span style='color: var(--text-secondary);'>{item.get('filename', '')}</span>
            </div>
            """, unsafe_allow_html=True)

    # CTA
    if not st.session_state.uploaded_docs:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📂  Upload Your First Document →", type="primary", use_container_width=True):
                st.session_state.page = "Upload Documents"
                st.rerun()


# ─── Page: Upload Documents ───────────────────────────────────────────────────
def page_upload():
    st.markdown("""
    <div class='section-header'>
        <span class='section-icon'>📂</span>
        <div class='section-title'>Upload Documents</div>
    </div>
    """, unsafe_allow_html=True)

    api_key = get_active_api_key()

    # Upload Zone
    uploaded_files = st.file_uploader(
        "Drop files here — PDFs, DOCX, TXT, JPG, PNG, JPEG",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Maximum 50MB per file. All processing happens securely."
    )

    if uploaded_files:
        process_btn = st.button("🚀  Process All Documents", type="primary")

        for file in uploaded_files:
            # File Info Preview
            ext = file.name.rsplit(".", 1)[-1].lower()
            size_kb = file.size / 1024
            badge_class = f"badge-{ext if ext in ['pdf','docx','txt'] else 'img'}"
            ext_display = ext if ext in ['pdf','docx','txt'] else 'img'

            st.markdown(f"""
            <div class='doc-card'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <span class='badge {badge_class}'>{ext.upper()}</span>
                        <span style='margin-left: 10px; font-weight: 500;'>{file.name}</span>
                    </div>
                    <div style='color: var(--text-muted); font-size: 13px;'>
                        {size_kb:.1f} KB
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if process_btn:
            if not api_key:
                st.error("⚠️ No Gemini API key configured. Add one in Settings or the sidebar.")
                return

            progress_bar = st.progress(0, text="Starting processing...")
            total = len(uploaded_files)

            for i, file in enumerate(uploaded_files):
                fname = file.name
                progress_bar.progress((i) / total, text=f"Processing {fname}...")

                # Security validation
                is_valid, err_msg = validate_file(file)
                if not is_valid:
                    st.error(f"❌ {fname}: {err_msg}")
                    continue

                ext = fname.rsplit(".", 1)[-1].lower()
                doc_data = {
                    "type": ext if ext in ["pdf", "docx", "txt"] else "img",
                    "size": file.size,
                    "filename": fname,
                    "text": "",
                    "tables": [],
                    "pages": 1,
                    "language": "en",
                    "timestamp": datetime.now().isoformat(),
                }

                try:
                    with st.spinner(f"Extracting text from {fname}..."):
                        file_bytes = file.read()
                        file.seek(0)

                        if ext == "pdf":
                            text, tables, pages = extract_text_from_pdf(file_bytes)
                            doc_data["text"] = sanitize_text(text)
                            doc_data["tables"] = tables
                            doc_data["pages"] = pages

                        elif ext == "docx":
                            from utils.pdf_utils import extract_text_from_docx
                            text = extract_text_from_docx(file_bytes)
                            doc_data["text"] = sanitize_text(text)

                        elif ext == "txt":
                            text = file_bytes.decode("utf-8", errors="ignore")
                            doc_data["text"] = sanitize_text(text)

                        elif ext in ["jpg", "jpeg", "png"]:
                            text = extract_text_from_image(file_bytes,
                                                           ocr_enabled=st.session_state.settings["ocr_enabled"])
                            doc_data["text"] = sanitize_text(text)
                            doc_data["type"] = "img"

                        # Auto-detect language
                        if doc_data["text"] and len(doc_data["text"]) > 50:
                            try:
                                lang = detect_language(doc_data["text"][:500], api_key,
                                                       st.session_state.settings["model"])
                                doc_data["language"] = lang
                            except Exception:
                                pass

                        st.session_state.uploaded_docs[fname] = doc_data

                        # Add to embedder for RAG
                        if st.session_state.embedder is None:
                            st.session_state.embedder = DocumentEmbedder()
                        st.session_state.embedder.add_document(fname, doc_data["text"])

                        # Log to history
                        st.session_state.processing_history.append({
                            "filename": fname,
                            "action": "Document Processed",
                            "timestamp": datetime.now().isoformat(),
                        })

                except Exception as e:
                    st.error(f"❌ Error processing {fname}: {str(e)}")
                    continue

            progress_bar.progress(1.0, text="✅ All documents processed!")
            time.sleep(0.5)
            st.success(f"✅ Successfully processed {len(st.session_state.uploaded_docs)} document(s)!")
            st.rerun()

    # Loaded Documents Display
    if st.session_state.uploaded_docs:
        st.markdown("---")
        st.markdown("### 📋 Loaded Documents")

        for fname, doc in st.session_state.uploaded_docs.items():
            with st.expander(f"📄 {fname}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Type", doc["type"].upper())
                with col2:
                    st.metric("Size", f"{doc['size']/1024:.1f} KB")
                with col3:
                    st.metric("Pages", doc.get("pages", 1))
                with col4:
                    st.metric("Language", doc.get("language", "en").upper())

                if doc["text"]:
                    st.markdown("**Preview:**")
                    preview = doc["text"][:600] + ("..." if len(doc["text"]) > 600 else "")
                    st.markdown(f"""
                    <div style='background: var(--bg-card2); border-radius: 10px; padding: 16px;
                               font-size: 13px; color: var(--text-secondary); line-height: 1.7;
                               font-family: monospace;'>{preview}</div>
                    """, unsafe_allow_html=True)

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button(f"🔍 Analyze", key=f"analyze_{fname}"):
                        with st.spinner("Analyzing..."):
                            summary = summarize_document(doc["text"], api_key,
                                                         st.session_state.settings["model"],
                                                         st.session_state.settings["chunk_size"])
                            st.session_state.current_summary[fname] = summary
                            st.session_state.token_usage += len(doc["text"]) // 4
                            st.session_state.processing_history.append({
                                "filename": fname, "action": "AI Analysis",
                                "timestamp": datetime.now().isoformat()
                            })
                        st.success("Analysis complete!")

                    if fname in st.session_state.current_summary:
                        st.info(st.session_state.current_summary[fname][:300] + "...")

                with col_b:
                    if st.button(f"🗑 Remove", key=f"remove_{fname}"):
                        del st.session_state.uploaded_docs[fname]
                        if fname in st.session_state.current_summary:
                            del st.session_state.current_summary[fname]
                        st.rerun()

                with col_c:
                    if doc.get("tables"):
                        st.caption(f"📊 {len(doc['tables'])} table(s) found")


# ─── Page: AI Chat ────────────────────────────────────────────────────────────
def page_chat():
    st.markdown("""
    <div class='section-header'>
        <span class='section-icon'>💬</span>
        <div class='section-title'>AI Chat</div>
    </div>
    """, unsafe_allow_html=True)

    api_key = get_active_api_key()

    if not st.session_state.uploaded_docs:
        st.markdown("""
        <div class='alert-warning'>
            ⚠️ No documents loaded. Please upload documents first before chatting.
        </div>
        """, unsafe_allow_html=True)
        if st.button("📂 Go to Upload"):
            st.session_state.page = "Upload Documents"
            st.rerun()
        return

    if not api_key:
        st.error("⚠️ Configure your Gemini API key in the sidebar or Settings.")
        return

    # Chat Options Bar
    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        selected_docs = st.multiselect(
            "Search in",
            list(st.session_state.uploaded_docs.keys()),
            default=list(st.session_state.uploaded_docs.keys()),
            label_visibility="collapsed"
        )
    with col3:
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Chat History Display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style='text-align: center; padding: 60px 20px; color: var(--text-muted);'>
                <div style='font-size: 48px; margin-bottom: 16px;'>💬</div>
                <div style='font-family: Syne, sans-serif; font-size: 18px; color: var(--text-secondary);
                           margin-bottom: 8px;'>Start a conversation</div>
                <div style='font-size: 14px;'>Ask anything about your uploaded documents</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class='chat-msg-user'>
                        <div style='font-size: 11px; opacity: 0.7; margin-bottom: 6px;'>You</div>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    citation_html = ""
                    if msg.get("citations"):
                        citation_html = f"""
                        <div class='chat-citation'>
                            📎 Sources: {', '.join(msg['citations'])}
                        </div>"""
                    st.markdown(f"""
                    <div class='chat-msg-ai'>
                        <div style='font-size: 11px; color: var(--primary-light); margin-bottom: 6px;'>
                            🧠 DocuMind AI
                        </div>
                        {msg['content'].replace(chr(10), '<br>')}
                        {citation_html}
                    </div>
                    """, unsafe_allow_html=True)

    # Suggested Questions
    if not st.session_state.chat_history:
        st.markdown("**Suggested questions:**")
        suggestions = [
            "Summarize the main points of all documents",
            "What are the key entities mentioned?",
            "What is the overall sentiment/tone?",
            "List any important dates or numbers",
            "Classify the document type and purpose",
        ]
        cols = st.columns(len(suggestions))
        for i, q in enumerate(suggestions):
            with cols[i]:
                if st.button(q, key=f"suggest_{i}"):
                    process_chat_message(q, selected_docs or list(st.session_state.uploaded_docs.keys()), api_key)
                    st.rerun()

    # Input
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([5, 1])
        with col_inp:
            user_input = st.text_input(
                "Message",
                placeholder="Ask anything about your documents...",
                label_visibility="collapsed"
            )
        with col_btn:
            send = st.form_submit_button("Send →", type="primary", use_container_width=True)

    if send and user_input.strip():
        target_docs = selected_docs or list(st.session_state.uploaded_docs.keys())
        process_chat_message(user_input.strip(), target_docs, api_key)
        st.rerun()


def process_chat_message(question: str, doc_names: list, api_key: str):
    """Process a chat message and get AI response with citations."""
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now().isoformat(),
    })

    # Gather context via RAG if embedder is available
    context_parts = []
    citations = []

    if st.session_state.embedder and doc_names:
        try:
            results = st.session_state.embedder.search(question, top_k=3)
            for res in results:
                if res["doc_name"] in doc_names:
                    context_parts.append(f"[From: {res['doc_name']}]\n{res['chunk']}")
                    if res["doc_name"] not in citations:
                        citations.append(res["doc_name"])
        except Exception:
            pass

    # Fallback: concatenate full document text (chunked)
    if not context_parts:
        for dname in doc_names:
            if dname in st.session_state.uploaded_docs:
                text = st.session_state.uploaded_docs[dname]["text"]
                context_parts.append(f"[Document: {dname}]\n{text[:2000]}")
                citations.append(dname)

    context = "\n\n---\n\n".join(context_parts)

    # Build conversation history for multi-turn
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history[:-1]
    ]

    try:
        answer = answer_question(
            question=question,
            context=context,
            history=history,
            api_key=api_key,
            model=st.session_state.settings["model"],
        )
        st.session_state.token_usage += (len(context) + len(answer)) // 4
    except Exception as e:
        answer = f"❌ Error getting AI response: {str(e)}"
        citations = []

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "timestamp": datetime.now().isoformat(),
    })

    st.session_state.processing_history.append({
        "filename": ", ".join(citations),
        "action": "AI Q&A",
        "timestamp": datetime.now().isoformat(),
    })


# ─── Page: Compare Documents ──────────────────────────────────────────────────
def page_compare():
    st.markdown("""
    <div class='section-header'>
        <span class='section-icon'>⚖️</span>
        <div class='section-title'>Compare Documents</div>
    </div>
    """, unsafe_allow_html=True)

    api_key = get_active_api_key()
    docs = st.session_state.uploaded_docs

    if len(docs) < 2:
        st.markdown("""
        <div class='alert-warning'>
            ⚠️ Please upload at least 2 documents to use the comparison feature.
        </div>
        """, unsafe_allow_html=True)
        return

    doc_names = list(docs.keys())
    col1, col2 = st.columns(2)
    with col1:
        doc_a = st.selectbox("Document A", doc_names, key="compare_a")
    with col2:
        remaining = [d for d in doc_names if d != doc_a]
        doc_b = st.selectbox("Document B", remaining, key="compare_b")

    if st.button("⚖️  Compare Documents", type="primary"):
        if not api_key:
            st.error("Configure API key first.")
            return

        with st.spinner("🤔 Analyzing and comparing documents..."):
            text_a = docs[doc_a]["text"]
            text_b = docs[doc_b]["text"]

            result = compare_documents(text_a, text_b, doc_a, doc_b, api_key,
                                       st.session_state.settings["model"])
            st.session_state.token_usage += (len(text_a) + len(text_b)) // 4

        st.markdown("### 📊 Comparison Results")

        tabs = st.tabs(["Overview", "Similarities", "Differences", "Recommendation"])

        with tabs[0]:
            st.markdown(f"""
            <div class='doc-card'>
                {result.get('overview', 'No overview generated.').replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

        with tabs[1]:
            similarities = result.get("similarities", [])
            if similarities:
                for sim in similarities:
                    st.markdown(f"""
                    <div class='insight-card'>
                        <div style='color: var(--accent2); font-size: 14px;'>✓ {sim}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No clear similarities identified.")

        with tabs[2]:
            differences = result.get("differences", [])
            if differences:
                for diff in differences:
                    st.markdown(f"""
                    <div class='insight-card'>
                        <div style='color: var(--accent); font-size: 14px;'>↕ {diff}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant differences identified.")

        with tabs[3]:
            rec = result.get("recommendation", "")
            st.markdown(f"""
            <div class='gradient-card'>
                <div class='gradient-card-inner'>
                    <div style='font-family: Syne, sans-serif; font-weight: 700;
                               margin-bottom: 12px;'>AI Recommendation</div>
                    <div style='color: var(--text-secondary); line-height: 1.7;'>
                        {rec.replace(chr(10), '<br>')}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── Page: Analytics Dashboard ────────────────────────────────────────────────
def page_analytics():
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    st.markdown("""
    <div class='section-header'>
        <span class='section-icon'>📊</span>
        <div class='section-title'>Analytics Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    api_key = get_active_api_key()

    if not st.session_state.uploaded_docs:
        st.markdown('<div class="alert-warning">⚠️ Upload documents to see analytics.</div>',
                    unsafe_allow_html=True)
        return

    docs = st.session_state.uploaded_docs
    doc_names = list(docs.keys())

    # ── Top Metrics ──
    total_docs = len(docs)
    total_chars = sum(len(d["text"]) for d in docs.values())
    total_words = sum(len(d["text"].split()) for d in docs.values())
    total_pages = sum(d.get("pages", 1) for d in docs.values())

    m1, m2, m3, m4 = st.columns(4)
    for col, val, label in zip(
        [m1, m2, m3, m4],
        [total_docs, f"{total_words:,}", f"{total_chars:,}", total_pages],
        ["Documents", "Total Words", "Total Characters", "Total Pages"]
    ):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Document Size Chart
        df_size = pd.DataFrame([
            {"Document": d["filename"][:20], "Words": len(d["text"].split()),
             "Type": d["type"].upper()}
            for d in docs.values()
        ])
        fig1 = px.bar(
            df_size, x="Document", y="Words", color="Type",
            title="Document Word Counts",
            color_discrete_map={"PDF": "#FF6584", "DOCX": "#43E6B8",
                                 "TXT": "#8B85FF", "IMG": "#FFB547"},
            template="plotly_dark"
        )
        fig1.update_layout(
            plot_bgcolor="rgba(22,24,41,1)",
            paper_bgcolor="rgba(22,24,41,1)",
            font=dict(family="DM Sans", color="#9BA3CF"),
            title_font=dict(family="Syne", size=16, color="#F0F2FF"),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Document Type Pie
        type_counts = {}
        for d in docs.values():
            t = d["type"].upper()
            type_counts[t] = type_counts.get(t, 0) + 1

        fig2 = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Document Types",
            color_discrete_sequence=["#FF6584", "#43E6B8", "#8B85FF", "#FFB547"],
            template="plotly_dark"
        )
        fig2.update_layout(
            plot_bgcolor="rgba(22,24,41,1)",
            paper_bgcolor="rgba(22,24,41,1)",
            font=dict(family="DM Sans", color="#9BA3CF"),
            title_font=dict(family="Syne", size=16, color="#F0F2FF"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── AI Insights Section ──
    st.markdown("---")
    selected_doc = st.selectbox("Select document for AI analysis", doc_names)

    if st.button("🔬  Run Deep Analysis", type="primary"):
        if not api_key:
            st.error("Configure API key first.")
            return

        doc = docs[selected_doc]
        text = doc["text"]

        with st.spinner("Running AI analysis..."):
            tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🏷️ Entities", "😊 Sentiment", "🗂 Classification"])

            with tab1:
                summary = summarize_document(text, api_key, st.session_state.settings["model"],
                                             st.session_state.settings["chunk_size"])
                st.markdown(f"""
                <div class='doc-card' style='line-height: 1.8;'>
                    {summary.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

                # Export buttons
                c1, c2 = st.columns(2)
                with c1:
                    txt_bytes = export_to_txt(f"Summary: {selected_doc}\n\n{summary}")
                    st.download_button("📥 Download TXT", txt_bytes, f"summary_{selected_doc}.txt", "text/plain")

            with tab2:
                entities = extract_entities(text, api_key, st.session_state.settings["model"])
                for etype, items in entities.items():
                    if items:
                        st.markdown(f"**{etype}**")
                        pills_html = " ".join([f"<span class='tag-pill'>{item}</span>" for item in items])
                        st.markdown(pills_html, unsafe_allow_html=True)
                        st.markdown("")

            with tab3:
                sentiment = analyze_sentiment(text, api_key, st.session_state.settings["model"])
                score = sentiment.get("score", 0.5)

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score * 100,
                    title={"text": f"Sentiment: {sentiment.get('label', 'Neutral')}",
                           "font": {"family": "Syne", "size": 16, "color": "#F0F2FF"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#5C638A"},
                        "bar": {"color": "#6C63FF"},
                        "bgcolor": "#161829",
                        "bordercolor": "#2A2D45",
                        "steps": [
                            {"range": [0, 33], "color": "rgba(255,101,132,0.3)"},
                            {"range": [33, 66], "color": "rgba(255,181,71,0.3)"},
                            {"range": [66, 100], "color": "rgba(67,230,184,0.3)"},
                        ],
                    },
                    number={"suffix": "%", "font": {"family": "Syne", "color": "#F0F2FF"}}
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(22,24,41,1)",
                    font=dict(family="DM Sans", color="#9BA3CF"),
                    height=300,
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown(f"""
                <div class='insight-card'>
                    <div class='insight-label'>Analysis</div>
                    <div>{sentiment.get('explanation', '')}</div>
                </div>
                """, unsafe_allow_html=True)

            with tab4:
                classification = classify_document(text, api_key, st.session_state.settings["model"])
                st.markdown(f"""
                <div class='gradient-card'>
                    <div class='gradient-card-inner'>
                        <div style='font-size: 11px; color: var(--text-muted); margin-bottom: 8px;
                                   text-transform: uppercase; letter-spacing: 1px;'>Document Type</div>
                        <div style='font-family: Syne, sans-serif; font-size: 24px; font-weight: 700;
                                   color: var(--primary-light);'>{classification.get('type', 'Unknown')}</div>
                        <div style='margin-top: 12px; color: var(--text-secondary);'>
                            {classification.get('description', '')}
                        </div>
                        <div style='margin-top: 16px;'>
                            <span style='font-size: 12px; color: var(--text-muted);'>Confidence: </span>
                            <span style='color: var(--accent2); font-weight: 600;'>
                                {classification.get('confidence', 'N/A')}
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.session_state.token_usage += len(text) // 4


# ─── Page: Study Tools ────────────────────────────────────────────────────────
def page_study():
    st.markdown("""
    <div class='section-header'>
        <span class='section-icon'>📚</span>
        <div class='section-title'>Study Tools</div>
    </div>
    """, unsafe_allow_html=True)

    api_key = get_active_api_key()

    if not st.session_state.uploaded_docs:
        st.markdown('<div class="alert-warning">⚠️ Upload documents to use study tools.</div>',
                    unsafe_allow_html=True)
        return

    if not api_key:
        st.error("Configure API key first.")
        return

    doc_names = list(st.session_state.uploaded_docs.keys())
    selected = st.selectbox("Select Document", doc_names)
    doc_text = st.session_state.uploaded_docs[selected]["text"]

    tabs = st.tabs(["📝 Study Notes", "🃏 Flashcards", "❓ Quiz"])

    with tabs[0]:
        st.markdown("Generate structured study notes from your document.")
        detail = st.select_slider("Detail Level", ["Brief", "Standard", "Comprehensive"], value="Standard")
        if st.button("📝 Generate Study Notes", type="primary", key="gen_notes"):
            with st.spinner("Generating study notes..."):
                notes = generate_study_notes(doc_text, api_key,
                                             st.session_state.settings["model"], detail)
                st.session_state.study_notes[selected] = notes
                st.session_state.token_usage += len(doc_text) // 4

        if selected in st.session_state.study_notes:
            notes = st.session_state.study_notes[selected]
            st.markdown(f"""
            <div class='doc-card' style='line-height: 1.9;'>
                {notes.replace(chr(10), '<br>').replace('##', '<h4 style=font-family:Syne>').replace('**', '<strong>')}
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                txt_data = export_to_txt(f"Study Notes: {selected}\n\n{notes}")
                st.download_button("📥 Download Notes (TXT)", txt_data, f"notes_{selected}.txt", "text/plain")
            with c2:
                try:
                    pdf_data = export_to_pdf(f"Study Notes: {selected}", notes)
                    st.download_button("📄 Download Notes (PDF)", pdf_data, f"notes_{selected}.pdf", "application/pdf")
                except Exception:
                    st.caption("PDF export unavailable")

    with tabs[1]:
        st.markdown("Generate flashcards for active recall study sessions.")
        num_cards = st.slider("Number of Flashcards", 5, 20, 10)

        if st.button("🃏 Generate Flashcards", type="primary", key="gen_flash"):
            with st.spinner("Creating flashcards..."):
                cards = generate_flashcards(doc_text, api_key,
                                            st.session_state.settings["model"], num_cards)
                st.session_state.flashcards = cards
                st.session_state.token_usage += len(doc_text) // 4

        if st.session_state.flashcards:
            st.markdown(f"**{len(st.session_state.flashcards)} Flashcards Generated**")

            for i, card in enumerate(st.session_state.flashcards):
                with st.expander(f"Card {i+1}: {card.get('question', '')[:60]}..."):
                    col_q, col_a = st.columns(2)
                    with col_q:
                        st.markdown(f"""
                        <div style='background: rgba(108,99,255,0.1); border-radius: 10px;
                                   padding: 16px; min-height: 80px;'>
                            <div style='font-size: 11px; color: var(--primary-light);
                                       margin-bottom: 8px; text-transform: uppercase;
                                       letter-spacing: 1px;'>Question</div>
                            <div style='font-size: 15px; font-weight: 500;'>
                                {card.get('question', '')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_a:
                        st.markdown(f"""
                        <div style='background: rgba(67,230,184,0.08); border-radius: 10px;
                                   padding: 16px; min-height: 80px;'>
                            <div style='font-size: 11px; color: var(--accent2);
                                       margin-bottom: 8px; text-transform: uppercase;
                                       letter-spacing: 1px;'>Answer</div>
                            <div style='color: var(--text-secondary); line-height: 1.6;'>
                                {card.get('answer', '')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("Test your understanding with an AI-generated quiz.")
        num_q = st.slider("Number of Questions", 3, 15, 5)
        difficulty = st.select_slider("Difficulty", ["Easy", "Medium", "Hard"], value="Medium")

        if st.button("❓ Generate Quiz", type="primary", key="gen_quiz"):
            with st.spinner("Creating quiz..."):
                questions = generate_quiz(doc_text, api_key,
                                          st.session_state.settings["model"], num_q, difficulty)
                st.session_state.quiz_questions = questions
                st.session_state.token_usage += len(doc_text) // 4

        if st.session_state.quiz_questions:
            st.markdown(f"**{len(st.session_state.quiz_questions)} Questions**")
            score = 0

            for i, q in enumerate(st.session_state.quiz_questions):
                st.markdown(f"""
                <div class='doc-card'>
                    <div style='font-weight: 600; margin-bottom: 12px;'>
                        Q{i+1}. {q.get('question', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                options = q.get("options", [])
                if options:
                    answer = st.radio(
                        f"Select your answer for Q{i+1}",
                        options,
                        key=f"quiz_q_{i}",
                        label_visibility="collapsed"
                    )
                    correct = q.get("correct", "")
                    if answer:
                        if answer == correct:
                            st.success(f"✓ Correct! {q.get('explanation', '')}")
                            score += 1
                        else:
                            st.error(f"✗ Incorrect. Answer: {correct}. {q.get('explanation', '')}")


# ─── Page: Settings ───────────────────────────────────────────────────────────
def page_settings():
    st.markdown("""
    <div class='section-header'>
        <span class='section-icon'>⚙️</span>
        <div class='section-title'>Settings</div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🤖 AI Model", "🔒 Security", "📄 Processing", "📤 Export"])

    with tabs[0]:
        st.markdown("### AI Model Configuration")
        st.session_state.settings["model"] = st.selectbox(
            "Gemini Model",
            ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
            index=0,
            help="gemini-1.5-flash: Fast & cost-effective. gemini-1.5-pro: More capable."
        )
        st.session_state.settings["max_tokens"] = st.slider(
            "Max Response Tokens", 256, 4096, st.session_state.settings["max_tokens"])
        st.session_state.settings["temperature"] = st.slider(
            "Temperature", 0.0, 1.0, st.session_state.settings["temperature"], step=0.1,
            help="Higher = more creative. Lower = more focused.")

        st.markdown(f"""
        <div class='alert-info' style='margin-top: 16px;'>
            ℹ️ Current model: <strong>{st.session_state.settings['model']}</strong>
        </div>
        """, unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("### Security Settings")
        st.markdown("""
        <div class='doc-card'>
            <div style='font-weight: 600; margin-bottom: 12px;'>Active Security Measures</div>
            <div style='color: var(--text-secondary); font-size: 14px; line-height: 2;'>
                ✅ File type validation (PDF, DOCX, TXT, JPG, PNG only)<br>
                ✅ Upload size limit (50 MB per file)<br>
                ✅ Text sanitization (removes HTML/script injection)<br>
                ✅ Prompt injection detection<br>
                ✅ API key hidden from frontend<br>
                ✅ No code execution from uploaded files<br>
                ✅ Session isolation<br>
                ✅ Temp file cleanup
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑 Clear All Documents & Session", type="secondary"):
            st.session_state.uploaded_docs = {}
            st.session_state.chat_history = []
            st.session_state.embedder = None
            st.session_state.flashcards = []
            st.session_state.quiz_questions = []
            st.session_state.study_notes = {}
            st.session_state.current_summary = {}
            st.success("✅ Session cleared.")

    with tabs[2]:
        st.markdown("### Document Processing")
        st.session_state.settings["chunk_size"] = st.slider(
            "Chunk Size (characters)", 500, 5000, st.session_state.settings["chunk_size"],
            help="For long documents, controls how text is split for processing.")
        st.session_state.settings["ocr_enabled"] = st.toggle(
            "Enable OCR for Images/Scans",
            value=st.session_state.settings["ocr_enabled"])

        lang_options = {"auto": "Auto-detect", "en": "English", "es": "Spanish",
                        "fr": "French", "de": "German", "zh": "Chinese", "ar": "Arabic"}
        lang_key = st.selectbox("Processing Language",
                                list(lang_options.keys()),
                                format_func=lambda x: lang_options[x])
        st.session_state.settings["language"] = lang_key

    with tabs[3]:
        st.markdown("### Export Options")
        st.markdown("""
        <div class='doc-card'>
            <div style='font-weight: 600; margin-bottom: 12px;'>Available Export Formats</div>
            <div style='color: var(--text-secondary); font-size: 14px; line-height: 2;'>
                📄 <strong>PDF</strong> — Full analysis report with formatting<br>
                📃 <strong>TXT</strong> — Plain text summary or notes<br>
                💬 <strong>Chat History</strong> — Export conversation as JSON<br>
                🃏 <strong>Flashcards</strong> — Download as text or PDF
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.chat_history:
            chat_json = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button("💬 Export Chat History (JSON)",
                               chat_json.encode(), "chat_history.json", "application/json")

        if st.session_state.flashcards:
            cards_txt = "\n\n".join(
                [f"Q: {c.get('question','')}\nA: {c.get('answer','')}"
                 for c in st.session_state.flashcards]
            )
            st.download_button("🃏 Export Flashcards (TXT)",
                               cards_txt.encode(), "flashcards.txt", "text/plain")


# ─── Main Router ──────────────────────────────────────────────────────────────
render_sidebar()

page_map = {
    "Home": page_home,
    "Upload Documents": page_upload,
    "AI Chat": page_chat,
    "Compare Documents": page_compare,
    "Analytics": page_analytics,
    "Study Tools": page_study,
    "Settings": page_settings,
}

current_page = st.session_state.get("page", "Home")
render_fn = page_map.get(current_page, page_home)
render_fn()
