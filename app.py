import os
import sys
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN", "")

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from rag import load_and_index_pdf, load_existing_vectorstore, search_documents
from agent import run_agent

UPLOAD_DIR = "/tmp/uploads"
VECTORSTORE_DIR = "/tmp/vectorstore"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

st.set_page_config(
    page_title="Agentic Document Intelligence",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700&family=Space+Grotesk:wght=600;700&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

.stApp {
    background: radial-gradient(ellipse at 20% 0%, #1a1040 0%, #0d0d1a 50%, #0a0a14 100%);
    min-height: 100vh;
}

header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 2rem !important; }

/* ── GLOW ORBS ── */
.glow-orb-1 {
    position: fixed; top: -100px; left: -100px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
}
.glow-orb-2 {
    position: fixed; bottom: -100px; right: -100px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(168,85,247,0.08) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
}

/* ── TOP BADGE ── */
.top-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 30px; padding: 5px 16px;
    font-size: 11px; color: #a5b4fc; margin-bottom: 14px;
    letter-spacing: 0.04em;
}
.live-dot {
    width: 7px; height: 7px; background: #22c55e;
    border-radius: 50%; animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.3); }
}

/* ── HERO TITLE ── */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem; font-weight: 700;
    color: #f1f5f9; margin: 0 0 10px 0; line-height: 1.15;
}
.hero-title .accent { 
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    color: #94a3b8; font-size: 0.92rem;
    max-width: 520px; line-height: 1.6; margin: 0 0 18px 0;
}

/* ── TECH PILLS (HIGH CONTRAST) ── */
.tech-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 28px; }
.tech-pill {
    background: rgba(99, 102, 241, 0.2) !important;
    border: 1px solid rgba(129, 140, 248, 0.5) !important;
    border-radius: 6px; padding: 4px 12px;
    font-size: 12px; font-weight: 600; color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

/* ── STATS BAR ── */
.stats-bar {
    display: flex; gap: 1px; margin-bottom: 28px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; overflow: hidden;
}
.stat-item {
    flex: 1; padding: 18px 20px;
    background: rgba(13,13,26,0.8);
    text-align: center; position: relative;
}
.stat-item:not(:last-child) { border-right: 1px solid rgba(255,255,255,0.05); }
.stat-item::before {
    content: ''; position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 40%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(129,140,248,0.5), transparent);
}
.stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem; font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    display: block;
}
.stat-label {
    font-size: 0.68rem; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px;
}

/* ── CHAT AREA ── */
.chat-wrapper {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 20px;
    margin-bottom: 16px; min-height: 200px;
}

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    margin-bottom: 12px !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] span { color: #e2e8f0 !important; }

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(99,102,241,0.08) !important;
    border-color: rgba(99,102,241,0.15) !important;
    margin-left: 48px !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1e 0%, #0d0d1a 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label { color: #e2e8f0 !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.05) !important;
    border: 2px dashed rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
    padding: 8px !important;
}
[data-testid="stFileDropzone"] { background: transparent !important; border: none !important; }
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] small { color: #818cf8 !important; }
[data-testid="stFileUploader"] button {
    background: rgba(99,102,241,0.15) !important;
    color: #a5b4fc !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 8px !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 10px 20px !important; width: 100% !important;
    font-size: 0.88rem !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}

/* ── STATUS BADGES ── */
.status-ready {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 30px; padding: 6px 14px;
    font-size: 0.8rem; color: #86efac; font-weight: 500;
}
.status-empty {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 30px; padding: 6px 14px;
    font-size: 0.8rem; color: #fde68a; font-weight: 500;
}
.green-dot { width: 7px; height: 7px; background: #22c55e; border-radius: 50%; display: inline-block; animation: pulse 2s infinite; }

/* ── STEPS ── */
.section-label {
    font-size: 0.68rem; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;
}
.step { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; }
.step-num {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2));
    border: 1px solid rgba(99,102,241,0.2);
    color: #818cf8; width: 22px; height: 22px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700; flex-shrink: 0;
}
.step-text { color: #94a3b8 !important; font-size: 0.82rem; line-height: 1.5; padding-top: 2px; }

/* ── FOOTER HIGH CONTRAST ── */
.footer {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 4px;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin-top: 20px;
}
.footer-left { color: #94a3b8; font-size: 0.75rem; }
.footer-tags { display: flex; gap: 6px; }
.footer-tag {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 6px; padding: 2px 10px;
    font-size: 11px; color: #94a3b8;
}

.ai-tool-badge {
    display: flex; align-items: center; gap: 10px;
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 10px; padding: 10px 14px; margin-bottom: 16px;
}
.ai-tool-icon { font-size: 20px; }
.ai-tool-text { font-size: 0.78rem; color: #94a3b8; line-height: 1.4; }
.ai-tool-text strong { color: #a5b4fc; display: block; font-size: 0.82rem; }

/* ──────────────────────────────────────────────────────────────
   ── CRITICAL FIX: CHAT INPUT ABSOLUTE FORCE INVISIBILITY FIX ── 
   ────────────────────────────────────────────────────────────── */
footer, footer * { display: none !important; }

div[data-testid="stBottomBlockContainer"],
section[data-testid="stBottom"] {
    background-color: #0a0a14 !important;
}

/* Targets the complete custom layout frame */
[data-testid="stChatInput"] {
    background-color: #ffffff !important;
    border: 1px solid rgba(129,140,248,0.7) !important;
    border-radius: 14px !important;
}

/* DEEP TARGETING FOR THE TEXT AREA ELEMENT AND ALL STREAMLIT GRAPHIC LAYERS */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] p,
[data-impl="st_chat_input"] textarea,
.stChatInputContainer textarea {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 600 !important;
    background-color: transparent !important;
}

/* Custom placeholder visibility control */
[data-testid="stChatInput"] textarea::placeholder,
.stChatInputContainer textarea::placeholder {
    color: #555566 !important;
    -webkit-text-fill-color: #555566 !important;
    opacity: 0.85 !important;
}
</style>
""", unsafe_allow_html=True)

# Glow orbs
st.markdown('<div class="glow-orb-1"></div><div class="glow-orb-2"></div>', unsafe_allow_html=True)

# ── HEADER ──
st.markdown("""
<div class="top-badge"><span class="live-dot"></span> AI powered · LangGraph agent · RAG pipeline</div>
<div class="hero-title">Agentic Document <span class="accent">Intelligence</span></div>
<div class="hero-subtitle">Upload any PDF and ask questions in natural language. Your AI agent searches your documents, the web, and combines insights intelligently.</div>
<div class="tech-pills">
  <span class="tech-pill">🔗 LangChain</span>
  <span class="tech-pill">🕸 LangGraph</span>
  <span class="tech-pill">⚡ FAISS</span>
  <span class="tech-pill">🦙 Groq LLaMA 3.3</span>
  <span class="tech-pill">🌐 Tavily Search</span>
  <span class="tech-pill">🤗 HuggingFace</span>
</div>
""", unsafe_allow_html=True)

# ── STATS BAR ──
st.markdown("""
<div class="stats-bar">
  <div class="stat-item"><span class="stat-num">3</span><span class="stat-label">AI Tools</span></div>
  <div class="stat-item"><span class="stat-num">2</span><span class="stat-label">Sources Searched</span></div>
  <div class="stat-item"><span class="stat-num">70B</span><span class="stat-label">Model Params</span></div>
  <div class="stat-item"><span class="stat-num">RAG</span><span class="stat-label">Pipeline</span></div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("""
    <div class="ai-tool-badge">
      <div class="ai-tool-icon">🤖</div>
      <div class="ai-tool-text">
        <strong>Agentic RAG Assistant</strong>
        Powered by LLaMA 3.3 70B via Groq
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">📁 Upload Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop your PDF here or click to browse", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("⬆ Upload & Index"):
            with st.spinner("Indexing your document..."):
                try:
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    load_and_index_pdf(file_path)
                    st.success(f"✅ {uploaded_file.name} ready!")
                    st.session_state["doc_uploaded"] = True
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")

    st.markdown("<br>", unsafe_allow_html=True)
    has_documents = os.path.exists(VECTORSTORE_DIR) and len(os.listdir(VECTORSTORE_DIR)) > 0
    if has_documents:
        st.markdown('<div class="status-ready"><span class="green-dot"></span> Documents ready to query</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-empty">⚠ No documents uploaded yet</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">📖 How to Use</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step"><div class="step-num">1</div><div class="step-text">Upload your PDF document</div></div>
    <div class="step"><div class="step-num">2</div><div class="step-text">Click Upload & Index to process it</div></div>
    <div class="step"><div class="step-num">3</div><div class="step-text">Ask any question in the chat</div></div>
    <div class="step"><div class="step-num">4</div><div class="step-text">AI searches docs and web together</div></div>
    """, unsafe_allow_html=True)

# ── CHAT ──
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I am your AI document assistant. Upload a PDF and I will answer questions from it — or search the web if needed.",
        "source": None
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("✦ Ask anything about your document or any topic..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "source": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Agent is thinking..."):
            try:
                answer = run_agent(prompt)
                source = "From your document"
            except Exception as e:
                answer = f"❌ Error: {str(e)}"
                source = None
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer, "source": source})

# ── FOOTER ──
st.markdown("""
<div class="footer">
  <div class="footer-left">Agentic RAG Document Intelligence · Built with LangChain, LangGraph & Streamlit</div>
  <div class="footer-tags">
    <span class="footer-tag">Python</span>
    <span class="footer-tag">LangGraph</span>
    <span class="footer-tag">Streamlit</span>
  </div>
</div>
""", unsafe_allow_html=True)
#