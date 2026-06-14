import os
import sys

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Space+Grotesk:wght@700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] { background: transparent; }

.hero {
    background: linear-gradient(120deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
    border: 1px solid rgba(168,85,247,0.3);
    border-radius: 20px;
    padding: 48px 40px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(168,85,247,0.25), transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.25), transparent 70%);
    pointer-events: none;
}
.hero-icon {
    font-size: 64px;
    display: block;
    margin-bottom: 12px;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 12px 0;
}
.hero p {
    color: #c4b5fd;
    font-size: 1.05rem;
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.6;
}
.tech-pills {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 20px;
}
.tech-pill {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    color: #c4b5fd;
}

.features {
    display: flex;
    gap: 16px;
    margin-bottom: 32px;
    flex-wrap: wrap;
}
.feat-card {
    flex: 1;
    min-width: 140px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.feat-card:hover {
    transform: translateY(-4px);
    border-color: rgba(168,85,247,0.5);
}
.feat-card .icon { font-size: 28px; margin-bottom: 8px; }
.feat-card h3 {
    color: #e2e8f0;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0 0 6px 0;
}
.feat-card p {
    color: #94a3b8;
    font-size: 0.8rem;
    margin: 0;
    line-height: 1.4;
}

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(168,85,247,0.4) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b, #312e81) !important;
    border-right: 1px solid rgba(168,85,247,0.2) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 8px;
}
.status-ready {
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.4);
    color: #86efac;
}
.status-empty {
    background: rgba(251,191,36,0.15);
    border: 1px solid rgba(251,191,36,0.4);
    color: #fde68a;
}

.step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 12px;
}
.step-num {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}
.step-text { color: #c4b5fd; font-size: 0.85rem; line-height: 1.4; }

.chat-section-title {
    font-family: 'Space Grotesk', sans-serif;
    color: #e2e8f0;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.empty-chat {
    text-align: center;
    padding: 40px;
    color: #64748b;
    background: rgba(255,255,255,0.02);
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 16px;
    margin-bottom: 16px;
}
.empty-chat .empty-icon { font-size: 40px; margin-bottom: 12px; }
.empty-chat p { font-size: 0.9rem; line-height: 1.6; }
.empty-chat em { color: #a78bfa; }

.footer {
    text-align: center;
    padding: 20px;
    color: #64748b;
    font-size: 0.8rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 24px;
}
.footer span { color: #a78bfa; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <span class="hero-icon">🤖</span>
    <h1>Agentic Document Intelligence</h1>
    <p>Upload any PDF and ask questions — your AI agent searches your documents and the web to give you precise, sourced answers instantly.</p>
    <div class="tech-pills">
        <span class="tech-pill">LangChain</span>
        <span class="tech-pill">LangGraph</span>
        <span class="tech-pill">FAISS</span>
        <span class="tech-pill">Groq LLaMA 3.3</span>
        <span class="tech-pill">Tavily Search</span>
        <span class="tech-pill">HuggingFace Embeddings</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="features">
    <div class="feat-card">
        <div class="icon">📄</div>
        <h3>Document Search</h3>
        <p>Semantically searches inside your uploaded PDFs instantly</p>
    </div>
    <div class="feat-card">
        <div class="icon">🌐</div>
        <h3>Web Search</h3>
        <p>Falls back to live internet when your docs lack the answer</p>
    </div>
    <div class="feat-card">
        <div class="icon">🧠</div>
        <h3>Agentic AI</h3>
        <p>LLaMA 3.3 70B via Groq — fast, intelligent and context-aware</p>
    </div>
    <div class="feat-card">
        <div class="icon">⚡</div>
        <h3>Lightning Fast</h3>
        <p>FAISS vector store for sub-second semantic retrieval</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📄 Upload Document")
    st.markdown("<p style='color:#c4b5fd;font-size:0.85rem'>Upload a PDF to get started</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("⬆️ Upload & Index", type="primary"):
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

    st.markdown("---")

    has_documents = os.path.exists(VECTORSTORE_DIR) and len(os.listdir(VECTORSTORE_DIR)) > 0
    if has_documents:
        st.markdown('<div class="status-badge status-ready">📚 Documents ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-empty">⚠️ No documents yet</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    <div class="step"><div class="step-num">1</div><div class="step-text">Upload a PDF using the uploader above</div></div>
    <div class="step"><div class="step-num">2</div><div class="step-text">Click Upload & Index to process it</div></div>
    <div class="step"><div class="step-num">3</div><div class="step-text">Type your question in the chat below</div></div>
    <div class="step"><div class="step-num">4</div><div class="step-text">AI searches your docs and the web!</div></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#64748b;font-size:0.75rem;text-align:center'>Built with LangChain · LangGraph · Streamlit</p>", unsafe_allow_html=True)

st.markdown('<div class="chat-section-title">💬 Ask Anything</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-chat">
        <div class="empty-icon">💡</div>
        <p>Upload a PDF and start asking questions.<br>
        Try <em>"Summarize this document"</em> or <em>"What are the key points?"</em></p>
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about your document or any topic..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                answer = run_agent(prompt)
            except Exception as e:
                answer = f"❌ Error: {str(e)}"
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("""
<div class="footer">
    Agentic RAG · Built with <span>LangChain</span>, <span>LangGraph</span> &amp; <span>Streamlit</span> · Powered by <span>LLaMA 3.3 70B</span>
</div>
""", unsafe_allow_html=True)