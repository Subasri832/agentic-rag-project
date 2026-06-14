import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Fix import paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from rag import load_and_index_pdf, load_existing_vectorstore, search_documents
from agent import run_agent

load_dotenv()

# Create folders (use /tmp on cloud platforms — local folders don't persist)
UPLOAD_DIR = "/tmp/uploads"
VECTORSTORE_DIR = "/tmp/vectorstore"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# PAGE CONFIG
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic Document Intelligence")
st.markdown("Upload your documents and ask questions — AI will search your docs and the web!")
st.divider()

# SIDEBAR
with st.sidebar:
    st.header("📄 Upload Document")
    st.markdown("Upload a PDF to get started")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Upload & Index", type="primary"):
            with st.spinner("Uploading and indexing..."):
                try:
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    load_and_index_pdf(file_path)
                    st.success(f"✅ {uploaded_file.name} uploaded!")
                    st.session_state["doc_uploaded"] = True
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")

    st.divider()

    has_documents = os.path.exists(VECTORSTORE_DIR) and \
                    len(os.listdir(VECTORSTORE_DIR)) > 0
    if has_documents:
        st.success("📚 Documents ready!")
    else:
        st.warning("⚠️ No documents uploaded yet")

    st.divider()
    st.markdown("**How to use:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. Click Upload & Index")
    st.markdown("3. Ask any question below")
    st.markdown("4. AI searches docs + web!")


# CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about your document or any topic..."):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Agent is thinking..."):
            try:
                answer = run_agent(prompt)
            except Exception as e:
                answer = f"❌ Error: {str(e)}"

        st.markdown(answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Agentic RAG Document Intelligence | Built with LangChain, LangGraph & Streamlit</p>",
    unsafe_allow_html=True
)