import streamlit as st
import requests

# Backend URL
API_URL = "http://127.0.0.1:8000"

# ─── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ─── HEADER ──────────────────────────────────────────────
st.title("🤖 Agentic Document Intelligence")
st.markdown("Upload your documents and ask questions — AI will search your docs and the web!")
st.divider()

# ─── SIDEBAR: FILE UPLOAD ────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    st.markdown("Upload a PDF to get started")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if uploaded_file is not None:
        if st.button("Upload & Index", type="primary"):
            with st.spinner("Uploading and indexing..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    }
                    response = requests.post(
                        f"{API_URL}/upload",
                        files=files
                    )
                    if response.status_code == 200:
                        st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                        st.session_state["doc_uploaded"] = True
                    else:
                        st.error("Upload failed. Try again.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.divider()

    # Check document status
    try:
        status = requests.get(f"{API_URL}/status").json()
        if status.get("documents_uploaded"):
            st.success("📚 Documents ready!")
        else:
            st.warning("⚠️ No documents uploaded yet")
    except:
        st.error("❌ Backend not running")

    st.divider()
    st.markdown("**How to use:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. Click Upload & Index")
    st.markdown("3. Ask any question below")
    st.markdown("4. AI searches docs + web!")


# ─── CHAT HISTORY ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ─── CHAT INPUT ──────────────────────────────────────────
if prompt := st.chat_input("Ask anything about your document or any topic..."):

    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from agent
    with st.chat_message("assistant"):
        with st.spinner("🤔 Agent is thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": prompt}
                )

                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer received")
                else:
                    answer = f"Error: {response.status_code}"

            except Exception as e:
                answer = f"Could not connect to backend: {str(e)}"

        st.markdown(answer)

        # Add assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


# ─── FOOTER ──────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Agentic RAG Document Intelligence | Built with LangChain, LangGraph & Streamlit</p>",
    unsafe_allow_html=True
)