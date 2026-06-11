import os
import shutil
import threading
import requests
import streamlit as st
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Fix import paths
from app.rag import load_and_index_pdf, load_existing_vectorstore
from app.agent import run_agent

# Load API keys
load_dotenv()

# ─── FASTAPI BACKEND ─────────────────────────────────────
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class QueryRequest(BaseModel):
    question: str


@fastapi_app.get("/")
def home():
    return {"status": "running"}


@fastapi_app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        load_and_index_pdf(file_path)
        return {"status": "success", "message": f"{file.filename} uploaded!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fastapi_app.post("/ask")
async def ask_question(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        answer = run_agent(request.question)
        return {"status": "success", "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fastapi_app.get("/status")
def check_status():
    vectorstore_path = "vectorstore"
    has_documents = os.path.exists(vectorstore_path) and \
                    len(os.listdir(vectorstore_path)) > 0
    return {"status": "ready", "documents_uploaded": has_documents}


# ─── START FASTAPI IN BACKGROUND THREAD ──────────────────
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run_fastapi, daemon=True)
thread.start()

# ─── STREAMLIT FRONTEND ───────────────────────────────────
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic Document Intelligence")
st.markdown("Upload your documents and ask questions — AI will search your docs and the web!")
st.divider()

with st.sidebar:
    st.header("📄 Upload Document")
    st.markdown("Upload a PDF to get started")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

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
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success(f"✅ {uploaded_file.name} uploaded!")
                        st.session_state["doc_uploaded"] = True
                    else:
                        st.error("Upload failed. Try again.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.divider()

    try:
        status = requests.get(f"{API_URL}/status").json()
        if status.get("documents_uploaded"):
            st.success("📚 Documents ready!")
        else:
            st.warning("⚠️ No documents uploaded yet")
    except:
        st.warning("⏳ Backend starting...")

    st.divider()
    st.markdown("**How to use:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. Click Upload & Index")
    st.markdown("3. Ask any question below")
    st.markdown("4. AI searches docs + web!")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about your document or any topic..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Agent is thinking..."):
            try:
                response = requests.post(f"{API_URL}/ask", json={"question": prompt})
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer received")
                else:
                    answer = f"Error: {response.status_code}"
            except Exception as e:
                answer = f"Could not connect to backend: {str(e)}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Agentic RAG Document Intelligence | Built with LangChain, LangGraph & Streamlit</p>",
    unsafe_allow_html=True
)