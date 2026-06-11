import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from app.rag import load_and_index_pdf, load_existing_vectorstore
from app.agent import run_agent

# Load API keys
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG Document Intelligence",
    description="Upload documents and ask questions using AI",
    version="1.0.0"
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder paths
UPLOAD_FOLDER = "app/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─── REQUEST MODEL ───────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


# ─── ROUTE 1: Health Check ───────────────────────────────
@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Agentic RAG API is live!"
    }


# ─── ROUTE 2: Upload PDF ─────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and index it into FAISS vectorstore
    """
    # Check if file is PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    try:
        # Save uploaded file to uploads folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Index the PDF into FAISS
        load_and_index_pdf(file_path)

        return {
            "status": "success",
            "message": f"{file.filename} uploaded and indexed successfully!",
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


# ─── ROUTE 3: Ask Question ───────────────────────────────
@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Ask a question — agent searches docs and/or web
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        # Run the AI agent
        answer = run_agent(request.question)

        return {
            "status": "success",
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent failed: {str(e)}"
        )


# ─── ROUTE 4: Check if vectorstore exists ────────────────
@app.get("/status")
def check_status():
    """
    Check if any documents have been uploaded
    """
    vectorstore_path = "app/vectorstore"
    has_documents = os.path.exists(vectorstore_path) and \
                    len(os.listdir(vectorstore_path)) > 0

    return {
        "status": "ready",
        "documents_uploaded": has_documents
    }