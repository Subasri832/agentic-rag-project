import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ✅ Use /tmp for cloud platforms (Streamlit Cloud is read-only except /tmp)
UPLOAD_FOLDER = "/tmp/uploads"
VECTORSTORE_FOLDER = "/tmp/vectorstore"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

# Load embeddings model once globally
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


def load_and_index_pdf(pdf_path: str):
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Total pages loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_FOLDER)   # ✅ saves to /tmp/vectorstore
    print("Vectorstore saved successfully!")
    return vectorstore


def load_existing_vectorstore():
    vectorstore = FAISS.load_local(
        VECTORSTORE_FOLDER,                      # ✅ loads from /tmp/vectorstore
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Existing vectorstore loaded!")
    return vectorstore


def search_documents(query: str, vectorstore) -> str:
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )
    results = retriever.invoke(query)
    combined = "\n\n".join([doc.page_content for doc in results])
    return combined