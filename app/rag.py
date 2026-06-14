import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

UPLOAD_FOLDER = "/tmp/uploads"
VECTORSTORE_FOLDER = "/tmp/vectorstore"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

# ✅ HF_TOKEN already in os.environ by the time this loads
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def load_and_index_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_FOLDER)
    return vectorstore


def load_existing_vectorstore():
    return FAISS.load_local(
        VECTORSTORE_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )


def search_documents(query: str, vectorstore) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    results = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in results])