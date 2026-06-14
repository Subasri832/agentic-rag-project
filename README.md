# 🤖 Agentic Document Intelligence System

An AI-powered document Q&A system that uses RAG and 
an AI Agent to answer questions from uploaded PDFs 
and the web.

## Features
- Upload any PDF and ask questions in natural language
- AI Agent decides to search docs, web, or both
- Built with LangChain, LangGraph, FAISS, Groq LLaMA 3
- Streamlit chat interface

## Tech Stack
- Python, LangChain, LangGraph
- FAISS Vector Database
- Groq LLaMA 3.3 (Free LLM)
- HuggingFace Embeddings
- FastAPI, Streamlit
- Tavily Web Search

## How to Run
1. Clone the repo
2. Install requirements: `pip install -r requirements.txt`
3. Add API keys in `.env` file
4. Run backend: `uvicorn app.main:app --reload`
5. Run frontend: `streamlit run app.py`