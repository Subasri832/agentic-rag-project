---
title: Agentic RAG Project
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.58.0"
app_file: app.py
pinned: false
---

<div align="center">

# 🤖 Agentic Document Intelligence System

**An AI Agent that reads your PDFs, searches the web, and synthesizes answers — deciding on its own which tools to use.**

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/Subasri832/agentic-rag-project)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Subasri832/agentic-rag-project)
[![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)

</div>

---

## 🚀 What is this?

This is not just a chatbot. It's a **true AI Agent** built with LangGraph that autonomously decides how to answer your questions — whether by searching your uploaded PDF, fetching live web results, or combining both.

Upload any PDF → Ask anything in natural language → Get accurate, cited answers.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **PDF Intelligence** | Upload any PDF up to 200MB — chunked, embedded, and indexed automatically |
| 🧠 **Agentic Reasoning** | LangGraph agent decides which tools to use per query — no hardcoded logic |
| 🌐 **Live Web Search** | Falls back to Tavily web search when the document doesn't have the answer |
| ⚡ **Blazing Fast** | Powered by Groq's custom silicon running LLaMA 3.3 70B |
| 🔍 **Semantic Search** | FAISS + HuggingFace embeddings find meaning, not just keywords |
| 📝 **Source Citations** | Every answer traceable to document pages or web URLs |

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│  LangGraph      │  ← AI Agent decides tool strategy
│  Agent (LLM)   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌────────┐
│ FAISS │ │ Tavily │
│  RAG  │ │  Web   │
│Search │ │ Search │
└───┬───┘ └───┬────┘
    │          │
    └────┬─────┘
         │
         ▼
  Cited Answer → Streamlit UI
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Agent Orchestration** | LangChain + LangGraph | Agentic workflows & multi-tool reasoning |
| **LLM** | Groq LLaMA 3.3 70B | Fast, accurate language understanding |
| **Vector DB** | FAISS | Semantic similarity search |
| **Embeddings** | HuggingFace | Text-to-vector conversion |
| **Web Search** | Tavily API | Real-time web retrieval |
| **Backend** | FastAPI | REST API & business logic |
| **Frontend** | Streamlit | Interactive chat UI |
| **Language** | Python 3.10+ | Core implementation |

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Subasri832/agentic-rag-project.git
cd agentic-rag-project
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get your keys:
- Groq API Key → [console.groq.com](https://console.groq.com)
- Tavily API Key → [tavily.com](https://tavily.com)

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 How to Use

1. **Upload** your PDF document using the sidebar uploader
2. **Wait** for indexing to complete (the agent will confirm)
3. **Ask** any question in natural language in the chat box
4. **Get** an accurate, cited answer — from the doc, the web, or both

---

## 🧠 How the Agent Works

The LangGraph agent follows a **decide → retrieve → synthesize** loop:

1. **Receives** your question
2. **Reasons** about whether the answer is likely in the document or requires web search
3. **Calls** the appropriate tool(s) — FAISS retriever, Tavily search, or both
4. **Synthesizes** a final answer with source citations
5. **Streams** the response back to the UI

This is fundamentally different from static RAG pipelines — the agent can chain multiple tool calls and adapt its strategy mid-response.

---

## 📚 Key Concepts Demonstrated

- **Retrieval-Augmented Generation (RAG)** — grounding LLM answers in real documents
- **Agentic AI & Tool Use** — LLMs that plan and execute multi-step reasoning
- **Vector Databases** — semantic similarity search with FAISS
- **Prompt Engineering** — system prompts that guide reliable agent behavior
- **Full-Stack AI Development** — FastAPI + Streamlit production architecture
- **API Integration** — Groq, HuggingFace, and Tavily in a single pipeline

---

## 🗂️ Project Structure

```
agentic-rag-project/
├── app.py                  # Streamlit frontend & chat UI
├── agent/
│   ├── graph.py            # LangGraph agent definition
│   ├── tools.py            # RAG & web search tools
│   └── prompts.py          # System & tool prompts
├── rag/
│   ├── loader.py           # PDF loading & chunking
│   ├── embeddings.py       # HuggingFace embeddings
│   └── retriever.py        # FAISS vector store
├── api/
│   └── main.py             # FastAPI backend
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🌱 About the Author

Built by **Subasri** — ECE student passionate about AI engineering and building production-ready intelligent systems.

This project was built end-to-end: from architecture design, RAG pipeline implementation, agentic workflow design, to full deployment on Hugging Face Spaces.

[![LinkedIn](https://img.shields.io/badge/Connect-LinkedIn-blue?style=flat&logo=linkedin)](https://linkedin.com)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow?style=flat)](https://huggingface.co/Subasri832)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with ❤️ using LangChain · LangGraph · Groq · FAISS · Streamlit</sub>
</div>