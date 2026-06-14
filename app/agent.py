import os
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient
from app.rag import search_documents, load_existing_vectorstore

# ✅ Pure os.getenv — HF Spaces sets these from the Secrets panel
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

vectorstore = None


def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        try:
            vectorstore = load_existing_vectorstore()
        except Exception:
            vectorstore = None
    return vectorstore


@tool
def search_document_tool(query: str) -> str:
    """Search the uploaded documents for relevant information.
    Use this tool first when user asks about document content."""
    vs = get_vectorstore()
    if vs is None:
        return "No documents uploaded yet. Please upload a PDF first."
    results = search_documents(query, vs)
    return f"From documents:\n{results}" if results else "No relevant information found in documents."


@tool
def web_search_tool(query: str) -> str:
    """Search the web for current information.
    Use this when document doesn't have the answer or user asks about recent/general topics."""
    try:
        response = tavily_client.search(query=query, max_results=3)
        results = response.get("results", [])
        if not results:
            return "No web results found."
        combined = ""
        for r in results:
            combined += f"Source: {r['url']}\nContent: {r['content']}\n\n"
        return f"From web search:\n{combined}"
    except Exception as e:
        return f"Web search failed: {str(e)}"


@tool
def search_both_tool(query: str) -> str:
    """Search both documents and web together.
    Use this when user wants comprehensive information from both uploaded docs and internet."""
    return f"{search_document_tool.invoke(query)}\n\n{web_search_tool.invoke(query)}"


def create_agent():
    return create_react_agent(
        model=llm,
        tools=[search_document_tool, web_search_tool, search_both_tool],
        prompt="""You are an intelligent document assistant.

You have access to three tools:
1. search_document_tool - Search uploaded PDF documents
2. web_search_tool - Search the internet
3. search_both_tool - Search both documents and web

Rules:
- Always try search_document_tool first
- If document doesn't have the answer, use web_search_tool
- If user wants comprehensive info, use search_both_tool
- Always mention where the answer came from
- Be clear, concise and helpful
- If no information found anywhere, say so honestly
"""
    )


def run_agent(query: str) -> str:
    agent = create_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content