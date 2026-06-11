import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient
from app.rag import search_documents, load_existing_vectorstore

# Load API keys
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Initialize Tavily web search client
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# Load vectorstore once when agent starts
vectorstore = None


def get_vectorstore():
    """Load vectorstore if it exists"""
    global vectorstore
    if vectorstore is None:
        try:
            vectorstore = load_existing_vectorstore()
        except Exception:
            vectorstore = None
    return vectorstore


# ─── TOOL 1: Search Documents ───────────────────────────
@tool
def search_document_tool(query: str) -> str:
    """
    Search the uploaded documents for relevant information.
    Use this tool first when user asks about document content.
    """
    vs = get_vectorstore()
    if vs is None:
        return "No documents uploaded yet. Please upload a PDF first."
    
    results = search_documents(query, vs)
    if not results:
        return "No relevant information found in documents."
    
    return f"From documents:\n{results}"


# ─── TOOL 2: Web Search ─────────────────────────────────
@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for current information.
    Use this when document doesn't have the answer
    or user asks about recent/general topics.
    """
    try:
        response = tavily_client.search(
            query=query,
            max_results=3
        )
        results = response.get("results", [])
        if not results:
            return "No web results found."
        
        # Combine top results
        combined = ""
        for r in results:
            combined += f"Source: {r['url']}\n"
            combined += f"Content: {r['content']}\n\n"
        
        return f"From web search:\n{combined}"
    
    except Exception as e:
        return f"Web search failed: {str(e)}"


# ─── TOOL 3: Combine Both Sources ───────────────────────
@tool
def search_both_tool(query: str) -> str:
    """
    Search both documents and web together.
    Use this when user wants comprehensive information
    from both uploaded docs and internet.
    """
    doc_results = search_document_tool.invoke(query)
    web_results = web_search_tool.invoke(query)
    
    return f"{doc_results}\n\n{web_results}"


# ─── CREATE AGENT ────────────────────────────────────────
def create_agent():
    """
    Creates the LangGraph ReAct agent with all tools
    """
    tools = [
        search_document_tool,
        web_search_tool,
        search_both_tool
    ]
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
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
    return agent


def run_agent(query: str) -> str:
    """
    Runs the agent with a user query and returns the answer
    """
    agent = create_agent()
    
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    # Extract final answer from messages
    final_message = result["messages"][-1]
    return final_message.content