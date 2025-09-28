# export OPENAI_API_KEY=sk-123456789abcdef
# pip install -e .
# python3 src/api/index.py 
# uvicorn src.api.index:app --reload
# http://localhost:8000/docs
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import os

# ----- Clean imports (no sys.path needed)
from pocketagent.pocketagent_app import PocketAgentApp
from pocketagent.ragagent.ragagent import RagAgent
from pocketagent.ragagent.ragbot import RAGBot
from pocketagent.mcpagent.mcpagent import MCPAgent
# ----- LLM / Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver

# =====================================================================
# Config
# =====================================================================
WORKING_DIR = os.getcwd()
MCP_SERVER_PATH = os.path.join(WORKING_DIR, "src", "mcpserver", "server.py")


import os
WORKING_DIR = os.getcwd()
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
MD_PATH_STR = os.path.join(WORKING_DIR, "product_faq.md")
FAISS_PATH = os.path.join(WORKING_DIR, "faiss_index")
MCP_SERVER_COMMAND = "python3"
MCP_SERVER_PATH = os.path.join(WORKING_DIR, "src", "mcpserver", "server.py")
MCP_SERVER_ARGS = [MCP_SERVER_PATH]
# =====================================================================
# Core singletons (safe to build synchronously)
# =====================================================================
llm = ChatOpenAI(model="gpt-4o-mini")
emb = OpenAIEmbeddings(model="text-embedding-3-small")
memory_saver = MemorySaver()  # Global memory for persistence

# =====================================================================
# Lifespan context manager
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    #-------------------------------------
    # RagAgent setup
    rag_agent = RagAgent(RAGBot(
        llm=llm,
        emb=emb,
        md_path_str=MD_PATH_STR,
        faiss_path=FAISS_PATH
    ))
    
    #-------------------------------------
    # MCPAgent setup
    # Using langchain_mcp_adapters to create a MultiServerMCPClient
    from langchain_mcp_adapters.client import MultiServerMCPClient
    mcp_client = MultiServerMCPClient(
        {
            "math": {
                "command": MCP_SERVER_COMMAND,
                "args": MCP_SERVER_ARGS,
                "transport": "stdio",
            }
        }
    )
    tools = await mcp_client.get_tools()
    mcp_agent = MCPAgent(llm=llm,tools=tools)
    
    #-------------------------------------
    # Create PocketAgentApp with persistent memory
    pocket_app = PocketAgentApp(llm=llm, emb=emb, rag_agent=rag_agent, mcp_agent=mcp_agent, memory_saver=memory_saver)

    app.state.pocket_app = pocket_app
    app.state.mcp_client = mcp_client
    
    yield
    
    # Cleanup
    await mcp_client.close()
 
# =====================================================================
# FastAPI app
# =====================================================================
app = FastAPI(title="PocketAgent API", lifespan=lifespan)

class ChatRequest(BaseModel):
    text: str
    thread_id: Optional[str] = "demo"

class ChatResponse(BaseModel):
    response: str
    thread_id: str

# Encapsulated chat logic
async def run_pocketagent_chat(text: str, thread_id: str) -> str:
    from langchain_core.messages import HumanMessage
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        out = await app.state.pocket_app.get_graph().ainvoke({
            "messages": [HumanMessage(content=text)],
            "intent": "",
            "thread_id": thread_id
        }, config)
        return out["messages"][-1].content
    except Exception as e:
        print(f"Error in chat: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    response = await run_pocketagent_chat(req.text, req.thread_id)
    return ChatResponse(response=response, thread_id=req.thread_id)

@app.get("/ping")
def ping():
    return {"ok": True, "status": "PocketAgent API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "components": {
            "llm": "gpt-4o-mini",
            "embeddings": "text-embedding-3-small",
            "rag_enabled": True,
            "mcp_enabled": True
        }
    }

# =====================================================================
# Uvicorn entry (optional, for local runs)
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.index:app", host="0.0.0.0", port=8000, reload=True)
