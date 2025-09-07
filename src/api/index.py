# export OPENAI_API_KEY=sk-123456789abcdef
# python3 src/api/index.py 
# uvicorn src.api.index:app --reload
# http://localhost:8000/docs
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import sys
import os

# ----- local imports / path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pocketagent_app import PocketAgentApp
from pocketagent import RAGBot

# ----- LLM / Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ----- MCP + LangGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import load_mcp_tools
from langgraph.prebuilt import create_react_agent

# =====================================================================
# Config
# =====================================================================
MD_PATH_STR = "/mnt/c/repos/vercel.pocketagent/product_faq.md"
FAISS_PATH = "/mnt/c/repos/vercel.pocketagent/faiss_index"
MCP_SERVER_CMD = "python3"
MCP_SERVER_ARGS = ["/mnt/c/repos/vercel.pocketagent/src/mcpserver/server.py"]

# =====================================================================
# Core singletons (safe to build synchronously)
# =====================================================================
llm = ChatOpenAI(model="gpt-4o-mini")
emb = OpenAIEmbeddings(model="text-embedding-3-small")
rag_bot_instance = RAGBot(
    llm=llm,
    emb=emb,
    md_path_str=MD_PATH_STR,
    faiss_path=FAISS_PATH,
)


# =====================================================================
# MCP Agent init (async)
# =====================================================================
async def get_mcp_react_agent():
    server_params = StdioServerParameters(command=MCP_SERVER_CMD, args=MCP_SERVER_ARGS)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(llm, tools)
            return agent

# =====================================================================
# Lifespan context manager
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    mcp_agent = await get_mcp_react_agent()
    pocket_app = PocketAgentApp(
    llm=llm,
    emb=emb,
    rag_bot_instance=rag_bot_instance,
    mcp_agent_instance=mcp_agent,
	)

    app.state.pocket_app = pocket_app
    yield
    # shutdown
    # you could gracefully stop MCP server here if needed

# =====================================================================
# FastAPI app
# =====================================================================
app = FastAPI(title="PocketAgent API", lifespan=lifespan)

class ChatRequest(BaseModel):
    text: str
    thread_id: Optional[str] = "demo"

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    response = app.state.pocket_app.chat(req.text, thread_id=req.thread_id)
    return {"response": response}

@app.get("/ping")
def ping():
    return {"ok": True}

# =====================================================================
# Uvicorn entry (optional, for local runs)
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    import os
    module_path = os.path.splitext(os.path.relpath(__file__, os.getcwd()))[0].replace(os.sep, ".")
    uvicorn.run(f"{(__file__.replace('.py', ''))}:app", host="0.0.0.0", port=8000, reload=True)
