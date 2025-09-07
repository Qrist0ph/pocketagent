# export OPENAI_API_KEY=sk-123456789abcdef
# streamlit run src/streamlit/chatbot.py 

import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pocketagent_app import PocketAgentApp
from pocketagent import RAGBot
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# === CONFIGURATION STRINGS ===
import os
WORKING_DIR = os.getcwd()
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
MD_PATH_STR = os.path.join(WORKING_DIR, "product_faq.md")
FAISS_PATH = os.path.join(WORKING_DIR, "faiss_index")
MCP_SERVER_CMD = "python3"
MCP_SERVER_ARGS = [os.path.join(WORKING_DIR, "src", "mcpserver", "server.py")]

llm = ChatOpenAI(model=MODEL_NAME)
emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
rag_bot_instance = RAGBot(
    llm=llm,
    emb=emb,
    md_path_str=MD_PATH_STR,
    faiss_path=FAISS_PATH,
)

# MCP agent creation (async)
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import load_mcp_tools
from langgraph.prebuilt import create_react_agent

def get_mcp_react_agent():
    server_params = StdioServerParameters(command=MCP_SERVER_CMD, args=MCP_SERVER_ARGS)
    async def mcp_main():
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                agent = create_react_agent(llm, tools)
                return agent
    return asyncio.run(mcp_main())

if "OPENAI_API_KEY" not in os.environ:
    api_key = st.text_input("Bitte geben Sie Ihren OpenAI API-Schlüssel ein:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API-Schlüssel wurde gesetzt.")

if "OPENAI_API_KEY" in os.environ:
    mcp_agent = get_mcp_react_agent()
    pocket_app = PocketAgentApp(
        llm=llm,
        emb=emb,
        rag_bot_instance=rag_bot_instance,
        mcp_agent_instance=mcp_agent,
    )

    st.title("PocketAgent Chatbot")
    st.write("Stellen Sie Ihre Fragen an den PocketAgent!")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    user_input = st.text_input("Ihre Nachricht:")
    if st.button("Senden") and user_input:
        thread_id = "demo"
        response = pocket_app.chat(user_input, thread_id=thread_id)
        st.session_state["messages"].append(("user", user_input))
        st.session_state["messages"].append(("agent", response))

    for role, msg in st.session_state["messages"]:
        if role == "user":
            st.markdown(f"**Sie:** {msg}")
        else:
            st.markdown(f"**Agent:** {msg}")
