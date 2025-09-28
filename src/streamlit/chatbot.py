# export OPENAI_API_KEY=sk-123456789abcdef
# streamlit run src/streamlit/chatbot.py 
# http://localhost:8501/

import streamlit as st
import sys
import os
import uuid
import asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pocketagent.pocketagent_app import PocketAgentApp
from pocketagent import RAGBot, RagAgent
from pocketagent.mcpagent.mcpagent import MCPAgent
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

# Encapsulated chat logic
async def run_pocketagent_chat(text, thread_id):
    global pocket_app
    from langchain_core.messages import HumanMessage
    config = {"configurable": {"thread_id": thread_id}}
    out = await pocket_app.get_graph().ainvoke({
        "messages": [HumanMessage(content=text)],
        "intent": "",
        "thread_id": thread_id
    }, config)
    return out["messages"][-1].content

# Initialize session state
def initialize_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory_saver" not in st.session_state:
        from langgraph.checkpoint.memory import MemorySaver
        st.session_state.memory_saver = MemorySaver()

# Create PocketAgentApp with persistent memory
@st.cache_resource
def create_pocket_app(_llm, _emb, _rag_agent, _mcp_agent, _memory_saver):
    return PocketAgentApp(llm=_llm, emb=_emb, rag_agent=_rag_agent, mcp_agent=_mcp_agent, memory_saver=_memory_saver)

if "OPENAI_API_KEY" not in os.environ:
    api_key = st.text_input("Bitte geben Sie Ihren OpenAI API-Schlüssel ein:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API-Schlüssel wurde gesetzt.")

if "OPENAI_API_KEY" in os.environ:
     #-------------------------------------
    # MCPAgent setup
    # Using langchain_mcp_adapters to create a MultiServerMCPClient
    from langchain_mcp_adapters.client import MultiServerMCPClient
    mcp_client = MultiServerMCPClient(
        {
            "math": {
                "command": MCP_SERVER_CMD,
                "args": MCP_SERVER_ARGS,
                "transport": "stdio",
            }
        }
    )
    tools = asyncio.run(mcp_client.get_tools())
    mcp_agent = MCPAgent(llm=llm,tools=tools)
    
    #-------------------------------------
   # RagAgent is now imported from pocketagent
    from pocketagent import RAGBot, RagAgent
    rag_agent = RagAgent(RAGBot(
        llm=llm,
        emb=emb,
        md_path_str=MD_PATH_STR,
        faiss_path=FAISS_PATH
    ))

    # Initialize session state BEFORE using it
    initialize_session()
    
    pocket_app = create_pocket_app(llm, emb, rag_agent, mcp_agent, st.session_state.memory_saver)

    st.title("PocketAgent Chatbot")
    st.write("Stellen Sie Ihre Fragen an den PocketAgent!")

    # Add a sidebar with clear chat button
    with st.sidebar:
        if st.button("🗑️ Chat verlöschen"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())  # Generate new session ID
            # Reset the memory saver to clear conversation history
            from langgraph.checkpoint.memory import MemorySaver
            st.session_state.memory_saver = MemorySaver()
            st.cache_resource.clear()  # Clear the cached PocketAgentApp
            st.rerun()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Was möchten Sie wissen?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Denke nach..."):
                response = asyncio.run(run_pocketagent_chat(prompt, thread_id=st.session_state.session_id))
                st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
