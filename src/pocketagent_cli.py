# export OPENAI_API_KEY=sk-123456789abcdef
# pip install -e .
# python3 src/pocketagent_cli.py 

from pocketagent.pocketagent_app import PocketAgentApp
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pocketagent import RAGBot, RagAgent
from langchain_core.messages import HumanMessage
import asyncio

# === CONFIGURATION STRINGS ===
import os
WORKING_DIR = os.getcwd()
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
MD_PATH_STR = os.path.join(WORKING_DIR, "product_faq.md")
FAISS_PATH = os.path.join(WORKING_DIR, "faiss_index")
MCP_SERVER_COMMAND = "python3"
MCP_SERVER_PATH = os.path.join(WORKING_DIR, "src", "mcpserver", "server.py")

from pocketagent.mcpagent.mcpagent import MCPAgent


def test_chat(app):
    print(app.chat("Was geht?", thread_id="demo"))
    print(app.chat("Finde mir ein Hotel in München", thread_id="demo"))
    print(app.chat("Und wie ist das Wetter dort?", thread_id="demo"))
    print(app.chat("Und sonst, wie ist die Lage?", thread_id="demo"))
    print(app.chat("ich habe ein frage zum produkt travel rucksack?", thread_id="demo"))
    print(app.chat("hast du auch gartenstühle", thread_id="demo"))
    print(app.chat("Was geht, münchen is tooll, oder?", thread_id="demo"))
    print(app.chat("Finde mir ein Hotel in ...", thread_id="demo"))
    print(app.chat("Und wie ist das Wetter dort?", thread_id="demo"))
    print(app.chat("Ich möchte eine retoure machen", thread_id="demo"))
    print(app.chat("Christoph@test.com", thread_id="demo"))
    print(app.chat("3234", thread_id="demo"))



def chat_via_stdin(app, thread_id: str = "demo"):
        import sys
        print("Geben Sie Ihre Nachrichten ein (Ctrl+D zum Beenden):")
        graph = app.get_graph()
        for line in sys.stdin:
            user_input = line.strip()
            if user_input:
                # response = app.chat(user_input, thread_id=thread_id)
                config = {"configurable": {"thread_id": thread_id}}
                out  = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content=user_input)]},config) )
                bar = out["messages"][-1].content
                print(f"Agent: {bar}")


def main():
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Bitte geben Sie Ihren OpenAI API-Schlüssel ein: ")
        os.environ["OPENAI_API_KEY"] = api_key
        print("API-Schlüssel wurde gesetzt.")
    llm = ChatOpenAI(model=MODEL_NAME)
    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    #-------------------------------------
    # MCPAgent setup
    # Using langchain_mcp_adapters to create a MultiServerMCPClient
    from langchain_mcp_adapters.client import MultiServerMCPClient
    mcp_client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": ["/mnt/c/repos/vercel.pocketagent/src/mcpserver/server.py"],
                "transport": "stdio",
            }
        }
    )
    
    tools = asyncio.run(mcp_client.get_tools())
    mcp_agent = MCPAgent(llm=llm, tools=tools)
    #-------------------------------------

    #-------------------------------------
    # RagAgent setup
    rag_agent = RagAgent(RAGBot(
        llm=llm,
        emb=emb,
        md_path_str=MD_PATH_STR,
        faiss_path=FAISS_PATH
    ))
    #-------------------------------------
        
    # here we pass the mcp_agent and rag_agent instance to PocketAgentApp via dependecy injection
    app = PocketAgentApp(llm=llm, emb=emb, rag_agent=rag_agent, mcp_agent=mcp_agent)
    #test_chat()
    chat_via_stdin(app, thread_id="demo")

if __name__ == "__main__":
    main()
