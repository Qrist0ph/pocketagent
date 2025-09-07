# export OPENAI_API_KEY=sk-123456789abcdef
# python3 src/pocketagent_cli.py 

from pocketagent_app import PocketAgentApp
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pocketagent import RAGBot


# === CONFIGURATION STRINGS ===
import os
WORKING_DIR = os.getcwd()
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
MD_PATH_STR = os.path.join(WORKING_DIR, "product_faq.md")
FAISS_PATH = os.path.join(WORKING_DIR, "faiss_index")
MCP_SERVER_COMMAND = "python3"
MCP_SERVER_PATH = os.path.join(WORKING_DIR, "src", "mcpserver", "server.py")



# https://github.com/langchain-ai/langchain-mcp-adapters
def get_mcp_reactagent(llm):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.client import ClientSession, load_mcp_tools
    from langgraph.prebuilt import create_react_agent

    server_params = StdioServerParameters(
        command=MCP_SERVER_COMMAND,
        args=[MCP_SERVER_PATH]
    )
    async def mcp_main(llm):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                agent = create_react_agent(llm, tools)
                return agent
    import asyncio
    return asyncio.run(mcp_main(llm))


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
        for line in sys.stdin:
            user_input = line.strip()
            if user_input:
                response = app.chat(user_input, thread_id=thread_id)
                print(f"Agent: {response}")


import os



def main():
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Bitte geben Sie Ihren OpenAI API-Schlüssel ein: ")
        os.environ["OPENAI_API_KEY"] = api_key
        print("API-Schlüssel wurde gesetzt.")

    llm = ChatOpenAI(model=MODEL_NAME)
    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    rag_bot_instance = RAGBot(
		llm=llm,
		emb=emb,
		md_path_str=MD_PATH_STR,
		faiss_path=FAISS_PATH
	)
    mcp_agent = get_mcp_reactagent(llm)
    app = PocketAgentApp(llm=llm, emb=emb, rag_bot_instance=rag_bot_instance, mcp_agent_instance=mcp_agent)
    #test_chat()
    chat_via_stdin(app, thread_id="demo")

if __name__ == "__main__":
    main()
