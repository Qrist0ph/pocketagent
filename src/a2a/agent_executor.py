# https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents/langgraph
import logging
import os

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (   
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError



from pocketagent.pocketagent_app import PocketAgentApp
from pocketagent import RAGBot, RagAgent
from pocketagent.mcpagent.mcpagent import MCPAgent
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def GetAgent():
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    import os
    WORKING_DIR = os.getcwd()
    MODEL_NAME = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"
    MD_PATH_STR = os.path.join(WORKING_DIR, "product_faq.md")
    FAISS_PATH = os.path.join(WORKING_DIR, "faiss_index")
    MCP_SERVER_COMMAND = "python3"
    MCP_SERVER_PATH = os.path.join(WORKING_DIR, "src", "mcpserver", "server.py")
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Bitte geben Sie Ihren OpenAI API-Schlüssel ein: ")
        os.environ["OPENAI_API_KEY"] = api_key
        print("API-Schlüssel wurde gesetzt.")
    llm = ChatOpenAI(model=MODEL_NAME)
    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
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
    return app
    #test_chat()
    
class CurrencyAgentExecutor(AgentExecutor):
    """Currency Conversion AgentExecutor Example."""

    def __init__(self):
        # self.agent = CurrencyAgent()
        self.agent = GetAgent()


    # execution per invoke
    # for streaming check https://github.com/a2aproject/a2a-samples/blob/main/samples/python/agents/langgraph/app/agent_executor.py#L33
    async def execute(self,context: RequestContext,event_queue: EventQueue) -> None:
        result = await self.agent.invoke("hello hast du rucksäcke" )
        await event_queue.enqueue_event(new_agent_text_message(result))
    

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
