
# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.messages import HumanMessage

# PocketAgent imports
from pocketagent import RouterNode, SmalltalkNode, S
from pocketagent import ReturnAgent
from pocketagent.TravelAgent.agent import TravelWeatherAgent



class PocketAgentApp:
    def __init__(self, llm, emb, rag_agent, mcp_agent, memory_saver=None):
        self.llm = llm
        self.emb = emb
        self.rag_agent = rag_agent
        self.router_node_instance = RouterNode(llm=self.llm)
        self.smalltalk_node_instance = SmalltalkNode(llm=self.llm)
        self.travel_weather_agent = TravelWeatherAgent(llm=self.llm)
        self.memory_saver = memory_saver if memory_saver is not None else MemorySaver()
        self.return_agent_instance = ReturnAgent(llm=self.llm, checkpointer=self.memory_saver)
        self.mcp_agent_instance = mcp_agent

       

    def chat(self, text: str, thread_id: str = "demo"):
        config = {"configurable": {"thread_id": thread_id}}
        out = self.app.invoke({"messages": [HumanMessage(content=text)], "intent": "", "thread_id": thread_id}, config)
        return out["messages"][-1].content

    def get_graph(self):
        g = StateGraph(S)
        g.add_node("router", self.router_node_instance.process)
        g.add_node("smalltalk", self.smalltalk_node_instance.get_graph())
        g.add_node("travel_weather", self.travel_weather_agent.get_graph())
        g.add_node("rag_agent", self.rag_agent.get_graph())
        g.add_node("return_agent_subgraph", self.return_agent_instance.get_graph())
        g.add_node("mcp_agent", self.mcp_agent_instance.get_graph())
        g.add_edge(START, "router")
        g.add_conditional_edges("router", lambda s: s["intent"],
            {"weather": "travel_weather", "travel": "travel_weather", "chitchat": "smalltalk", "rag": "rag_agent", "return_agent": "return_agent_subgraph", "mcp_agent": "mcp_agent"})
        g.add_edge("rag_agent", END)
        g.add_edge("smalltalk", END)
        return  g.compile(checkpointer=self.memory_saver)
    
    
    async def invoke(self,query) -> str:
        print( f"Invoking PocketAgentApp with query: {query}" )
        config = {"configurable": {"thread_id": "thread_id"}}
        out  = await self.get_graph().ainvoke({"messages": [HumanMessage(content=query)]},config) 
        return out["messages"][-1].content