
# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# PocketAgent imports
from pocketagent import RouterNode, SmalltalkNode, S
from pocketagent import ReturnAgent
from pocketagent.TravelAgent.agent import TravelWeatherAgent


class PocketAgentApp:
    def __init__(self, llm, emb, rag_agent, mcp_agent, memory_saver=None):
        self.llm = llm
        self.emb = emb
        self.rag_agent = rag_agent
        self.memory_saver = memory_saver if memory_saver is not None else MemorySaver()
        self.router_node_instance = RouterNode(llm=self.llm)
        self.smalltalk_node_instance = SmalltalkNode(llm=self.llm)
        self.travel_weather_agent = TravelWeatherAgent(llm=self.llm)
        self.return_agent_instance = ReturnAgent(llm=self.llm, checkpointer=self.memory_saver)
        self.mcp_agent_instance = mcp_agent
        self._compiled_graph = None

    def get_graph(self):
        if self._compiled_graph is None:
            g = StateGraph(S)
            g.add_node("router", self.router_node_instance.process)
            g.add_node("smalltalk", self.smalltalk_node_instance.get_graph())
            g.add_node("travel_weather", self.travel_weather_agent.get_graph())
            g.add_node("rag_agent", self.rag_agent.get_graph())
            g.add_node("return_agent_subgraph", self.return_agent_instance.get_graph())
            g.add_node("mcp_agent", self.mcp_agent_instance.get_graph())
            g.add_edge(START, "router")
            g.add_conditional_edges("router", lambda s: s["intent"], {
                "weather": "travel_weather",
                "travel": "travel_weather",
                "chitchat": "smalltalk",
                "rag": "rag_agent",
                "return_agent": "return_agent_subgraph",
                "mcp_agent": "mcp_agent",
            })
            g.add_edge("rag_agent", END)
            g.add_edge("smalltalk", END)
            g.add_edge("travel_weather", END)
            g.add_edge("return_agent_subgraph", END)
            g.add_edge("mcp_agent", END)
            self._compiled_graph = g.compile(checkpointer=self.memory_saver)
        return self._compiled_graph

    async def invoke(self, query) -> str:
        from langchain_core.messages import HumanMessage
        config = {"configurable": {"thread_id": "default"}}
        out = await self.get_graph().ainvoke(
            {"messages": [HumanMessage(content=query)]}, config
        )
        return out["messages"][-1].content
