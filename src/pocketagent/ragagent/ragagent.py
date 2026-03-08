
from langchain_core.messages import AIMessage
from ..state_types import S


class RagAgent:
    def __init__(self, rag_bot_instance):
        self.rag_bot_instance = rag_bot_instance

    def _process(self, state: S) -> dict:
        msgs = state["messages"]
        user_msg = next((m.content for m in reversed(msgs) if getattr(m, "type", None) == "human"), "")
        inp = {"messages": [{"role": "user", "content": user_msg}]}
        cfg = {"configurable": {"thread_id": "tools-graph"}}
        out = self.rag_bot_instance.build_graph().invoke(inp, cfg)
        return {"messages": [AIMessage(content=out["answer"])]}

    def get_graph(self):
        from langgraph.graph import StateGraph, START, END
        graph = StateGraph(S)
        graph.add_node("ragagent", self._process)
        graph.add_edge(START, "ragagent")
        graph.add_edge("ragagent", END)
        return graph.compile()
