
from ..state_types import S

class RagAgent:
    def __init__(self, rag_bot_instance):
        self.rag_bot_instance = rag_bot_instance
    # Wrapper around invoke to fit the node signature
    def _process(self, state: S) -> S:
        msgs = state["messages"]
        user_msg = next((m.content for m in reversed(msgs) if getattr(m, "type", None) == "human"), "")
        inp = {"messages": [{"role": "user", "content": user_msg}]}
        cfg = {"configurable": {"thread_id": "tools-graph"}}
        out = self.rag_bot_instance.build_graph().invoke(inp, cfg)
        from langchain_core.messages import AIMessage
        state["messages"].append(AIMessage(content=out["answer"]))
        return state

    def get_graph(self):
        """
        Returns a compiled LangGraph StateGraph object using _process as the node.
        """
        from langgraph.graph import StateGraph, START, END
        from ..state_types import S
        graph = StateGraph(S)
        graph.add_node("ragagent", self._process)
        graph.add_edge(START, "ragagent")
        graph.add_edge("ragagent", END)
        return graph.compile()
