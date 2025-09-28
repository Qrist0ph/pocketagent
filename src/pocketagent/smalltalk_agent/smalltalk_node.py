
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from ..state_types import S

class SmalltalkNode:
    """Node responsible for handling casual conversation."""

    def __init__(self, llm):
        if llm is None:
            raise ValueError("llm parameter is required for SmalltalkNode")
        self.llm = llm

    def _process(self, state: S) -> S:
        """
         Process the current state and generate a smalltalk response.

        Args:
            state: Current conversation state

        Returns:
            Updated state with AI response added
        """
        msgs = state["messages"]
        last_user = next((m for m in reversed(msgs) if m.type == "human"), msgs[-1])
        q = last_user.content if isinstance(last_user.content, str) else ""

        reply = self.llm.invoke([
            SystemMessage(content="Du bist ein freundlicher Smalltalk-Assistent. Antworte locker und kurz auf Smalltalk-Fragen."),
            HumanMessage(content=q)
        ]).content

        state["messages"].append(AIMessage(content=reply))
        return state

    def get_graph(self):
        """
        Returns a compiled LangGraph StateGraph object using process() as the node.
        """
        from langgraph.graph import StateGraph, START, END
        graph = StateGraph(S)
        graph.add_node("smalltalk", self._process)
        graph.add_edge(START, "smalltalk")
        graph.add_edge("smalltalk", END)
        return graph.compile()
