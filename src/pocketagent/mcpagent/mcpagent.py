from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from ..state_types import S


class MCPAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def get_graph(self):
        def call_model(state: S):
            response = self.llm.bind_tools(self.tools).invoke(state["messages"])
            return {"messages": [response]}

        builder = StateGraph(S)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", ToolNode(self.tools))

        builder.add_edge(START, "call_model")
        builder.add_conditional_edges("call_model", tools_condition)
        builder.add_edge("tools", "call_model")

        return builder.compile()
