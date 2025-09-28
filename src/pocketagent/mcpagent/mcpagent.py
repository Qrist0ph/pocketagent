from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

# https://github.com/langchain-ai/langchain-mcp-adapters
class MCPAgent:
    def __init__(self, llm,tools):
        self.llm = llm       
        self.tools = tools

   

    def get_graph(self):
        model = self.llm

        # 4) Define the model node (sync function is fine inside the async graph)
        def call_model(state: MessagesState,tools):
            response = model.bind_tools(tools).invoke(state["messages"])
            return {"messages": response}

        # 5) Build the graph (be sure to NAME your nodes)
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", lambda state: call_model(state, self.tools))
        builder.add_node("tools", ToolNode(self.tools))

        builder.add_edge(START, "call_model")
        builder.add_conditional_edges("call_model", tools_condition)
        builder.add_edge("tools", "call_model")

        graph = builder.compile()
        return graph
