from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver

# reference implementation
from pocketagent import RAGBot, ReturnAgent, Agents, RouterNode, SmalltalkNode, S

class PocketAgentApp:
    def __init__(self, llm, emb, rag_bot_instance, mcp_agent_instance):        
        self.llm = llm
        self.emb = emb
        self.rag_bot_instance = rag_bot_instance
        self.router_node_instance = RouterNode(llm=self.llm)
        self.smalltalk_node_instance = SmalltalkNode(llm=self.llm)
        self.agents_instance = Agents(llm=self.llm)
        self.memory_saver = MemorySaver()
        self.return_agent_instance = ReturnAgent(llm=self.llm, checkpointer=self.memory_saver)
        self.mcp_agent_instance = mcp_agent_instance

        def return_data_output_node(state: S) -> S:
            thread_id = state.get("thread_id", "demo")
            self.return_agent_instance.set_thread_id(thread_id)
            form_data = self.return_agent_instance._get_form(thread_id)
            answers = form_data.get("answers", {})
            required_fields = ["email", "order_number"]
            all_fields_filled = all(field in answers and answers[field] for field in required_fields)
            confirmed = form_data.get("confirmed", False)
            confirmed = True
            if all_fields_filled and confirmed:
                print("=== RETOUR-DATEN VOLLSTÄNDIG ===")
                print(f"Thread ID: {thread_id}")
                print(f"E-Mail: {answers.get('email')}")
                print(f"Bestellnummer: {answers.get('order_number')}")
                print("Status: Retoure wird bearbeitet")
                print("================================")
            else:
                print("=== RETOUR-DATEN UNVOLLSTÄNDIG ===")
                print(f"Thread ID: {thread_id}")
                print(f"E-Mail: {'✓' if 'email' in answers else '✗'} {answers.get('email', 'Nicht angegeben')}")
                print(f"Bestellnummer: {'✓' if 'order_number' in answers else '✗'} {answers.get('order_number', 'Nicht angegeben')}")
                print(f"Bestätigt: {'✓' if confirmed else '✗'}")
                print("Warten auf weitere Eingaben...")
                print("===================================")
            return state
        self.return_data_output_node = return_data_output_node

    
      

        def rag_agent_wrapper_node(state: S) -> S:
            msgs = state["messages"]
            user_msg = next((m.content for m in reversed(msgs) if getattr(m, "type", None) == "human"), "")
            inp = {"messages": [{"role": "user", "content": user_msg}]}
            cfg = {"configurable": {"thread_id": "tools-graph"}}
            out = self.rag_bot_instance.build_graph().invoke(inp, cfg)
            from langchain_core.messages import AIMessage
            state["messages"].append(AIMessage(content=out["answer"]))
            return state
        self.rag_agent_wrapper_node = rag_agent_wrapper_node

        g = StateGraph(S)
        g.add_node("router", self.router_node_instance.process)
        g.add_node("smalltalk", self.smalltalk_node_instance.process)
        g.add_node("weather_agent", self.agents_instance.weather_react_agent)
        g.add_node("weather_tools", self.agents_instance.weather_tools)
        g.add_node("travel_agent", self.agents_instance.travel_react_agent)
        g.add_node("travel_tools", self.agents_instance.travel_tools)
        g.add_node("rag_agent", self.rag_agent_wrapper_node)
        g.add_node("return_agent", self.return_agent_instance.react_agent)
        g.add_node("return_output", self.return_data_output_node)
        g.add_node("mcp_agent", self.mcp_agent_instance)
        g.add_edge(START, "router")
        g.add_conditional_edges("router", lambda s: s["intent"],
                                {"weather": "weather_agent", "travel": "travel_agent", "chitchat": "smalltalk", "rag": "rag_agent", "return_agent": "return_agent", "mcp_agent": "mcp_agent"})
        g.add_edge("weather_agent", "weather_tools")
        # g.add_edge("weather_tools", "weather_agent")
        g.add_edge("weather_agent", END)
        g.add_edge("travel_agent", "travel_tools")
        # g.add_edge("travel_tools", "travel_agent")
        g.add_edge("travel_agent", END)
        g.add_edge("rag_agent", END)
        g.add_edge("smalltalk", END)
        g.add_edge("return_agent", "return_output")
        g.add_edge("return_output", END)

        self.app = g.compile(checkpointer=self.memory_saver)

    def chat(self, text: str, thread_id: str = "demo"):
        config = {"configurable": {"thread_id": thread_id}}
        out = self.app.invoke({"messages": [HumanMessage(content=text)], "intent": "", "thread_id": thread_id}, config)
        return out["messages"][-1].content



    
        # Pass BaseMessage, not dict; add_messages will append it to prior checkpointed messages

