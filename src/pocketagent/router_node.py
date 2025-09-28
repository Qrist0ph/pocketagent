
from langchain_core.messages import HumanMessage, SystemMessage
from .state_types import S


class RouterNode:
    """Node responsible for routing user intents to appropriate agents."""

    def __init__(self, llm):
        if llm is None:
            raise ValueError("llm parameter is required for RouterNode")
        self.llm = llm

    def process(self, state: S) -> S:
        """
        Process the current state and classify user intent.

        Args:
            state: Current conversation state

        Returns:
            Updated state with classified intent
        """
        # Don't replace messages; just read from it
        msgs = state["messages"]
        if not msgs:
            raise ValueError("router_node: messages is empty")
        q = "\n".join([f"{m.type}: {m.content}" for m in msgs])
        # print(q)

        # Try to use LLM for classification, fallback to keyword-based if not available
        try:
            label = self.llm.invoke([
                SystemMessage(content=(
                    "Klassifiziere Intent als 'weather', 'travel', 'rag' (für Produktanfragen aus meinem Online-Shop), 'return_agent' oder 'mcp_agent' (für Rechenoperationen, Mathematik, Berechnungen, etc.) oder 'chitchat'.\n"
                    "Antworte NUR mit dem Label.\n"
                    "Wenn die Frage sich auf Produkte, Artikel, Preise, Verfügbarkeit oder Details aus meinem Online-Shop bezieht, wähle 'rag'.\n"
                    "Falls eine Retoure für den Onlineshop gemacht werden soll wähle 'return_agent'.\n"
                    "Wenn nach Rechenoperationen, Mathematik, Berechnungen, Zahlen, Formeln, etc. gefragt wird, wähle 'mcp_agent'."
                )),
                HumanMessage(content=q),
            ]).content.strip().lower()
        except Exception as e:
            print(f"Warning: Could not use LLM for classification: {e}")
            # Fallback: simple keyword-based classification
            q_lower = q.lower()
            if any(word in q_lower for word in ["wetter", "weather", "temperature"]):
                label = "weather"
            elif any(word in q_lower for word in ["hotel", "travel", "reise"]):
                label = "travel"
            elif any(word in q_lower for word in ["produkt", "artikel", "preis", "shop"]):
                label = "rag"
            elif any(word in q_lower for word in ["retoure", "return", "bestellung"]):
                label = "return_agent"
            elif any(word in q_lower for word in ["rechnen", "mathe", "mathematik", "berechnung", "calculation", "math", "formel", "summe", "plus", "minus", "mal", "geteilt", "+", "-", "*", "/", "%", "zahl", "zahlen"]):
                label = "mcp_agent"
            else:
                label = "chitchat"

        if "chitchat" in label or "smalltalk" in label:
            state["intent"] = "chitchat"
            state["chitchat"] = True
        elif "rag" in label:
            state["intent"] = "rag"
            state["chitchat"] = False
        elif "return_agent" in label:
            state["intent"] = "return_agent"
            state["chitchat"] = False
        elif "mcp_agent" in label:
            state["intent"] = "mcp_agent"
            state["chitchat"] = False
        else:
            state["intent"] = "travel" if ("travel" in label or "hotel" in label) else "weather"
            state["chitchat"] = False

        print(f"Classified intent: {state['intent']}")
        return state

