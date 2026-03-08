
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """
    Gibt das Wetter für einen angegebenen Ort zurück (Demo).
    Args:
        location (str): Der Ort, für den das Wetter abgefragt wird.
    Returns:
        str: Wetterbeschreibung für den Ort.
    """
    print(f"Fetching weather for {location}...")
    return f"Wetter in {location}: 24°C"

@tool
def find_hotel(city: str) -> str:
    """
    Findet ein Hotel in einer angegebenen Stadt (Demo).
    Args:
        city (str): Die Stadt, in der ein Hotel gesucht wird.
    Returns:
        str: Name des Top-Hotels in der Stadt.
    """
    print(f"Finding hotel in {city}...")
    return f"Top-Hotel in {city}: Hotel Alpha"

class TravelWeatherAgent:
    """
    Agent that routes based on S.Intent to either travel or weather branch.
    """
    def __init__(self, llm):
        if llm is None:
            raise ValueError("llm parameter is required for TravelWeatherAgent")
        self.llm = llm
        self.weather_react_agent = create_react_agent(
            self.llm,
            tools=[get_weather]
        )
        self.travel_react_agent = create_react_agent(
            self.llm,
            tools=[find_hotel]
        )

    def _route_by_intent(self, state) -> str:
        intent = state.get("intent", "weather")
        dest = intent if intent in ("weather", "travel") else "weather"
        print(f"TravelWeather routing to: {dest}")
        return dest

    def get_graph(self):
        from langgraph.graph import StateGraph, START, END
        from ..state_types import S
        graph = StateGraph(S)
        graph.add_node("weather", self.weather_react_agent)
        graph.add_node("travel", self.travel_react_agent)
        graph.add_conditional_edges(START, self._route_by_intent, {"weather": "weather", "travel": "travel"})
        graph.add_edge("weather", END)
        graph.add_edge("travel", END)
        return graph.compile()
