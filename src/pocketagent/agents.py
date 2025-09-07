from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_openai import ChatOpenAI
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
    return f"Top-Hotel in {city}: Hotel Alpha"


class Agents:
    """Container class for weather and travel agents with their tools."""

    def __init__(self, llm):
        if llm is None:
            raise ValueError("llm parameter is required for Agents")
        
        self.llm = llm

        # Initialize agents
        self.weather_react_agent = create_react_agent(
            self.llm,
            tools=[get_weather]
        )
        self.travel_react_agent = create_react_agent(
            self.llm,
            tools=[find_hotel]
        )

        # Initialize tool nodes
        self.weather_tools = ToolNode(tools=[get_weather])
        self.travel_tools = ToolNode(tools=[find_hotel])

