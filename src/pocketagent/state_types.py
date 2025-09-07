"""
Shared type definitions for the PocketAgent application.
"""

from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages, AnyMessage


class S(TypedDict):
    """State type for the conversation graph."""
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    chitchat: bool
    thread_id: str
