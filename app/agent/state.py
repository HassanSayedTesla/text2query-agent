"""LangGraph state shared between the agent and tool nodes."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """Graph state: a rolling list of chat and tool messages."""

    messages: Annotated[list, add_messages]