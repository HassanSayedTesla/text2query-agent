"""LangGraph orchestration for the text-to-query agent.

The graph has two nodes:

* ``agent``  — invokes the tool-augmented LLM to either answer or emit a tool call.
* ``tools``  — executes any MongoDB tool calls requested by the agent.

Execution flow: ``START -> agent -> (tools <-> agent) ... -> END`` using the
prebuilt ``tools_condition`` router, exactly as in the course notebooks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from app.agent.prompt import build_prompt
from app.agent.state import GraphState


def build_graph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    top_k: int = 5,
    memory_system_prompt: str | None = None,
    checkpointer: Any = None,
    history_window: int = 12,
) -> Any:
    """Build and compile the LangGraph agent.

    Args:
        llm: The chat model the agent reasons with (already tool-augmented).
        tools: MongoDB toolkit tools the agent can call.
        top_k: Number of rows the agent should reason over.
        memory_system_prompt: Optional stand-in for the default memory prompt.
        checkpointer: Optional LangGraph checkpointer (e.g. MongoDBSaver) to
            enable multi-turn short-term memory.
        history_window: Maximum number of recent messages sent to the LLM on
            each turn. The full conversation is still persisted by the
            checkpointer (and shown in the UI); only the request sent to the
            model is trimmed, keeping provider token limits in check.

    Returns:
        A compiled ``CompiledStateGraph``.
    """
    tool_map: dict[str, BaseTool] = {tool.name: tool for tool in tools}
    prompt = build_prompt(
        tool_names=", ".join(tool.name for tool in tools),
        top_k=top_k,
    )
    if memory_system_prompt is not None:
        prompt = prompt.partial(memory_instruction=memory_system_prompt)

    llm_with_tools: Callable = prompt | llm.bind_tools(tools)

    def agent_node(state: GraphState) -> dict[str, list[BaseMessage]]:
        messages = state["messages"]
        if len(messages) > history_window:
            messages = messages[-history_window:]
        result = llm_with_tools.invoke(messages)
        return {"messages": [result]}

    def tool_node(state: GraphState) -> dict[str, list[BaseMessage]]:
        result: list[BaseMessage] = []
        last_message = state["messages"][-1]
        for tool_call in last_message.tool_calls:
            tool = tool_map[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(
                ToolMessage(
                    content=observation, tool_call_id=tool_call["id"]
                )
            )
        return {"messages": result}

    graph = StateGraph(GraphState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )

    return graph.compile(checkpointer=checkpointer)


def is_tool_call(message: BaseMessage) -> bool:
    """Return True when an AI message requests a tool call."""
    return isinstance(message, AIMessage) and bool(getattr(message, "tool_calls", []))