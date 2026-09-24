"""High-level agent facade used by the UI and scripts."""

from __future__ import annotations

import os
from collections.abc import Iterator

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

from app.agent.graph import build_graph
from app.agent.prompt import MEMORY_INSTRUCTION
from app.db.connection import get_checkpointer, get_toolkit
from app.settings import Settings


class TextToQueryAgent:
    """A MongoDB text-to-query agent with multi-turn memory.

    Wraps the MongoDB database toolkit, LangGraph orchestration and the
    MongoDBSaver checkpointer behind a small, UI-friendly interface.

    Example:
        agent = TextToQueryAgent.from_settings(Settings())
        for step in agent.stream("How many theaters are in California?", thread_id):
            print(step)
    """

    def __init__(
        self,
        mongodb_uri: str | None = None,
        database: str = "sample_mflix",
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.0,
        top_k: int = 5,
    ) -> None:
        self.mongodb_uri = mongodb_uri or os.environ.get("MONGODB_URI")
        if not self.mongodb_uri:
            raise ValueError("MONGODB_URI must be provided or set in the environment.")

        self.database = database
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.toolkit = get_toolkit(self.mongodb_uri, database, self.llm)
        self.tools = self.toolkit.get_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.top_k = top_k
        self.checkpointer = get_checkpointer(self.mongodb_uri)
        self.app = build_graph(
            llm=self.llm,
            tools=self.tools,
            top_k=self.top_k,
            memory_system_prompt=MEMORY_INSTRUCTION,
            checkpointer=self.checkpointer,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> TextToQueryAgent:
        """Create an agent from a :class:`app.settings.Settings` object."""
        return cls(
            mongodb_uri=settings.mongodb_uri,
            database=settings.database_name,
            model=settings.groq_model,
            temperature=settings.temperature,
            top_k=settings.top_k,
        )

    def list_collections(self) -> list[str]:
        """List the collections available in the target database."""
        tool = self.tool_map.get("mongodb_list_collections")
        if tool is None:
            return []
        return sorted(tool.invoke({}).splitlines())

    def stream(self, user_input: str, thread_id: str) -> Iterator[BaseMessage]:
        """Run one agent turn, yielding the latest message after each step.

        Args:
            user_input: The user's natural-language question.
            thread_id: Conversation thread used by the checkpointer to
                retrieve/store short-term memory.

        Yields:
            The most recent message of the graph state after every node runs.
        """
        config = {"configurable": {"thread_id": thread_id}}
        for step in self.app.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values",
        ):
            yield step["messages"][-1]

    def stream_full(self, user_input: str, thread_id: str) -> Iterator[list[BaseMessage]]:
        """Run one agent turn, yielding the full message state after each step.

        This exposes intermediate tool calls and results so a UI can render
        the agent's reasoning process in real time.

        Yields:
            The full list of conversation messages after every node runs.
        """
        config = {"configurable": {"thread_id": thread_id}}
        for step in self.app.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values",
        ):
            yield step["messages"]

    def ask(self, user_input: str, thread_id: str) -> BaseMessage:
        """Run one agent turn and return the final message."""
        last_message: BaseMessage | None = None
        for step in self.stream(user_input, thread_id):
            last_message = step
        if last_message is None:
            raise RuntimeError("The agent produced no messages.")
        return last_message