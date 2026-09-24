"""Fake LLM and tool used to test the graph without external services."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import ConfigDict


class FakeChatModel(BaseChatModel):
    """A chat model that replays a scripted list of AI messages."""

    responses: list[AIMessage] = []
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            raise AssertionError("FakeChatModel ran out of scripted responses.")
        response = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> Any:
        """Return a runnable that replays the next scripted response.

        ``prompt | llm.bind_tools(tools)`` therefore behaves like a real
        model call without network or API access.
        """
        from langchain_core.runnables import RunnableLambda

        def _replay(_input: Any) -> AIMessage:
            if not self.responses:
                raise AssertionError("FakeChatModel ran out of scripted responses.")
            return self.responses.pop(0)

        return RunnableLambda(_replay)


class FakeTool(BaseTool):
    """A trivial tool used to exercise the tool loop."""

    name: str = "fake_tool"
    description: str = "Returns 'result:<query>' for a query string."

    def _run(self, query: str) -> str:
        return f"result:{query}"

    async def _arun(self, query: str) -> str:
        return f"result:{query}"