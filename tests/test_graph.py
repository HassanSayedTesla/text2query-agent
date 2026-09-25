"""Tests for the LangGraph agent orchestration."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.graph import build_graph, is_tool_call
from tests.fakes import FakeChatModel, FakeTool

TOOL_CALL_ID = "call-test-1"


def test_graph_answers_without_tools() -> None:
    app = build_graph(FakeChatModel(responses=[AIMessage(content="Hello!")]), tools=[])
    result = app.invoke({"messages": [HumanMessage(content="hi")]})
    assert result["messages"][-1].content == "Hello!"


def test_graph_tool_loop_executes_tools_and_returns_final_answer() -> None:
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "fake_tool", "args": {"query": "x"}, "id": TOOL_CALL_ID}
            ],
        ),
        AIMessage(content="answer 42"),
    ]
    app = build_graph(FakeChatModel(responses=responses), tools=[FakeTool()])

    result = app.invoke({"messages": [HumanMessage(content="question?")]})
    contents = [str(m.content) for m in result["messages"]]

    assert "result:x" in contents
    assert result["messages"][-1].content == "answer 42"


def test_graph_streams_tool_messages_before_final_answer() -> None:
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "fake_tool", "args": {"query": "y"}, "id": TOOL_CALL_ID}
            ],
        ),
        AIMessage(content="done"),
    ]
    app = build_graph(FakeChatModel(responses=responses), tools=[FakeTool()])

    seen_tool = False
    for step in app.stream(
        {"messages": [HumanMessage(content="where?")]}, stream_mode="values"
    ):
        for msg in step["messages"]:
            if isinstance(msg, ToolMessage):
                seen_tool = True
    assert seen_tool


def test_graph_trims_history_sent_to_llm() -> None:
    from langchain_core.runnables import RunnableLambda
    from pydantic import ConfigDict, Field

    class RecordingModel(FakeChatModel):
        seen: list[int] = Field(default_factory=list)
        model_config = ConfigDict(arbitrary_types_allowed=True)

        def bind_tools(self, tools, **kwargs):
            def _record(input_):
                self.seen.append(len(input_.messages))
                if not self.responses:
                    raise AssertionError("FakeChatModel ran out of scripted responses.")
                return self.responses.pop(0)

            return RunnableLambda(_record)

    model = RecordingModel(responses=[AIMessage(content="ok")])
    app = build_graph(model, tools=[], history_window=3)
    history = [HumanMessage(content=f"question {i}") for i in range(8)]

    app.invoke({"messages": history})

    assert model.seen == [3 + 2]


def test_is_tool_call_detects_tool_requests() -> None:
    assert is_tool_call(
        AIMessage(
            content="",
            tool_calls=[{"name": "fake_tool", "args": {}, "id": TOOL_CALL_ID}],
        )
    )
    assert not is_tool_call(AIMessage(content="no tools here"))