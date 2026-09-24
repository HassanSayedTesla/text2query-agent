"""Unit tests for the TextToQueryAgent facade (dependencies mocked)."""

import pytest

from app.agent import agent as agent_module
from app.agent.agent import TextToQueryAgent


class _FakeToolkit:
    def __init__(self) -> None:
        self.tools = []

    def get_tools(self) -> list:
        return self.tools


def test_agent_requires_mongodb_uri(monkeypatch) -> None:
    monkeypatch.delenv("MONGODB_URI", raising=False)
    with pytest.raises(ValueError, match="MONGODB_URI"):
        TextToQueryAgent(mongodb_uri=None)


def test_agent_wires_tools_and_checkpointer(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    spy = {}

    def fake_toolkit(*args, **kwargs):
        return _FakeToolkit()

    def fake_checkpointer(*args, **kwargs):
        return "checkpointer"

    def fake_build_graph(llm, tools, top_k=5, memory_system_prompt=None, checkpointer=None):
        spy["llm"] = llm
        spy["tools"] = tools
        spy["top_k"] = top_k
        spy["checkpointer"] = checkpointer
        return "app"

    monkeypatch.setattr(agent_module, "get_toolkit", fake_toolkit)
    monkeypatch.setattr(agent_module, "get_checkpointer", fake_checkpointer)
    monkeypatch.setattr(agent_module, "build_graph", fake_build_graph)

    agent = TextToQueryAgent(mongodb_uri="mongodb://localhost:27017")

    assert agent.checkpointer == "checkpointer"
    assert agent.app == "app"
    assert spy["tools"] == []
    assert spy["top_k"] == 5
    assert spy["checkpointer"] == "checkpointer"


def test_list_collections_returns_empty_without_tool(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def fake_toolkit(*args, **kwargs):
        return _FakeToolkit()

    def fake_checkpointer(*args, **kwargs):
        return "checkpointer"

    monkeypatch.setattr(agent_module, "get_toolkit", fake_toolkit)
    monkeypatch.setattr(agent_module, "get_checkpointer", fake_checkpointer)
    monkeypatch.setattr(agent_module, "build_graph", lambda *a, **k: "app")

    agent = TextToQueryAgent(mongodb_uri="mongodb://localhost:27017")
    assert agent.list_collections() == []