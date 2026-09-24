"""Tests for prompt construction."""

from langchain_core.prompts import MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_mongodb.agent_toolkit import MONGODB_AGENT_SYSTEM_PROMPT

from app.agent.prompt import MEMORY_INSTRUCTION, build_prompt


def test_build_prompt_uses_mongodb_system_prompt() -> None:
    prompt = build_prompt("mongodb_query")
    system_templates = [
        m.prompt.template
        for m in prompt.messages
        if isinstance(m, SystemMessagePromptTemplate)
    ]
    assert MONGODB_AGENT_SYSTEM_PROMPT in system_templates
    assert any(MEMORY_INSTRUCTION in template for template in system_templates)


def test_build_prompt_partials_tool_names_and_top_k() -> None:
    prompt = build_prompt("mongodb_query, mongodb_schema", top_k=3)
    assert prompt.partial_variables["tool_names"] == "mongodb_query, mongodb_schema"
    assert prompt.partial_variables["top_k"] == 3


def test_build_prompt_has_message_placeholder() -> None:
    prompt = build_prompt("mongodb_query")
    assert any(
        isinstance(m, MessagesPlaceholder) and m.variable_name == "messages"
        for m in prompt.messages
    )