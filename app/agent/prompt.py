"""Prompt construction for the text-to-query agent.

Leverages the MongoDB agent system prompt provided by
``langchain-mongodb`` (MONGODB_AGENT_SYSTEM_PROMPT) and augments it
with a short-term-memory instruction so the agent reuses information
it already retrieved in the conversation instead of re-running tools.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mongodb.agent_toolkit import MONGODB_AGENT_SYSTEM_PROMPT

MEMORY_INSTRUCTION = (
    "IMPORTANT: Always start by checking your memory for relevant information "
    "before calling any tools. Do not re-run tools unless absolutely necessary. "
    "If you are not able to get enough information using the tools, reply with "
    "I DON'T KNOW. You have access to the following tools: {tool_names}."
)


def build_prompt(tool_names: str, top_k: int = 5) -> ChatPromptTemplate:
    """Build the chat prompt used by the agent.

    Args:
        tool_names: Comma-separated list of tool names the agent can call.
        top_k: Number of rows the agent should consider when reasoning.

    Returns:
        A ``ChatPromptTemplate`` with ``top_k`` and ``tool_names`` pre-filled.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MONGODB_AGENT_SYSTEM_PROMPT),
            ("system", MEMORY_INSTRUCTION),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt.partial(top_k=top_k, tool_names=tool_names)