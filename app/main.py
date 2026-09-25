"""Streamlit chat interface for the MongoDB text-to-query agent.

Run with:

    streamlit run app/main.py

Requires ``MONGODB_URI`` (and ``GROQ_API_KEY``) either as environment
variables or entered in the sidebar.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.agent.agent import TextToQueryAgent

DEFAULT_MODELS = (
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "allam-2-7b",
)

SAMPLE_PROMPTS = (
    "Which states have the most theaters?",
    "How many movies did Arthur C. Clarke write, and which are the highest rated?",
    "List the 5 newest movies in the database.",
    "What is the average runtime of movies starring Tom Hanks?",
)


@st.cache_resource(show_spinner="Connecting to MongoDB...")
def get_agent(
    mongodb_uri: str,
    database: str,
    model: str,
    temperature: float,
    top_k: int,
) -> TextToQueryAgent:
    """Build (and cache) the agent for the current sidebar configuration."""
    return TextToQueryAgent(
        mongodb_uri=mongodb_uri,
        database=database,
        model=model,
        temperature=temperature,
        top_k=top_k,
    )


def serialize(msg: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message into plain session-state data."""
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": str(msg.content)}
    if isinstance(msg, ToolMessage):
        return {"role": "tool", "content": str(msg.content)}
    if isinstance(msg, AIMessage):
        tool_calls = [
            {"name": tc["name"], "args": tc["args"]}
            for tc in getattr(msg, "tool_calls", [])
        ]
        return {
            "role": "assistant",
            "content": str(msg.content),
            "tool_calls": tool_calls,
        }
    return {"role": "assistant", "content": str(msg.content)}


def render_stored(message: dict[str, Any], show_trace: bool) -> None:
    """Render a previously persisted message."""
    role = message["role"]
    if role == "tool":
        if show_trace:
            with st.expander("Tool result", expanded=False):
                st.code(message["content"], language="json")
        return
    with st.chat_message(role):
        st.markdown(message["content"] or "*no output*")


def render_live(msg: BaseMessage, show_trace: bool) -> None:
    """Render a message as it streams in from the graph."""
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(str(msg.content))
    elif isinstance(msg, ToolMessage):
        if show_trace:
            with st.expander(f"Tool result: {msg.tool_call_id}", expanded=False):
                st.code(str(msg.content), language="json")
    elif isinstance(msg, AIMessage):
        tool_calls = getattr(msg, "tool_calls", [])
        if tool_calls and not str(msg.content):
            with st.status("Using MongoDB tools...", expanded=False):
                for tc in tool_calls:
                    st.write(f"`{tc['name']}`")
                    st.code(str(tc["args"]), language="json")
        elif str(msg.content):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(str(msg.content))
    else:
        with st.chat_message("assistant"):
            st.markdown(str(msg.content))


def run_chat(agent: TextToQueryAgent, thread_id: str, show_trace: bool) -> None:
    """Render the conversation and handle new user input."""
    st.session_state.setdefault("_messages", [])

    for message in st.session_state["_messages"]:
        render_stored(message, show_trace)

    prompt = st.chat_input("Ask your MongoDB a question...")
    if not prompt:
        prompt = st.session_state.pop("_pending_prompt", None)
    if not prompt:
        return

    st.session_state["_messages"].append({"role": "user", "content": prompt})
    known = len(st.session_state["_messages"])

    try:
        final_state: list[BaseMessage] = []
        for state_messages in agent.stream_full(prompt, thread_id):
            final_state = state_messages
            for msg in state_messages[known:]:
                render_live(msg, show_trace)
            known = len(state_messages)
    except Exception as exc:  # surface agent failures instead of crashing
        st.error(f"Agent failed: {exc}")
        return

    st.session_state["_messages"] += [
        serialize(msg) for msg in final_state[len(st.session_state["_messages"]) :]
    ]


def render_access_gate() -> bool:
    """Block access until the configured password is entered (optional).

    If ``APP_PASSWORD`` is not set, the app stays open to everyone.
    """
    password = os.environ.get("APP_PASSWORD", "")
    if not password or st.session_state.get("_authed"):
        return True
    st.warning("This app is protected by a password.")
    with st.form("unlock"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Unlock", use_container_width=True)
    if submitted:
        if entered == password:
            st.session_state["_authed"] = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()
    return False


def mask_secret(secret: str, visible: int = 5) -> str:
    """Mask a secret, keeping only the first ``visible`` characters."""
    if len(secret) <= visible:
        return "".join("x" for _ in secret)
    return f"{secret[:visible]}" + "x" * (len(secret) - visible)


def main() -> None:
    st.set_page_config(
        page_title="MongoDB Text-to-Query Agent",
        page_icon="🤖",
        layout="wide",
    )
    st.title("🤖 MongoDB Text-to-Query Agent")
    render_access_gate()

    env_uri = os.environ.get("MONGODB_URI", "")

    with st.sidebar:
        st.header("Connection")
        mongodb_uri = st.text_input(
            "MONGODB_URI",
            value=mask_secret(env_uri) if env_uri else "",
            type="password",
            help="mongodb+srv://... or mongodb://localhost:27017",
        )
        if env_uri and mongodb_uri == mask_secret(env_uri):
            mongodb_uri = env_uri
        database = st.text_input("Database", value="sample_mflix")
        model = st.selectbox("Model", DEFAULT_MODELS, index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        top_k = st.slider("Top K rows", 1, 20, 5)

        show_trace = st.toggle(
            "Show tool traces",
            value=False,
            help="Display the tool calls and raw results the agent executes.",
        )

        if st.button("Connect / Reconnect", use_container_width=True):
            st.session_state["_connected"] = True
            st.rerun()

        if st.button("New conversation", use_container_width=True):
            st.session_state["_thread_id"] = str(uuid.uuid4())
            st.session_state["_messages"] = []
            st.rerun()

        if st.session_state.get("_connected"):
            st.subheader("Try these")
            for sample in SAMPLE_PROMPTS:
                if st.button(sample, use_container_width=True):
                    st.session_state["_pending_prompt"] = sample
                    st.rerun()

            st.divider()
            st.caption(
                f"Thread: `{st.session_state.get('_thread_id', '')[:8]}...`"
            )

    if not st.session_state.get("_connected"):
        st.info("Enter your MongoDB connection string and click **Connect**.")
        return

    if not mongodb_uri:
        st.error("MONGODB_URI is required.")
        return

    try:
        agent = get_agent(mongodb_uri, database, model, temperature, top_k)
    except Exception as exc:
        st.error(f"Unable to initialise the agent: {exc}")
        return

    thread_id = st.session_state.setdefault("_thread_id", str(uuid.uuid4()))
    run_chat(agent, thread_id, show_trace)


if __name__ == "__main__":
    main()