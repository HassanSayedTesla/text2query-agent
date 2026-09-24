# Architecture

## Overview

The project turns the course's text-to-query workflow into a reusable agent
library plus a web UI. The core idea (identical to the notebooks): a LangGraph
state machine where an LLM alternates between *reasoning* (agent node) and
*calling MongoDB tools* (tools node) until it has enough information to answer.

## Components

### `app/settings.py`

Reads environment variables (`MONGODB_URI`, `DATABASE_NAME`, `GROQ_MODEL`,
`LLM_TEMPERATURE`, `TOP_K`) and auto-loads a `.env` file via `python-dotenv`.
`Settings` is the single source of runtime configuration.

### `app/db/connection.py`

Thin factories around the MongoDB integrations:

- `get_mongo_client` — raw `pymongo` client.
- `get_database` — `MongoDBDatabase` wrapper (from `langchain-mongodb`) used
  by the toolkit.
- `get_toolkit` — `MongoDBDatabaseToolkit` exposing the agent tools.
- `get_checkpointer` — `MongoDBSaver` for LangGraph short-term memory.

### `app/agent/state.py`

The shared graph state. `messages: Annotated[list, add_messages]` uses
LangGraph's reducer so each node's output is *appended* to the conversation
rather than overwritten.

### `app/agent/prompt.py`

Combines two system messages:

1. `MONGODB_AGENT_SYSTEM_PROMPT` — MongoDB's built-in guidance for generating
   safe MongoDB queries.
2. A memory instruction telling the agent to check its checkpoint for existing
   answers before re-running tools.

`top_k` and `tool_names` are pre-filled via `prompt.partial(...)`.

### `app/agent/graph.py`

Builds the LangGraph `StateGraph`:

```
START ──► agent ──┬─► (no tool calls) ──► END
                  └─► tools ──► agent (loop)
```

- `agent_node` invokes `prompt | llm.bind_tools(tools)` and appends the
  result to state.
- `tool_node` routes every requested call through `tool_map[name]`, wraps each
  observation in a `ToolMessage`, and returns them all.
- `tools_condition` (prebuilt) decides whether to finish or continue calling
  tools.

`build_graph` takes `llm` and `tools` explicitly, so tests can inject fakes.

### `app/agent/agent.py`

`TextToQueryAgent` is the facade the UI talks to. It wires the toolkit's
tools, the `MongoDBSaver` checkpointer and the compiled graph, and exposes:

- `stream_full(question, thread_id)` — yields the *full* message state after
  every node, letting the UI render tool calls live.
- `ask(question, thread_id)` — convenience for scripts.
- `list_collections()` — direct call to `mongodb_list_collections`.

### `app/main.py` (Streamlit)

- Sidebar: connection string, database, model, temperature, `top_k`, a
  **Show tool traces** toggle, **Connect**, and **New conversation**.
- The agent instance is cached with `@st.cache_resource`, keyed on the sidebar
  values — reconnecting edits rebuild the agent, chatting reuses it.
- **New conversation** rotates `thread_id`, which is the same mechanism the
  notebook uses: each thread is a context in MongoDB's checkpointer.
- Messages are serialized to plain dicts in `st.session_state` so the UI
  survives reruns; tool traces are hidden unless toggled.

## Memory model (short-term memory)

`MongoDBSaver` checkpoints the full graph state after **every node**:

- Keyed by `thread_id` → each conversation thread is isolated.
- On a follow-up turn, the checkpointer reloads the previous state, so the
  agent sees prior answers and avoids redundant tool calls.
- `checkpointer.get(config)` can be used to inspect/replay trajectories.

Trade-off: this is *short-term* memory. There is no long-term/vector memory in
this version — every new thread starts fresh (see Roadmap in the README).

## Error handling

- Missing `MONGODB_URI` fails fast with a descriptive error.
- Streamlit surfaces agent exceptions as `st.error` instead of crashing.
- The UI disconnects gracefully when initialization fails (bad URI/key).

## Testing strategy

All tests avoid live services:

- `tests/fakes.py` — scripted `FakeChatModel` and `FakeTool`.
- `test_graph.py` — builds real LangGraph apps with fakes; verifies the
  answer-only path, the tool loop, and streamed `ToolMessage`s.
- `test_prompt.py`, `test_agent.py`, `test_connection.py` — pure unit tests.

CI runs the suite against Python 3.9 and 3.11, plus `ruff` linting.

## Deployment

- **Local**: `streamlit run app/main.py`.
- **Docker**: `docker compose up --build` brings up MongoDB (+ seed) and the
  app; the seed script (`scripts/seed_local_mongo.py`) populates a small
  `theaters`/`movies` dataset so the demo works offline.