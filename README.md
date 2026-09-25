# MongoDB Text-to-Query Agent (LangGraph + Streamlit)

[![CI](https://github.com/HassanSayedTesla/text2query-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/HassanSayedTesla/text2query-agent/actions/workflows/ci.yml)

**Live demo:** <https://text2queryagent.streamlit.app>

Turn natural-language questions into MongoDB queries using a LangGraph-powered
agent — with multi-turn memory persisted in MongoDB.

Built as a production-style project from the hands-on exercises of the
*Text-to-Query Agents with MongoDB and LangGraph* course. The original course
notebooks are kept in [`tutorials/`](tutorials/) for reference.

## Demo

![Demo](media/demo.gif)

*Screen recording of the deployed app answering sample questions. Replace
`media/demo.gif` with your own 30-second capture.*

Try these from the sidebar once connected:

- "Which states have the most theaters?"
- "How many movies did Arthur C. Clarke write, and which are the highest rated?"
- "List the 5 newest movies in the database."
- "What is the average runtime of movies starring Tom Hanks?"

## What it does

Ask questions like *"Which states have the most theaters?"* or *"What are the
top 5 movies by IMDB rating?"* and the agent:

1. Lists the collections in the database.
2. Inspects the schema of the relevant collection.
3. Generates the MongoDB aggregation/find query.
4. Validates the query with a checker tool.
5. Executes the query.
6. Reasons over the results and answers in natural language.

Because conversations are checkpointed with `MongoDBSaver`, follow-up
questions like *"How many theaters does California have?"* are answered from
short-term memory without re-running tools. Each browser session keeps its own
conversation thread in MongoDB, so visitors never share memory; the **New
conversation** button rotates the thread ID to start fresh.

## Architecture

```
Streamlit UI (app/main.py)
        │
        ▼
TextToQueryAgent (app/agent/agent.py)
   │            │
   │            └── MongoDBSaver ──► MongoDB (memory + data)
   ▼
LangGraph state machine (app/agent/graph.py)
   │  agent ──► tools_condition ──► tools
   │                                  │
   ▼                                  ▼
ChatGroq (openai/gpt-oss-120b)   MongoDBDatabaseToolkit tools
                                (mongodb_query, mongodb_schema, ...)
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details.

## Quick start

### 1. Prerequisites

- Python 3.9+
- A MongoDB instance. Either:
  - a free [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) cluster with
    the `sample_mflix` sample dataset loaded, or
  - Docker (for the fully local path below).
- A [Groq API key](https://console.groq.com/keys) (free tier available).

### 2. Configure

```bash
cp .env.example .env
# fill in MONGODB_URI and GROQ_API_KEY
```

### 3. Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt

streamlit run app/main.py
```

Open <http://localhost:8501>, optionally paste your connection string into the
sidebar, and hit **Connect**.

### 4. Run with Docker (fully local, no Atlas required)

```bash
docker compose up --build
```

This starts a local MongoDB, seeds it with a small `sample_mflix` demo dataset
(`theaters`, `movies`), and launches the UI at <http://localhost:8501>.

> Set `GROQ_API_KEY` in your `.env` — `docker compose` reads it from there.

### 5. Run tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check app tests scripts
```

The tests are fully mocked — no MongoDB or Groq credentials are needed.

### 6. Deploy to Streamlit Community Cloud

1. Push the repo to GitHub, then deploy from
   <https://share.streamlit.io> with **Main file path** `app/main.py`
   (forward slashes).
2. In **Manage app → Settings → Secrets**, add the same values as
   environment variables, in TOML format:

   ```toml
   MONGODB_URI = "mongodb+srv://..."
   GROQ_API_KEY = "gsk_..."
   GROQ_MODEL = "openai/gpt-oss-120b"
   APP_PASSWORD = "anything-secret"
   ```

3. Click **Save** — the app restarts with the secrets loaded.

> If `APP_PASSWORD` is set, visitors must enter it before using the app, so
> only you pay for Groq queries.

## Project layout

```
app/
  main.py              Streamlit chat UI
  settings.py          Environment-variable configuration
  agent/
    agent.py           TextToQueryAgent facade
    graph.py           LangGraph build + nodes
    prompt.py          System prompt + memory instruction
    state.py           GraphState definition
  db/
    connection.py      Mongo client, toolkit, checkpointer helpers
scripts/
  seed_local_mongo.py  Populate local MongoDB with demo data
tests/                 Mocked pytest suite + fakes
tutorials/             Original course notebooks (reference)
docs/ARCHITECTURE.md   Deep dive into design decisions
```

## Configuration reference

| Variable          | Default         | Description                            |
| ----------------- | --------------- | -------------------------------------- |
| `MONGODB_URI`     | — (required)    | MongoDB connection string              |
| `DATABASE_NAME`   | `sample_mflix`  | Database the agent queries             |
| `GROQ_API_KEY`    | — (required)    | Groq key for `ChatGroq`                |
| `GROQ_MODEL`      | `openai/gpt-oss-120b` | Model used by the agent   |
| `LLM_TEMPERATURE` | `0`             | Sampling temperature                   |
| `TOP_K`           | `5`             | Rows the agent reasons over            |
| `APP_PASSWORD`    | *(unset)*       | Optional password gate for the app     |

## Roadmap ideas

- Semantic cache / vector memory for frequently asked questions.
- Evaluation harness replaying real agent trajectories from checkpoints.
- Read-only enforcement on `mongodb_query` (separate DB user) for safety.
- Streaming token-by-token answers via `stream_mode="messages"`.

## License

MIT