"""MongoDB connection helpers used by the agent and the Streamlit app."""

from __future__ import annotations

import os

from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient


def get_mongo_client(mongodb_uri: str | None = None) -> MongoClient:
    """Create a PyMongo client from an explicit URI or the environment."""
    uri = mongodb_uri or os.environ.get("MONGODB_URI")
    if not uri:
        raise ValueError(
            "MONGODB_URI is not set. Provide a connection string or set "
            "the MONGODB_URI environment variable."
        )
    return MongoClient(uri)


def get_database(
    mongodb_uri: str | None = None, database: str = "sample_mflix"
) -> MongoDBDatabase:
    """Create a MongoDBDatabase wrapper used by the agent toolkit."""
    uri = mongodb_uri or os.environ.get("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI is not set.")
    return MongoDBDatabase.from_connection_string(
        connection_string=uri, database=database
    )


def get_toolkit(
    mongodb_uri: str | None,
    database: str | None,
    llm,
) -> MongoDBDatabaseToolkit:
    """Create a MongoDBDatabaseToolkit bound to an LLM."""
    db = get_database(mongodb_uri, database or "sample_mflix")
    return MongoDBDatabaseToolkit(db=db, llm=llm)


def get_checkpointer(mongodb_uri: str | None = None) -> MongoDBSaver:
    """Create a MongoDBSaver used to persist LangGraph short-term memory."""
    client = get_mongo_client(mongodb_uri)
    return MongoDBSaver(client)