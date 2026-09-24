"""Runtime configuration for the text-to-query agent.

Reading environment variables:
- MONGODB_URI      MongoDB connection string (required).
- DATABASE_NAME    Database the agent queries against.
- GROQ_API_KEY     API key for the Groq chat model.
- GROQ_MODEL       Model identifier used by the LLM.
- LLM_TEMPERATURE  Sampling temperature for the model.
- TOP_K            Default number of rows the agent reasons over.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get_env(name: str, default: str | None = None) -> str | None:
    """Return an environment variable, or a fallback default."""
    return os.getenv(name, default)


def get_required(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Missing required environment variable: {name}. "
            "Set it in your environment or in a .env file."
        )
    return value


class Settings:
    """Immutable-ish view over the app configuration."""

    def __init__(self) -> None:
        self.mongodb_uri = get_required("MONGODB_URI")
        self.database_name = get_env("DATABASE_NAME", "sample_mflix")
        self.groq_model = get_env("GROQ_MODEL", "openai/gpt-oss-120b")
        self.temperature = float(get_env("LLM_TEMPERATURE", "0"))
        self.top_k = int(get_env("TOP_K", "5"))