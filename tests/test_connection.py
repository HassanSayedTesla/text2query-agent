"""Tests for MongoDB connection helpers (no live database required)."""

import pytest
from pymongo import MongoClient

from app.db.connection import get_database, get_mongo_client


def _clear_uri(monkeypatch) -> None:
    monkeypatch.delenv("MONGODB_URI", raising=False)


def test_get_mongo_client_raises_without_uri(monkeypatch) -> None:
    _clear_uri(monkeypatch)
    with pytest.raises(ValueError, match="MONGODB_URI"):
        get_mongo_client()


def test_get_mongo_client_accepts_explicit_uri(monkeypatch) -> None:
    _clear_uri(monkeypatch)
    client = get_mongo_client("mongodb://localhost:27017")
    assert isinstance(client, MongoClient)
    client.close()


def test_get_mongo_client_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    client = get_mongo_client()
    assert isinstance(client, MongoClient)
    client.close()


def test_get_database_raises_without_uri(monkeypatch) -> None:
    _clear_uri(monkeypatch)
    with pytest.raises(ValueError, match="MONGODB_URI"):
        get_database()