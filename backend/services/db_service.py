"""MongoDB connection (async, via Motor) and small helpers."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.environ.get("MONGODB_DB", "disease_surveillance")


@lru_cache(maxsize=1)
def _client() -> AsyncIOMotorClient:
    return AsyncIOMotorClient(MONGODB_URI)


def get_db():
    """Return the surveillance database handle (one client per process)."""
    return _client()[MONGODB_DB]


def iso(d) -> str:
    """Format a stored datetime as YYYY-MM-DD."""
    return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
