"""
cache_manager.py - Caching Configuration
==========================================
Implements BOTH caching strategies required by the assignment:

  1. InMemoryCache  - stored in RAM; fastest; does NOT survive app restart.
  2. SQLiteCache    - stored on disk (.db file); slightly slower; survives restart.

Use `setup_cache()` to activate the selected cache type.
Use `set_llm_cache(...)` from LangChain to register the cache globally.
Once set, LangChain automatically checks the cache before every LLM call.

Submitting the same form twice with caching enabled should reuse the
cached result and be visibly faster the second time.

IMPORTANT: This is an educational AI prototype, NOT a medical device.
"""

from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from src.config import SQLITE_CACHE_DB


def setup_cache(cache_type: str) -> str:
    """
    Configure the LangChain LLM cache.

    Args:
        cache_type: One of "None", "InMemoryCache", or "SQLiteCache".

    Returns:
        A human-readable status message describing what was configured.

    How it works:
        - "None"         → Disables caching; every call hits the API.
        - "InMemoryCache" → Stores results in RAM. Fast, but lost on restart.
        - "SQLiteCache"   → Stores results in a .db file on disk.
                            Slightly slower, but persists across restarts.

    LangChain's `set_llm_cache()` registers the cache globally.
    After registration, LangChain checks the cache AUTOMATICALLY
    before every model call — no extra code needed in chains.py.
    """
    if cache_type == "InMemoryCache":
        # ---------------------------------------------------------------
        # InMemoryCache: stored in RAM (Python dictionary internally).
        # Pros: Fastest possible lookup.
        # Cons: All cached data is lost when the app restarts.
        # Best for: Single-session use during development or demos.
        # ---------------------------------------------------------------
        set_llm_cache(InMemoryCache())
        return (
            "✅ **InMemoryCache** enabled.\n\n"
            "Results are cached in RAM. Identical requests will be "
            "served instantly. Cache is lost when the app restarts."
        )

    elif cache_type == "SQLiteCache":
        # ---------------------------------------------------------------
        # SQLiteCache: stored in a local SQLite database file on disk.
        # Pros: Survives app restarts; good for repeated sessions.
        # Cons: Slightly slower than in-memory (disk I/O).
        # Best for: Reusing cached results across multiple sessions.
        # ---------------------------------------------------------------
        set_llm_cache(SQLiteCache(database_path=SQLITE_CACHE_DB))
        return (
            "✅ **SQLiteCache** enabled.\n\n"
            f"Results are cached in `{SQLITE_CACHE_DB}` on disk. "
            "Cache survives app restarts. Identical requests will "
            "reuse the stored result."
        )

    else:
        # ---------------------------------------------------------------
        # No cache: every request hits the OpenAI API directly.
        # ---------------------------------------------------------------
        set_llm_cache(None)  # type: ignore[arg-type]
        return (
            "ℹ️ **Caching disabled.**\n\n"
            "Every request will call the OpenAI API directly."
        )
