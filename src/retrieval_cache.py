"""SQLite-backed cache for retrieval component results."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import zlib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DB_PATH = PROJECT_ROOT / "data/processed/retrieval_component_cache.sqlite3"
CACHE_SCHEMA_VERSION = "retrieval_component_cache_v1"
RETRIEVAL_LOGIC_VERSION = "retrieval_logic_v1"
_CACHE_STATS = {"hits": 0, "misses": 0}


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=512)
def fingerprint_path(path_value: str) -> dict:
    path = Path(path_value)
    resolved_path = path.resolve()

    if not path.exists():
        return {
            "path": str(resolved_path),
            "exists": False,
        }

    if path.is_file():
        stats = path.stat()
        return {
            "path": str(resolved_path),
            "exists": True,
            "kind": "file",
            "size": stats.st_size,
            "mtime_ns": stats.st_mtime_ns,
        }

    entries = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        stats = child.stat()
        entries.append(
            {
                "relative_path": str(child.relative_to(path)),
                "size": stats.st_size,
                "mtime_ns": stats.st_mtime_ns,
            }
        )

    return {
        "path": str(resolved_path),
        "exists": True,
        "kind": "directory",
        "entries": entries,
    }


def fingerprint_paths(paths: list[Path] | tuple[Path, ...]) -> str:
    return sha256_text(canonical_json([fingerprint_path(str(path)) for path in paths]))


def digest_query_embedding(query_embedding) -> str | None:
    if query_embedding is None:
        return None

    return sha256_bytes(query_embedding.tobytes())


def build_component_cache_key(
    *,
    component_name: str,
    query_text: str,
    query_category: str | None,
    component_limit: int,
    index_paths: list[Path] | tuple[Path, ...],
    extra_config: dict | None = None,
    query_embedding_digest: str | None = None,
) -> str:
    key_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "retrieval_logic_version": RETRIEVAL_LOGIC_VERSION,
        "component_name": component_name,
        "query_text": query_text,
        "query_category": query_category,
        "component_limit": component_limit,
        "index_fingerprint": fingerprint_paths(index_paths),
        "query_embedding_digest": query_embedding_digest,
        "extra_config": extra_config or {},
    }
    return sha256_text(canonical_json(key_payload))


def initialize_cache_database(cache_db_path: Path = DEFAULT_CACHE_DB_PATH) -> None:
    cache_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache_db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS component_cache (
                cache_key TEXT PRIMARY KEY,
                component_name TEXT NOT NULL,
                created_at REAL NOT NULL,
                payload BLOB NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_component_cache_name ON component_cache(component_name)"
        )
        connection.commit()


def reset_cache_stats() -> None:
    _CACHE_STATS["hits"] = 0
    _CACHE_STATS["misses"] = 0


def get_cache_stats() -> dict[str, int]:
    return {
        "hits": _CACHE_STATS["hits"],
        "misses": _CACHE_STATS["misses"],
    }


def load_cached_component_results(
    cache_key: str,
    cache_db_path: Path = DEFAULT_CACHE_DB_PATH,
) -> list[dict] | None:
    initialize_cache_database(cache_db_path)
    with sqlite3.connect(cache_db_path) as connection:
        row = connection.execute(
            "SELECT payload FROM component_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()

    if row is None:
        return None

    return json.loads(zlib.decompress(row[0]).decode("utf-8"))


def store_component_results(
    *,
    cache_key: str,
    component_name: str,
    results: list[dict],
    cache_db_path: Path = DEFAULT_CACHE_DB_PATH,
) -> None:
    initialize_cache_database(cache_db_path)
    payload = zlib.compress(
        json.dumps(results, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    with sqlite3.connect(cache_db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO component_cache (cache_key, component_name, created_at, payload)
            VALUES (?, ?, ?, ?)
            """,
            (cache_key, component_name, time.time(), payload),
        )
        connection.commit()


def get_or_compute_component_results(
    *,
    component_name: str,
    query_text: str,
    query_category: str | None,
    component_limit: int,
    index_paths: list[Path] | tuple[Path, ...],
    extra_config: dict | None,
    compute_results,
    query_embedding_digest: str | None = None,
    cache_db_path: Path = DEFAULT_CACHE_DB_PATH,
) -> list[dict]:
    cache_key = build_component_cache_key(
        component_name=component_name,
        query_text=query_text,
        query_category=query_category,
        component_limit=component_limit,
        index_paths=index_paths,
        extra_config=extra_config,
        query_embedding_digest=query_embedding_digest,
    )

    cached_results = load_cached_component_results(cache_key, cache_db_path=cache_db_path)
    if cached_results is not None:
        _CACHE_STATS["hits"] += 1
        return cached_results

    _CACHE_STATS["misses"] += 1
    results = compute_results()
    store_component_results(
        cache_key=cache_key,
        component_name=component_name,
        results=results,
        cache_db_path=cache_db_path,
    )
    return results
