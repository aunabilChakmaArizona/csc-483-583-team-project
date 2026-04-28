"""Score exported cross-encoder pairs and write a compatible SQLite cache.

Typical Colab usage:

    !pip install -q sentence-transformers
    !python colab_score_cross_encoder_pairs.py \
        --input cross_encoder_pairs.jsonl \
        --output cross_encoder_pair_cache.sqlite3 \
        --model-name cross-encoder/ms-marco-MiniLM-L-12-v2 \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import time
import zlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score cross-encoder pairs and write component_cache rows."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("cross_encoder_pair_cache.sqlite3"))
    parser.add_argument("--model-name", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--commit-every",
        type=int,
        default=0,
        help=(
            "Commit every N newly scored rows. Default 0 commits once at the end. "
            "Use a positive value for checkpointing."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def initialize_output_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA busy_timeout = 30000")
    with connection:
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
    return connection


def iter_pair_rows(path: Path, limit: int = 0):
    with path.open("r", encoding="utf-8") as input_file:
        for row_number, line in enumerate(input_file, start=1):
            if limit and row_number > limit:
                break
            if not line.strip():
                continue
            yield json.loads(line)


def cached_keys(connection: sqlite3.Connection, cache_keys: list[str]) -> set[str]:
    if not cache_keys:
        return set()

    placeholders = ",".join("?" for _ in cache_keys)
    rows = connection.execute(
        f"SELECT cache_key FROM component_cache WHERE cache_key IN ({placeholders})",
        cache_keys,
    ).fetchall()
    return {row[0] for row in rows}


def score_rows(model, rows: list[dict], batch_size: int) -> list[float]:
    pairs = [(row["text_a"], row["text_b"]) for row in rows]
    return [float(score) for score in model.predict(pairs, batch_size=batch_size)]


def score_payload(score: float) -> bytes:
    return zlib.compress(
        json.dumps(
            [{"score": score}],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def build_score_records(rows: list[dict], scores: list[float]) -> list[tuple]:
    now = time.time()
    records = []
    for row, score in zip(rows, scores):
        records.append((row["cache_key"], "cross_encoder_pair_score", now, score_payload(score)))
    return records


def insert_score_records(connection: sqlite3.Connection, records: list[tuple]) -> None:
    if not records:
        return

    connection.executemany(
        """
        INSERT OR REPLACE INTO component_cache (cache_key, component_name, created_at, payload)
        VALUES (?, ?, ?, ?)
        """,
        records,
    )


def main() -> None:
    args = parse_args()

    from sentence_transformers import CrossEncoder

    connection = initialize_output_database(args.output)
    model = CrossEncoder(args.model_name)
    pending_rows = []
    pending_records = []
    processed = 0
    skipped = 0
    rows_since_commit = 0

    def flush_pending() -> None:
        nonlocal processed, skipped, pending_rows, pending_records, rows_since_commit
        if not pending_rows:
            return

        existing = cached_keys(connection, [row["cache_key"] for row in pending_rows])
        rows_to_score = [row for row in pending_rows if row["cache_key"] not in existing]
        skipped += len(pending_rows) - len(rows_to_score)
        if rows_to_score:
            scores = score_rows(model, rows_to_score, args.batch_size)
            pending_records.extend(build_score_records(rows_to_score, scores))
            processed += len(rows_to_score)
            rows_since_commit += len(rows_to_score)
            if args.commit_every > 0 and rows_since_commit >= args.commit_every:
                insert_score_records(connection, pending_records)
                connection.commit()
                pending_records = []
                rows_since_commit = 0
        print(
            f"[colab-score-cross-encoder] processed: {processed} | skipped: {skipped}",
            flush=True,
        )
        pending_rows = []

    for row in iter_pair_rows(args.input, limit=args.limit):
        if row.get("model_name") != args.model_name:
            raise ValueError(
                f"Input row model_name {row.get('model_name')!r} does not match "
                f"--model-name {args.model_name!r}."
            )
        pending_rows.append(row)
        if len(pending_rows) >= args.batch_size:
            flush_pending()

    flush_pending()
    insert_score_records(connection, pending_records)
    connection.commit()
    connection.close()
    print(f"Wrote cache to {args.output}")


if __name__ == "__main__":
    main()
