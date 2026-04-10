"""Simple utilities for building a Whoosh inverted index from tokenized wiki articles."""

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time

from whoosh import index as whoosh_index

try:
    from src.schema import get_schema
except ModuleNotFoundError:
    from schema import get_schema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DB_PATH = PROJECT_ROOT / "data/processed/wiki_articles_step2_tokens.sqlite3"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index/whoosh_index"


@dataclass(frozen=True)
class WikiArticle:
    title: str
    body: str
    source_file: str
    article_index: int
    is_redirect: int


def initialize_index_directory(index_dir: Path = DEFAULT_INDEX_DIR) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)


def create_index(index_dir: Path = DEFAULT_INDEX_DIR):
    initialize_index_directory(index_dir)
    return whoosh_index.create_in(index_dir, get_schema())


def open_index(index_dir: Path = DEFAULT_INDEX_DIR):
    if not index_dir.exists() or not whoosh_index.exists_in(index_dir):
        raise FileNotFoundError(f"Whoosh index not found at {index_dir}. Build it first.")

    return whoosh_index.open_dir(index_dir)


def iter_articles_from_db(db_path: Path = DEFAULT_INPUT_DB_PATH):
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            SELECT title, body, source_file, article_index, is_redirect
            FROM articles
            ORDER BY source_file, article_index
            """
        )
        for row in cursor:
            yield WikiArticle(*row)


def article_document(article: WikiArticle) -> dict:
    return {
        "title": article.title,
        "body": article.body,
        "source_file": article.source_file,
        "article_index": article.article_index,
        "is_redirect": article.is_redirect,
    }


def write_batch(writer, batch: list[dict]) -> None:
    if not batch:
        return

    for document in batch:
        writer.add_document(**document)


def materialize_whoosh_index(
    input_db_path: Path = DEFAULT_INPUT_DB_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    batch_size: int = 1000,
) -> int:
    index = create_index(index_dir)
    total = 0
    batch = []
    start_time = time.time()

    writer = index.writer()
    for article in iter_articles_from_db(input_db_path):
        batch.append(article_document(article))
        total += 1

        if total % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"[processor4] Articles: {total} | Elapsed: {elapsed:.2f}s")

        if len(batch) >= batch_size:
            write_batch(writer, batch)
            batch = []

    write_batch(writer, batch)
    writer.commit()

    elapsed = time.time() - start_time
    print(f"[processor4] Finished | Articles: {total} | Elapsed: {elapsed:.2f}s")
    return total


if __name__ == "__main__":
    total = materialize_whoosh_index()
    print(f"Indexed {total} articles in {DEFAULT_INDEX_DIR}")
