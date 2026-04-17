"""Build a Cole-style Whoosh index that contains only redirect articles."""

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time

from whoosh import index as whoosh_index
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import STORED, TEXT, Schema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DB_PATH = PROJECT_ROOT / "data/processed/wiki_articles_step1_clean.sqlite3"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index/whoosh_cole_redirect_index"


@dataclass(frozen=True)
class WikiArticle:
    title: str
    body: str
    source_file: str
    article_index: int
    is_redirect: int


def get_schema() -> Schema:
    analyzer = StandardAnalyzer()
    return Schema(
        title=TEXT(stored=True, analyzer=analyzer),
        body=TEXT(stored=False, analyzer=analyzer),
        source_file=STORED(),
        article_index=STORED(),
        is_redirect=STORED(),
    )


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


def materialize_whoosh_cole_redirect_index(
    input_db_path: Path = DEFAULT_INPUT_DB_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    batch_size: int = 1000,
) -> int:
    index = create_index(index_dir)
    total = 0
    skipped_non_redirects = 0
    start_time = time.time()

    writer = index.writer()
    for article in iter_articles_from_db(input_db_path):
        if not article.is_redirect:
            skipped_non_redirects += 1
            continue

        writer.add_document(
            title=article.title,
            body=article.body,
            source_file=article.source_file,
            article_index=article.article_index,
            is_redirect=article.is_redirect,
        )
        total += 1

        if total % batch_size == 0:
            elapsed = time.time() - start_time
            print(f"[processor4_cole_redirect_index] Indexed: {total} | Elapsed: {elapsed:.2f}s")

    print("[processor4_cole_redirect_index] Committing index to disk (this may take a moment)...")
    writer.commit()

    elapsed = time.time() - start_time
    print(
        f"[processor4_cole_redirect_index] Done | Indexed: {total} | "
        f"Skipped non-redirects: {skipped_non_redirects} | Elapsed: {elapsed:.2f}s"
    )
    return total


if __name__ == "__main__":
    total = materialize_whoosh_cole_redirect_index()
    print(f"Indexed {total} redirect articles in {DEFAULT_INDEX_DIR}")
