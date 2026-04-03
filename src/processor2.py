"""Simple utilities for cleaning stored wiki article bodies before normalization."""

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DB_PATH = PROJECT_ROOT / "data/processed/wiki_articles.sqlite3"
DEFAULT_OUTPUT_DB_PATH = PROJECT_ROOT / "data/processed/wiki_articles_step1_clean.sqlite3"
SECTION_HEADER_RE = re.compile(r"^\s*=+\s*.*?\s*=+\s*$", re.MULTILINE)
REDIRECT_LINE_RE = re.compile(r"^\s*#REDIRECT\b.*$", re.MULTILINE)
CATEGORIES_LINE_RE = re.compile(r"^\s*CATEGORIES:.*$", re.MULTILINE)
TEMPLATE_RE = re.compile(r"\[tpl\].*?\[/tpl\]", re.DOTALL)
REF_RE = re.compile(r"\[ref\].*?(?:\[/ref\]|(?=\n|$))", re.DOTALL)
FILE_RE = re.compile(r"\[\[File:.*?\]\]", re.DOTALL)


@dataclass(frozen=True)
class WikiArticle:
    title: str
    body: str
    source_file: str
    article_index: int
    is_redirect: int


def remove_redirect_lines(text: str) -> str:
    return REDIRECT_LINE_RE.sub("", text)


def remove_section_headers(text: str) -> str:
    return SECTION_HEADER_RE.sub("", text)


def remove_templates(text: str) -> str:
    return TEMPLATE_RE.sub("", text)


def remove_references(text: str) -> str:
    return REF_RE.sub("", text)


def remove_file_markup(text: str) -> str:
    return FILE_RE.sub("", text)


def remove_categories_lines(text: str) -> str:
    return CATEGORIES_LINE_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def clean_body(text: str) -> str:
    cleaned = text
    cleaned = remove_redirect_lines(cleaned)
    cleaned = remove_section_headers(cleaned)
    cleaned = remove_templates(cleaned)
    cleaned = remove_references(cleaned)
    cleaned = remove_file_markup(cleaned)
    cleaned = remove_categories_lines(cleaned)
    return normalize_whitespace(cleaned)


def initialize_output_database(db_path: Path = DEFAULT_OUTPUT_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                source_file TEXT NOT NULL,
                article_index INTEGER NOT NULL,
                is_redirect INTEGER NOT NULL,
                UNIQUE(source_file, article_index)
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_articles_title ON articles(title)")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_file, article_index)"
        )
        connection.commit()


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


def processed_article_row(article: WikiArticle) -> tuple[str, str, str, int, int]:
    return (
        article.title,
        clean_body(article.body),
        article.source_file,
        article.article_index,
        article.is_redirect,
    )


def write_batch(connection: sqlite3.Connection, batch: list[tuple[str, str, str, int, int]]) -> None:
    if not batch:
        return

    connection.executemany(
        """
        INSERT OR REPLACE INTO articles
        (title, body, source_file, article_index, is_redirect)
        VALUES (?, ?, ?, ?, ?)
        """,
        batch,
    )


def materialize_cleaned_articles(
    input_db_path: Path = DEFAULT_INPUT_DB_PATH,
    output_db_path: Path = DEFAULT_OUTPUT_DB_PATH,
    batch_size: int = 1000,
) -> int:
    initialize_output_database(output_db_path)
    total = 0
    batch = []
    start_time = time.time()

    with sqlite3.connect(output_db_path) as output_connection:
        for article in iter_articles_from_db(input_db_path):
            batch.append(processed_article_row(article))
            total += 1

            if total % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"[processor2] Articles: {total} | Elapsed: {elapsed:.2f}s")

            if len(batch) >= batch_size:
                write_batch(output_connection, batch)
                batch = []

        write_batch(output_connection, batch)
        output_connection.commit()

    elapsed = time.time() - start_time
    print(f"[processor2] Finished | Articles: {total} | Elapsed: {elapsed:.2f}s")
    return total


if __name__ == "__main__":
    total = materialize_cleaned_articles()
    print(f"Stored {total} articles in {DEFAULT_OUTPUT_DB_PATH}")
