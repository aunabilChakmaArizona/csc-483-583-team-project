"""Simple utilities for tokenizing cleaned wiki article bodies."""

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DB_PATH = PROJECT_ROOT / "data/processed/wiki_articles_step1_clean.sqlite3"
DEFAULT_OUTPUT_DB_PATH = PROJECT_ROOT / "data/processed/wiki_articles_step2_tokens.sqlite3"
DISALLOWED_CHARS_RE = re.compile(r"[^\w-]+")
TOKEN_SPLIT_RE = re.compile(r"[_-]+")


@dataclass(frozen=True)
class WikiArticle:
    title: str
    body: str
    source_file: str
    article_index: int
    is_redirect: int


def load_stop_words() -> set[str]:
    try:
        from spacy.lang.en.stop_words import STOP_WORDS
    except ModuleNotFoundError as error:
        raise RuntimeError("spaCy is required for processor3_tokenize stop-word removal.") from error

    return set(STOP_WORDS)


def lowercase_line(line: str) -> str:
    return line.lower()


def split_line(line: str) -> list[str]:
    return line.split()


def remove_empty_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token]


def strip_boundary_punctuation(token: str) -> str:
    stripped = token

    while stripped and not re.match(r"[\w-]", stripped[0]):
        stripped = stripped[1:]

    while stripped and not re.match(r"[\w-]", stripped[-1]):
        stripped = stripped[:-1]

    return stripped


def remove_stop_words(tokens: list[str], stop_words: set[str]) -> list[str]:
    return [token for token in tokens if token not in stop_words]


def remove_disallowed_characters(token: str) -> str:
    return DISALLOWED_CHARS_RE.sub("", token)


def split_on_underscore_and_hyphen(token: str) -> list[str]:
    return [part for part in TOKEN_SPLIT_RE.split(token) if part]


def tokenize_line(line: str, stop_words: set[str]) -> list[str]:
    line = lowercase_line(line)
    tokens = split_line(line)
    tokens = remove_empty_tokens(tokens)
    tokens = [strip_boundary_punctuation(token) for token in tokens]
    tokens = remove_empty_tokens(tokens)
    tokens = remove_stop_words(tokens, stop_words)
    tokens = [remove_disallowed_characters(token) for token in tokens]
    tokens = remove_empty_tokens(tokens)

    final_tokens = []
    for token in tokens:
        final_tokens.extend(split_on_underscore_and_hyphen(token))

    return remove_empty_tokens(final_tokens)


def tokenize_body(text: str, stop_words: set[str]) -> list[str]:
    tokens = []

    for line in text.splitlines():
        tokens.extend(tokenize_line(line, stop_words))

    return tokens


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


def processed_article_row(article: WikiArticle, stop_words: set[str]) -> tuple[str, str, str, int, int]:
    tokens = tokenize_body(article.body, stop_words)
    return (
        article.title,
        " ".join(tokens),
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


def materialize_tokenized_articles(
    input_db_path: Path = DEFAULT_INPUT_DB_PATH,
    output_db_path: Path = DEFAULT_OUTPUT_DB_PATH,
    batch_size: int = 1000,
) -> int:
    initialize_output_database(output_db_path)
    stop_words = load_stop_words()
    total = 0
    batch = []
    start_time = time.time()

    with sqlite3.connect(output_db_path) as output_connection:
        for article in iter_articles_from_db(input_db_path):
            batch.append(processed_article_row(article, stop_words))
            total += 1

            if total % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"[processor3_tokenize] Articles: {total} | Elapsed: {elapsed:.2f}s")

            if len(batch) >= batch_size:
                write_batch(output_connection, batch)
                batch = []

        write_batch(output_connection, batch)
        output_connection.commit()

    elapsed = time.time() - start_time
    print(f"[processor3_tokenize] Finished | Articles: {total} | Elapsed: {elapsed:.2f}s")
    return total


if __name__ == "__main__":
    total = materialize_tokenized_articles()
    print(f"Stored {total} articles in {DEFAULT_OUTPUT_DB_PATH}")
