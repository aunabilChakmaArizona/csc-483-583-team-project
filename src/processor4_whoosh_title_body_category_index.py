"""Build a Whoosh index for cleaned title/body text plus semantic top-category matches."""

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time

from whoosh import index as whoosh_index

try:
    from src.processor4_whoosh_title_body_index import DEFAULT_INPUT_DB_PATH
    from src.processor_category_semantic import DEFAULT_OUTPUT_DB_PATH as DEFAULT_SEMANTIC_DB_PATH
    from src.schema import get_whoosh_title_body_category_schema
except ModuleNotFoundError:
    from processor4_whoosh_title_body_index import DEFAULT_INPUT_DB_PATH
    from processor_category_semantic import DEFAULT_OUTPUT_DB_PATH as DEFAULT_SEMANTIC_DB_PATH
    from schema import get_whoosh_title_body_category_schema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index/whoosh_title_body_category_index"


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
    return whoosh_index.create_in(index_dir, get_whoosh_title_body_category_schema())


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


def load_semantic_category_lookup(
    semantic_db_path: Path = DEFAULT_SEMANTIC_DB_PATH,
) -> dict[tuple[str, int], dict[str, list[str]]]:
    if not semantic_db_path.exists():
        raise FileNotFoundError(
            f"Semantic category DB not found at {semantic_db_path}. "
            "Run src.processor_category_semantic first."
        )

    with sqlite3.connect(semantic_db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                source_file,
                article_index,
                title_first_sentence_categories_json,
                title_first_two_sentences_categories_json
            FROM article_category_matches
            """
        ).fetchall()

    lookup = {}
    for (
        source_file,
        article_index,
        title_first_sentence_categories_json,
        title_first_two_sentences_categories_json,
    ) in rows:
        lookup[(source_file, article_index)] = {
            "title_first_sentence_categories": json.loads(title_first_sentence_categories_json),
            "title_first_two_sentences_categories": json.loads(
                title_first_two_sentences_categories_json
            ),
        }

    return lookup


def serialize_categories_for_keyword(categories: list[str]) -> str:
    return ",".join(category.strip().lower() for category in categories if category.strip())


def article_document(
    article: WikiArticle,
    semantic_lookup: dict[tuple[str, int], dict[str, list[str]]],
) -> dict:
    category_data = semantic_lookup.get(
        (article.source_file, article.article_index),
        {
            "title_first_sentence_categories": [],
            "title_first_two_sentences_categories": [],
        },
    )
    return {
        "title": article.title,
        "body": article.body,
        "title_first_sentence_categories": serialize_categories_for_keyword(
            category_data["title_first_sentence_categories"]
        ),
        "title_first_two_sentences_categories": serialize_categories_for_keyword(
            category_data["title_first_two_sentences_categories"]
        ),
        "source_file": article.source_file,
        "article_index": article.article_index,
        "is_redirect": article.is_redirect,
    }


def write_batch(writer, batch: list[dict]) -> None:
    if not batch:
        return

    for document in batch:
        writer.add_document(**document)


def materialize_whoosh_title_body_category_index(
    input_db_path: Path = DEFAULT_INPUT_DB_PATH,
    semantic_db_path: Path = DEFAULT_SEMANTIC_DB_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    batch_size: int = 1000,
) -> int:
    index = create_index(index_dir)
    semantic_lookup = load_semantic_category_lookup(semantic_db_path)
    total = 0
    batch = []
    matched_with_semantics = 0
    start_time = time.time()

    writer = index.writer()
    for article in iter_articles_from_db(input_db_path):
        if (article.source_file, article.article_index) in semantic_lookup:
            matched_with_semantics += 1

        batch.append(article_document(article, semantic_lookup))
        total += 1

        if total % 1000 == 0:
            elapsed = time.time() - start_time
            print(
                f"[processor4_whoosh_title_body_category_index] Articles: {total} | "
                f"Semantic matches: {matched_with_semantics} | Elapsed: {elapsed:.2f}s"
            )

        if len(batch) >= batch_size:
            write_batch(writer, batch)
            batch = []

    write_batch(writer, batch)
    writer.commit()

    elapsed = time.time() - start_time
    print(
        f"[processor4_whoosh_title_body_category_index] Finished | "
        f"Articles: {total} | Semantic matches: {matched_with_semantics} | "
        f"Elapsed: {elapsed:.2f}s"
    )
    return total


if __name__ == "__main__":
    total = materialize_whoosh_title_body_category_index()
    print(f"Indexed {total} articles in {DEFAULT_INDEX_DIR}")
