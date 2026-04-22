"""Compute semantic category matches for wiki articles using a small Sentence-BERT model."""

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

try:
    from src.processor1_parse import DEFAULT_DB_PATH as DEFAULT_INPUT_DB_PATH
    from src.processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from src.processor4_lead_index_common import extract_lead_text
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_DB_PATH as DEFAULT_INPUT_DB_PATH
    from processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from processor4_lead_index_common import extract_lead_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DB_PATH = PROJECT_ROOT / "data/processed/wiki_category_semantic.sqlite3"
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUESTION_CATEGORY_LIMIT = 100
TOP_CATEGORY_MATCHES = 10
ARTICLE_CATEGORY_MATCH_COLUMNS = [
    "id",
    "source_file",
    "article_index",
    "title",
    "title_first_sentence_categories_json",
    "title_first_sentence_scores_json",
    "title_first_two_sentences_categories_json",
    "title_first_two_sentences_scores_json",
]


@dataclass(frozen=True)
class WikiArticle:
    title: str
    body: str
    source_file: str
    article_index: int
    is_redirect: int


def load_unique_categories(
    questions_path: Path = DEFAULT_QUESTIONS_JSON_PATH,
    limit: int = QUESTION_CATEGORY_LIMIT,
) -> list[str]:
    rows = json.loads(questions_path.read_text(encoding="utf-8"))[:limit]
    seen = set()
    categories = []

    for row in rows:
        category = row["category"].strip()
        if not category or category in seen:
            continue
        seen.add(category)
        categories.append(category)

    return categories


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


def create_article_category_matches_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE article_category_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            article_index INTEGER NOT NULL,
            title TEXT NOT NULL,
            title_first_sentence_categories_json TEXT NOT NULL,
            title_first_sentence_scores_json TEXT NOT NULL,
            title_first_two_sentences_categories_json TEXT NOT NULL,
            title_first_two_sentences_scores_json TEXT NOT NULL,
            UNIQUE(source_file, article_index)
        )
        """
    )
    connection.execute(
        "CREATE INDEX idx_article_category_matches_title ON article_category_matches(title)"
    )
    connection.execute(
        """
        CREATE INDEX idx_article_category_matches_source
        ON article_category_matches(source_file, article_index)
        """
    )


def article_category_match_columns(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute("PRAGMA table_info(article_category_matches)").fetchall()
    return [row[1] for row in rows]


def initialize_output_database(db_path: Path = DEFAULT_OUTPUT_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        existing_columns = article_category_match_columns(connection)
        if existing_columns and existing_columns != ARTICLE_CATEGORY_MATCH_COLUMNS:
            connection.execute("DROP TABLE article_category_matches")
            create_article_category_matches_table(connection)
        elif not existing_columns:
            create_article_category_matches_table(connection)

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.commit()


def write_metadata(connection: sqlite3.Connection, categories: list[str]) -> None:
    rows = [
        ("model_name", SEMANTIC_MODEL_NAME),
        ("question_category_limit", str(QUESTION_CATEGORY_LIMIT)),
        ("category_count", str(len(categories))),
        ("top_category_matches", str(TOP_CATEGORY_MATCHES)),
        ("categories_json", json.dumps(categories)),
    ]
    connection.executemany(
        """
        INSERT OR REPLACE INTO metadata (key, value)
        VALUES (?, ?)
        """,
        rows,
    )


def load_sentence_transformer(model_name: str = SEMANTIC_MODEL_NAME):
    print(f"[processor_category_semantic] Loading model: {model_name}")
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "sentence-transformers is required for semantic category matching. "
            "Install it from requirements.txt."
        ) from error

    return SentenceTransformer(model_name)


def article_title_first_sentence_text(article: WikiArticle) -> str:
    first_sentence = extract_lead_text(article.body, "first_sentence")
    if not first_sentence:
        return article.title
    return f"{article.title}. {first_sentence}"


def article_title_first_two_sentences_text(article: WikiArticle) -> str:
    first_two_sentences = extract_lead_text(article.body, "first_two_sentences")
    if not first_two_sentences:
        return article.title
    return f"{article.title}. {first_two_sentences}"


def encode_texts(model, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = model.encode(
        texts,
        batch_size=min(len(texts), 128),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def top_category_matches(
    text_embeddings: np.ndarray,
    category_embeddings: np.ndarray,
    top_k: int = TOP_CATEGORY_MATCHES,
) -> tuple[np.ndarray, np.ndarray]:
    similarity = text_embeddings @ category_embeddings.T
    top_indexes = np.argsort(-similarity, axis=1)[:, :top_k]
    top_scores = np.take_along_axis(similarity, top_indexes, axis=1)
    return top_indexes, top_scores


def serialize_top_matches(
    categories: list[str],
    match_indexes: np.ndarray,
    match_scores: np.ndarray,
    row_index: int,
) -> tuple[str, str]:
    category_names = [
        categories[int(category_index)]
        for category_index in match_indexes[row_index].tolist()
    ]
    scores = [float(score) for score in match_scores[row_index].tolist()]
    return json.dumps(category_names), json.dumps(scores)


def write_batch(connection: sqlite3.Connection, batch: list[tuple]) -> None:
    if not batch:
        return

    connection.executemany(
        """
        INSERT OR REPLACE INTO article_category_matches (
            source_file,
            article_index,
            title,
            title_first_sentence_categories_json,
            title_first_sentence_scores_json,
            title_first_two_sentences_categories_json,
            title_first_two_sentences_scores_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )


def materialize_category_semantic_matches(
    input_db_path: Path = DEFAULT_INPUT_DB_PATH,
    output_db_path: Path = DEFAULT_OUTPUT_DB_PATH,
    questions_path: Path = DEFAULT_QUESTIONS_JSON_PATH,
    batch_size: int = 256,
) -> int:
    initialize_output_database(output_db_path)
    categories = load_unique_categories(questions_path)
    if not categories:
        raise RuntimeError("No categories found in questions file.")

    model = load_sentence_transformer()
    category_embeddings = encode_texts(model, categories)

    total = 0
    skipped_redirects = 0
    start_time = time.time()

    articles_batch: list[WikiArticle] = []

    with sqlite3.connect(output_db_path) as connection:
        write_metadata(connection, categories)

        def flush_batch() -> None:
            nonlocal total
            if not articles_batch:
                return

            title_first_sentence_texts = [
                article_title_first_sentence_text(article)
                for article in articles_batch
            ]
            title_first_two_sentences_texts = [
                article_title_first_two_sentences_text(article)
                for article in articles_batch
            ]

            title_first_sentence_embeddings = encode_texts(model, title_first_sentence_texts)
            title_first_two_sentences_embeddings = encode_texts(model, title_first_two_sentences_texts)

            title_first_sentence_best_indexes, title_first_sentence_best_scores = top_category_matches(
                title_first_sentence_embeddings,
                category_embeddings,
            )
            title_first_two_sentences_best_indexes, title_first_two_sentences_best_scores = top_category_matches(
                title_first_two_sentences_embeddings,
                category_embeddings,
            )

            rows = []
            for batch_index, article in enumerate(articles_batch):
                title_first_sentence_categories_json, title_first_sentence_scores_json = serialize_top_matches(
                    categories,
                    title_first_sentence_best_indexes,
                    title_first_sentence_best_scores,
                    batch_index,
                )
                title_first_two_sentences_categories_json, title_first_two_sentences_scores_json = serialize_top_matches(
                    categories,
                    title_first_two_sentences_best_indexes,
                    title_first_two_sentences_best_scores,
                    batch_index,
                )
                rows.append(
                    (
                        article.source_file,
                        article.article_index,
                        article.title,
                        title_first_sentence_categories_json,
                        title_first_sentence_scores_json,
                        title_first_two_sentences_categories_json,
                        title_first_two_sentences_scores_json,
                    )
                )

            write_batch(connection, rows)
            total += len(rows)
            articles_batch.clear()

        for article in iter_articles_from_db(input_db_path):
            if article.is_redirect:
                skipped_redirects += 1
                continue

            articles_batch.append(article)

            if len(articles_batch) >= batch_size:
                flush_batch()
                elapsed = time.time() - start_time
                print(
                    f"[processor_category_semantic] Articles: {total} | "
                    f"Skipped redirects: {skipped_redirects} | Elapsed: {elapsed:.2f}s"
                )

        flush_batch()
        connection.commit()

    elapsed = time.time() - start_time
    print(
        f"[processor_category_semantic] Done | Articles: {total} | "
        f"Skipped redirects: {skipped_redirects} | Categories: {len(categories)} | "
        f"Elapsed: {elapsed:.2f}s"
    )
    return total


if __name__ == "__main__":
    total = materialize_category_semantic_matches()
    print(f"Stored semantic category matches for {total} articles in {DEFAULT_OUTPUT_DB_PATH}")
