import json
import sqlite3

import numpy as np

from src.processor_category_semantic import (
    SEMANTIC_MODEL_NAME,
    article_title_first_sentence_text,
    article_title_first_two_sentences_text,
    load_unique_categories,
    materialize_category_semantic_matches,
)


def initialize_articles_database(db_path):
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                source_file TEXT NOT NULL,
                article_index INTEGER NOT NULL,
                is_redirect INTEGER NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO articles
            (title, body, source_file, article_index, is_redirect)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    "The Washington Post",
                    "CATEGORIES: Newspapers\n\nThe Washington Post is a newspaper in Washington, D.C.",
                    "a.txt",
                    0,
                    0,
                ),
                (
                    "Taiwan",
                    "Taiwan is an island near mainland China.",
                    "a.txt",
                    1,
                    0,
                ),
                (
                    "Redirect Page",
                    "#REDIRECT [[The Washington Post]]",
                    "a.txt",
                    2,
                    1,
                ),
            ],
        )
        connection.commit()


def initialize_questions_json(path):
    rows = [
        {"category": "NEWSPAPERS", "clue": "x", "answer": "x"},
        {"category": "OLD YEAR'S RESOLUTIONS", "clue": "x", "answer": "x"},
        {"category": "NEWSPAPERS", "clue": "x", "answer": "x"},
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        vectors = []
        for text in texts:
            normalized = text.lower()
            vectors.append(
                [
                    1.0 if "newspaper" in normalized or "washington post" in normalized else 0.0,
                    1.0
                    if (
                        "old year's resolutions" in normalized
                        or "resolution" in normalized
                        or "taiwan" in normalized
                        or "china" in normalized
                    )
                    else 0.0,
                ]
            )

        array = np.asarray(vectors, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            array = array / norms
        return array


def test_load_unique_categories_deduplicates_and_preserves_order(tmp_path):
    questions_path = tmp_path / "questions.json"
    initialize_questions_json(questions_path)

    categories = load_unique_categories(questions_path)

    assert categories == ["NEWSPAPERS", "OLD YEAR'S RESOLUTIONS"]


def test_materialize_category_semantic_matches_stores_top_matches(tmp_path, monkeypatch):
    db_path = tmp_path / "wiki_articles.sqlite3"
    output_db_path = tmp_path / "wiki_category_semantic.sqlite3"
    questions_path = tmp_path / "questions.json"
    initialize_articles_database(db_path)
    initialize_questions_json(questions_path)

    monkeypatch.setattr(
        "src.processor_category_semantic.load_sentence_transformer",
        lambda: FakeSentenceTransformer("fake-model"),
    )

    total = materialize_category_semantic_matches(
        input_db_path=db_path,
        output_db_path=output_db_path,
        questions_path=questions_path,
        batch_size=2,
    )

    assert total == 2

    with sqlite3.connect(output_db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                title,
                title_first_sentence_categories_json,
                title_first_two_sentences_categories_json
            FROM article_category_matches
            ORDER BY article_index
            """
        ).fetchall()
        metadata_rows = dict(connection.execute("SELECT key, value FROM metadata").fetchall())

    parsed_rows = [
        (
            title,
            json.loads(first_sentence_categories_json),
            json.loads(first_two_sentences_categories_json),
        )
        for title, first_sentence_categories_json, first_two_sentences_categories_json in rows
    ]

    assert parsed_rows == [
        (
            "The Washington Post",
            ["NEWSPAPERS", "OLD YEAR'S RESOLUTIONS"],
            ["NEWSPAPERS", "OLD YEAR'S RESOLUTIONS"],
        ),
        (
            "Taiwan",
            ["OLD YEAR'S RESOLUTIONS", "NEWSPAPERS"],
            ["OLD YEAR'S RESOLUTIONS", "NEWSPAPERS"],
        ),
    ]
    assert metadata_rows["category_count"] == "2"
    assert metadata_rows["model_name"] == SEMANTIC_MODEL_NAME
    assert metadata_rows["top_category_matches"] == "10"


def test_article_title_first_sentence_text_appends_clean_first_sentence():
    article = type(
        "Article",
        (),
        {
            "title": "The Washington Post",
            "body": (
                "CATEGORIES: Newspapers\n\n"
                "The Washington Post is a newspaper in Washington, D.C. "
                "It has won many awards."
            ),
        },
    )()

    text = article_title_first_sentence_text(article)

    assert text == "The Washington Post. The Washington Post is a newspaper in Washington, D.C."


def test_article_title_first_two_sentences_text_appends_clean_first_two_sentences():
    article = type(
        "Article",
        (),
        {
            "title": "The Washington Post",
            "body": (
                "CATEGORIES: Newspapers\n\n"
                "The Washington Post is a newspaper in Washington, D.C. "
                "It has won many awards. "
                "A third sentence should not be included."
            ),
        },
    )()

    text = article_title_first_two_sentences_text(article)

    assert text == (
        "The Washington Post. The Washington Post is a newspaper in Washington, D.C. "
        "It has won many awards."
    )
