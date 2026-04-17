import sqlite3

from src.processor4_cole_first_paragraph_index import (
    materialize_whoosh_cole_first_paragraph_index,
)
from src.processor4_cole_first_sentence_index import (
    materialize_whoosh_cole_first_sentence_index,
)
from src.processor4_cole_first_two_sentences_index import (
    materialize_whoosh_cole_first_two_sentences_index,
)
from src.processor4_lead_index_common import extract_lead_text, search_index


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
                    "Lead Article",
                    (
                        "CATEGORIES: Branch Davidianism, Adventism\n\n"
                        "alphaword sentence one. betaword sentence two.\n\n"
                        "gammaword later paragraph."
                    ),
                    "a.txt",
                    0,
                    0,
                ),
                (
                    "Redirect Article",
                    "#REDIRECT [[Lead Article]]",
                    "b.txt",
                    1,
                    1,
                ),
            ],
        )
        connection.commit()


def test_extract_lead_text_ignores_categories_and_uses_clean_lead():
    body = (
        "CATEGORIES: Branch Davidianism, Adventism\n\n"
        "alphaword sentence one. betaword sentence two.\n\n"
        "gammaword later paragraph."
    )

    assert extract_lead_text(body, "first_sentence") == "alphaword sentence one."
    assert (
        extract_lead_text(body, "first_two_sentences")
        == "alphaword sentence one. betaword sentence two."
    )
    assert (
        extract_lead_text(body, "first_paragraph")
        == "alphaword sentence one. betaword sentence two."
    )


def test_lead_indexes_store_expected_scope_and_skip_redirects(tmp_path):
    db_path = tmp_path / "wiki_articles.sqlite3"
    first_sentence_index_dir = tmp_path / "whoosh_cole_first_sentence_index"
    first_two_sentences_index_dir = tmp_path / "whoosh_cole_first_two_sentences_index"
    first_paragraph_index_dir = tmp_path / "whoosh_cole_first_paragraph_index"
    initialize_articles_database(db_path)

    total_first_sentence = materialize_whoosh_cole_first_sentence_index(
        input_db_path=db_path,
        index_dir=first_sentence_index_dir,
    )
    total_first_two_sentences = materialize_whoosh_cole_first_two_sentences_index(
        input_db_path=db_path,
        index_dir=first_two_sentences_index_dir,
    )
    total_first_paragraph = materialize_whoosh_cole_first_paragraph_index(
        input_db_path=db_path,
        index_dir=first_paragraph_index_dir,
    )

    assert total_first_sentence == 1
    assert total_first_two_sentences == 1
    assert total_first_paragraph == 1

    assert search_index("alphaword", first_sentence_index_dir) == ["Lead Article"]
    assert search_index("betaword", first_sentence_index_dir) == []
    assert search_index("gammaword", first_sentence_index_dir) == []

    assert search_index("betaword", first_two_sentences_index_dir) == ["Lead Article"]
    assert search_index("gammaword", first_two_sentences_index_dir) == []

    assert search_index("betaword", first_paragraph_index_dir) == ["Lead Article"]
    assert search_index("gammaword", first_paragraph_index_dir) == []

    assert search_index("Redirect", first_sentence_index_dir) == []
