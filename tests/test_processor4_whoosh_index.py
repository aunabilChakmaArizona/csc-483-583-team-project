import sqlite3

from src.processor4_whoosh_index import materialize_whoosh_default_index
from src.search import search_whoosh_default


def initialize_clean_articles_database(db_path):
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
                ("Alpha", "Apple, BANANA and carrot.", "a.txt", 0, 0),
                ("Beta", "Durian fruit.", "b.txt", 1, 0),
            ],
        )
        connection.commit()


def test_whoosh_default_index_uses_whoosh_text_analysis(tmp_path):
    db_path = tmp_path / "wiki_articles_step1_clean.sqlite3"
    index_dir = tmp_path / "whoosh_default_index"
    initialize_clean_articles_database(db_path)

    total = materialize_whoosh_default_index(input_db_path=db_path, index_dir=index_dir)

    assert total == 2

    results = search_whoosh_default("BANANA", index_dir=index_dir)

    assert [result["title"] for result in results] == ["Alpha"]
