import sqlite3

from src.processor4_cole_redirect_index import materialize_whoosh_cole_redirect_index
from src.search import search_whoosh_cole


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
                ("Nebraska", "A plains state in the midwestern United States.", "a.txt", 0, 0),
                ("Corn", "A cereal grain grown in many places.", "b.txt", 1, 0),
                ("Nebraska redirect", "#REDIRECT [[Nebraska]]", "c.txt", 2, 1),
            ],
        )
        connection.commit()


def test_cole_redirect_index_searches_only_redirects(tmp_path):
    db_path = tmp_path / "wiki_articles_step1_clean.sqlite3"
    index_dir = tmp_path / "whoosh_cole_redirect_index"
    initialize_clean_articles_database(db_path)

    total = materialize_whoosh_cole_redirect_index(input_db_path=db_path, index_dir=index_dir)

    assert total == 1

    redirect_title_results = search_whoosh_cole("redirect", index_dir=index_dir)
    redirect_body_results = search_whoosh_cole("Nebraska", index_dir=index_dir)
    non_redirect_results = search_whoosh_cole("cereal grain", index_dir=index_dir)

    assert [result["title"] for result in redirect_title_results] == ["Nebraska redirect"]
    assert [result["title"] for result in redirect_body_results] == ["Nebraska redirect"]
    assert non_redirect_results == []
    assert redirect_title_results[0]["is_redirect"] == 1
