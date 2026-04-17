import sqlite3

from src.processor4_whoosh_title_body_index import materialize_whoosh_title_body_index
from src.search import search_whoosh_weighted


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
                ("Core Article", "A canonical article body.", "a.txt", 0, 0),
                ("Alias Name", "", "b.txt", 1, 1),
                ("Body Match", "Useful grain term in the body.", "c.txt", 2, 0),
            ],
        )
        connection.commit()


def initialize_redirect_database(db_path):
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE redirects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                redirect_title TEXT NOT NULL,
                redirect_source_file TEXT NOT NULL,
                redirect_article_index INTEGER NOT NULL,
                target_title TEXT NOT NULL,
                target_section TEXT NOT NULL,
                resolved_title TEXT NOT NULL,
                resolved_source_file TEXT NOT NULL,
                resolved_article_index INTEGER NOT NULL,
                resolution_status TEXT NOT NULL,
                hops INTEGER NOT NULL,
                UNIQUE(redirect_source_file, redirect_article_index)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO redirects (
                redirect_title,
                redirect_source_file,
                redirect_article_index,
                target_title,
                target_section,
                resolved_title,
                resolved_source_file,
                resolved_article_index,
                resolution_status,
                hops
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("Alias Name", "b.txt", 1, "Core Article", "", "Core Article", "a.txt", 0, "resolved", 1),
        )
        connection.commit()


def test_weighted_search_maps_redirect_hits_to_canonical_article(tmp_path):
    db_path = tmp_path / "wiki_articles_step1_clean.sqlite3"
    index_dir = tmp_path / "whoosh_title_body_index"
    redirect_db_path = tmp_path / "wiki_redirects.sqlite3"
    initialize_clean_articles_database(db_path)
    initialize_redirect_database(redirect_db_path)

    total = materialize_whoosh_title_body_index(input_db_path=db_path, index_dir=index_dir)

    assert total == 3

    redirect_results = search_whoosh_weighted(
        "Alias Name",
        index_dir=index_dir,
        redirect_db_path=redirect_db_path,
    )
    body_results = search_whoosh_weighted(
        "grain",
        index_dir=index_dir,
        redirect_db_path=redirect_db_path,
    )

    assert [result["title"] for result in redirect_results] == ["Core Article"]
    assert redirect_results[0]["is_redirect"] == 0
    assert redirect_results[0]["redirect_score"] > 0
    assert redirect_results[0]["title_score"] == 0
    assert [result["title"] for result in body_results] == ["Body Match"]
    assert body_results[0]["body_score"] > 0
