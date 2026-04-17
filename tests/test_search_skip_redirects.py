import sqlite3

from src.processor4_whoosh_title_body_index import materialize_whoosh_title_body_index
from src.search import search_whoosh_title_body, search_whoosh_weighted


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
                ("Core Article", "canonical body text", "a.txt", 0, 0),
                ("Alias Name", "", "b.txt", 1, 1),
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


def test_skip_redirects_filters_redirect_docs_from_search_results(tmp_path):
    db_path = tmp_path / "wiki_articles_step1_clean.sqlite3"
    index_dir = tmp_path / "whoosh_title_body_index"
    initialize_clean_articles_database(db_path)

    materialize_whoosh_title_body_index(input_db_path=db_path, index_dir=index_dir)

    with_redirects = search_whoosh_title_body("Alias Name", index_dir=index_dir)
    without_redirects = search_whoosh_title_body(
        "Alias Name",
        index_dir=index_dir,
        skip_redirects=True,
    )

    assert [result["title"] for result in with_redirects] == ["Alias Name"]
    assert without_redirects == []


def test_skip_redirects_disables_redirect_component_in_weighted_search(tmp_path):
    db_path = tmp_path / "wiki_articles_step1_clean.sqlite3"
    index_dir = tmp_path / "whoosh_title_body_index"
    redirect_db_path = tmp_path / "wiki_redirects.sqlite3"
    initialize_clean_articles_database(db_path)
    initialize_redirect_database(redirect_db_path)

    materialize_whoosh_title_body_index(input_db_path=db_path, index_dir=index_dir)

    with_redirects = search_whoosh_weighted(
        "Alias Name",
        index_dir=index_dir,
        redirect_db_path=redirect_db_path,
    )
    without_redirects = search_whoosh_weighted(
        "Alias Name",
        index_dir=index_dir,
        redirect_db_path=redirect_db_path,
        skip_redirects=True,
    )

    assert [result["title"] for result in with_redirects] == ["Core Article"]
    assert with_redirects[0]["redirect_score"] > 0
    assert without_redirects == []
