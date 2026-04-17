import sqlite3

from src.processor_redirect import extract_redirect_target, materialize_redirect_mappings


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
                ("Alias A", "#REDIRECT [[Alias_B#History]]", "a.txt", 0, 1),
                ("Alias B", "#REDIRECT [[Core Article]]", "b.txt", 1, 1),
                ("Core Article", "Canonical body text.", "c.txt", 2, 0),
                ("Broken Alias", "#REDIRECT [[Missing Target]]", "d.txt", 3, 1),
            ],
        )
        connection.commit()


def test_extract_redirect_target_parses_title_and_section():
    target = extract_redirect_target("#REDIRECT [[Target Article#Legacy section|display]]")

    assert target is not None
    assert target.title == "Target Article"
    assert target.section == "Legacy section"


def test_materialize_redirect_mappings_resolves_chains_and_missing_targets(tmp_path):
    input_db_path = tmp_path / "wiki_articles.sqlite3"
    output_db_path = tmp_path / "wiki_redirects.sqlite3"
    initialize_articles_database(input_db_path)

    total = materialize_redirect_mappings(input_db_path=input_db_path, output_db_path=output_db_path)

    assert total == 3

    with sqlite3.connect(output_db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                redirect_title,
                target_title,
                target_section,
                resolved_title,
                resolved_source_file,
                resolved_article_index,
                resolution_status,
                hops
            FROM redirects
            ORDER BY redirect_article_index
            """
        ).fetchall()

    assert rows == [
        ("Alias A", "Alias B", "History", "Core Article", "c.txt", 2, "resolved", 2),
        ("Alias B", "Core Article", "", "Core Article", "c.txt", 2, "resolved", 1),
        ("Broken Alias", "Missing Target", "", "", "", -1, "missing_target", 0),
    ]
