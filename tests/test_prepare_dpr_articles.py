import json
import sqlite3

from src.prepare_dpr_articles import export_dpr_articles


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
                    "Normal Article",
                    (
                        "CATEGORIES: Example\n\n"
                        "Sentence one. Sentence two? Sentence three!\n\n"
                        "Later paragraph."
                    ),
                    "a.txt",
                    0,
                    0,
                ),
                (
                    "Redirect Article",
                    "#REDIRECT [[Normal Article]]",
                    "b.txt",
                    1,
                    1,
                ),
            ],
        )
        connection.commit()


def test_export_dpr_articles_skips_redirects_and_extracts_variants(tmp_path):
    db_path = tmp_path / "wiki_articles.sqlite3"
    output_path = tmp_path / "dpr_articles.jsonl"
    summary_path = tmp_path / "dpr_articles_summary.json"
    initialize_articles_database(db_path)

    summary = export_dpr_articles(
        input_db_path=db_path,
        output_path=output_path,
        summary_path=summary_path,
    )

    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(output_lines) == 1

    record = json.loads(output_lines[0])
    assert record["doc_id"] == "a.txt:0"
    assert record["title"] == "Normal Article"
    assert record["first_sentence_text"] == "Sentence one."
    assert record["first_two_sentences_text"] == "Sentence one. Sentence two?"
    assert record["first_paragraph_text"] == "Sentence one. Sentence two? Sentence three!"
    assert (
        record["entire_article_text"]
        == "Sentence one. Sentence two? Sentence three!\n\nLater paragraph."
    )

    assert summary.total_articles_scanned == 2
    assert summary.exported_articles == 1
    assert summary.skipped_redirects == 1

    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_summary["articles_with_fewer_than_two_sentences"] == 0
