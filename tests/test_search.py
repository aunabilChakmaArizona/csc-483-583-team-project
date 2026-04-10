import sqlite3

from src.processor4_index import materialize_whoosh_index
from src.search import multi_search, search


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
                ("Alpha", "apple banana", "a.txt", 0, 0),
                ("Beta", "banana carrot", "b.txt", 1, 0),
                ("Gamma", "durian", "c.txt", 2, 0),
            ],
        )
        connection.commit()


def test_search_returns_ranked_matching_documents(tmp_path, monkeypatch):
    db_path = tmp_path / "wiki_articles_step2_tokens.sqlite3"
    index_dir = tmp_path / "whoosh_index"
    initialize_articles_database(db_path)
    monkeypatch.setattr("src.search.get_stop_words", lambda: frozenset({"the", "and"}))

    total = materialize_whoosh_index(input_db_path=db_path, index_dir=index_dir, batch_size=2)

    assert total == 3

    results = search("banana carrot", limit=2, index_dir=index_dir)

    assert [result["title"] for result in results] == ["Beta", "Alpha"]


def test_search_returns_empty_list_for_blank_query(tmp_path, monkeypatch):
    db_path = tmp_path / "wiki_articles_step2_tokens.sqlite3"
    index_dir = tmp_path / "whoosh_index"
    initialize_articles_database(db_path)
    monkeypatch.setattr("src.search.get_stop_words", lambda: frozenset({"the", "and"}))
    materialize_whoosh_index(input_db_path=db_path, index_dir=index_dir)

    assert search("   ", index_dir=index_dir) == []


def test_search_normalizes_query_with_processor3_tokenize_pipeline(tmp_path, monkeypatch):
    db_path = tmp_path / "wiki_articles_step2_tokens.sqlite3"
    index_dir = tmp_path / "whoosh_index"
    initialize_articles_database(db_path)
    monkeypatch.setattr("src.search.get_stop_words", lambda: frozenset({"the", "and"}))
    materialize_whoosh_index(input_db_path=db_path, index_dir=index_dir)

    results = search("THE carrot,", index_dir=index_dir)

    assert [result["title"] for result in results] == ["Beta"]


def test_multi_search_reuses_single_index_open(tmp_path, monkeypatch):
    db_path = tmp_path / "wiki_articles_step2_tokens.sqlite3"
    index_dir = tmp_path / "whoosh_index"
    initialize_articles_database(db_path)
    monkeypatch.setattr("src.search.get_stop_words", lambda: frozenset({"the", "and"}))
    materialize_whoosh_index(input_db_path=db_path, index_dir=index_dir)

    open_calls = 0

    def counting_open_index(path):
        nonlocal open_calls
        open_calls += 1
        from src.processor4_index import open_index as real_open_index

        return real_open_index(path)

    monkeypatch.setattr("src.search.open_index", counting_open_index)

    results = multi_search(["banana", "carrot", "   "], limit=2, index_dir=index_dir)

    assert open_calls == 1
    assert [entry["query"] for entry in results] == ["banana", "carrot", "   "]
    assert [hit["title"] for hit in results[0]["results"]] == ["Alpha", "Beta"]
    assert [hit["title"] for hit in results[1]["results"]] == ["Beta"]
    assert results[2]["results"] == []
