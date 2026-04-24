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
from src.processor4_cole_index import materialize_whoosh_cole_index
from src.processor4_cole_redirect_index import materialize_whoosh_cole_redirect_index
from src.search import (
    WEIGHTED_FAISS_WEIGHT,
    WEIGHTED_NATURAL_QUESTIONS_CONCAT_QWEN_SUMMARY_FAISS_WEIGHT,
    WEIGHTED_QWEN_SUMMARY_FAISS_WEIGHT,
    search_whoosh_weighted_cole,
)


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
                (
                    "Core Article",
                    "leadword first sentence in 2011. secondword second sentence.\n\notherword later paragraph.",
                    "a.txt",
                    0,
                    0,
                ),
                ("Alias Name", "#REDIRECT [[Core Article]]", "b.txt", 1, 1),
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


def test_weighted_cole_search_maps_redirect_hits_to_canonical_article(tmp_path):
    db_path = tmp_path / "wiki_articles_step1_clean.sqlite3"
    cole_index_dir = tmp_path / "whoosh_cole_index"
    first_sentence_index_dir = tmp_path / "whoosh_cole_first_sentence_index"
    first_two_sentences_index_dir = tmp_path / "whoosh_cole_first_two_sentences_index"
    first_paragraph_index_dir = tmp_path / "whoosh_cole_first_paragraph_index"
    redirect_index_dir = tmp_path / "whoosh_cole_redirect_index"
    redirect_db_path = tmp_path / "wiki_redirects.sqlite3"
    initialize_clean_articles_database(db_path)
    initialize_redirect_database(redirect_db_path)

    total_cole = materialize_whoosh_cole_index(input_db_path=db_path, index_dir=cole_index_dir)
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
    total_redirect = materialize_whoosh_cole_redirect_index(
        input_db_path=db_path,
        index_dir=redirect_index_dir,
    )

    assert total_cole == 2
    assert total_first_sentence == 2
    assert total_first_two_sentences == 2
    assert total_first_paragraph == 2
    assert total_redirect == 1

    redirect_results = search_whoosh_weighted_cole(
        "Alias Name",
        index_dir=cole_index_dir,
        first_sentence_index_dir=first_sentence_index_dir,
        first_two_sentences_index_dir=first_two_sentences_index_dir,
        first_paragraph_index_dir=first_paragraph_index_dir,
        redirect_index_dir=redirect_index_dir,
        redirect_db_path=redirect_db_path,
        dpr_faiss_index_dir=tmp_path / "missing_dpr_faiss",
    )
    body_results = search_whoosh_weighted_cole(
        "secondword",
        index_dir=cole_index_dir,
        first_sentence_index_dir=first_sentence_index_dir,
        first_two_sentences_index_dir=first_two_sentences_index_dir,
        first_paragraph_index_dir=first_paragraph_index_dir,
        redirect_index_dir=redirect_index_dir,
        redirect_db_path=redirect_db_path,
        dpr_faiss_index_dir=tmp_path / "missing_dpr_faiss",
    )
    full_body_results = search_whoosh_weighted_cole(
        "otherword",
        index_dir=cole_index_dir,
        first_sentence_index_dir=first_sentence_index_dir,
        first_two_sentences_index_dir=first_two_sentences_index_dir,
        first_paragraph_index_dir=first_paragraph_index_dir,
        redirect_index_dir=redirect_index_dir,
        redirect_db_path=redirect_db_path,
        dpr_faiss_index_dir=tmp_path / "missing_dpr_faiss",
    )
    year_results = search_whoosh_weighted_cole(
        "2011",
        index_dir=cole_index_dir,
        first_sentence_index_dir=first_sentence_index_dir,
        first_two_sentences_index_dir=first_two_sentences_index_dir,
        first_paragraph_index_dir=first_paragraph_index_dir,
        redirect_index_dir=redirect_index_dir,
        redirect_db_path=redirect_db_path,
        dpr_faiss_index_dir=tmp_path / "missing_dpr_faiss",
        year_match_weight=1.0,
    )
    quote_results = search_whoosh_weighted_cole(
        '"leadword first sentence"',
        index_dir=cole_index_dir,
        first_sentence_index_dir=first_sentence_index_dir,
        first_two_sentences_index_dir=first_two_sentences_index_dir,
        first_paragraph_index_dir=first_paragraph_index_dir,
        redirect_index_dir=redirect_index_dir,
        redirect_db_path=redirect_db_path,
        dpr_faiss_index_dir=tmp_path / "missing_dpr_faiss",
    )
    missing_quote_results = search_whoosh_weighted_cole(
        '"leadword missing sentence"',
        index_dir=cole_index_dir,
        first_sentence_index_dir=first_sentence_index_dir,
        first_two_sentences_index_dir=first_two_sentences_index_dir,
        first_paragraph_index_dir=first_paragraph_index_dir,
        redirect_index_dir=redirect_index_dir,
        redirect_db_path=redirect_db_path,
        dpr_faiss_index_dir=tmp_path / "missing_dpr_faiss",
    )

    assert redirect_results == []
    assert [result["title"] for result in body_results] == ["Core Article"]
    assert body_results[0]["body_score"] > 0
    assert "first_sentence_score" not in body_results[0]
    assert "first_two_sentences_score" not in body_results[0]
    assert "first_paragraph_score" not in body_results[0]
    assert [result["title"] for result in full_body_results] == ["Core Article"]
    assert full_body_results[0]["body_score"] > 0
    assert "first_sentence_score" not in full_body_results[0]
    assert "first_two_sentences_score" not in full_body_results[0]
    assert "first_paragraph_score" not in full_body_results[0]
    assert [result["title"] for result in year_results] == ["Core Article"]
    assert year_results[0]["year_match_score"] > 0
    assert [result["title"] for result in quote_results] == ["Core Article"]
    assert "quote_match_score" not in quote_results[0]
    assert missing_quote_results == []


def test_weighted_cole_search_includes_faiss_score_component(tmp_path, monkeypatch):
    db_path = tmp_path / "wiki_articles_step1_clean.sqlite3"
    cole_index_dir = tmp_path / "whoosh_cole_index"
    first_sentence_index_dir = tmp_path / "whoosh_cole_first_sentence_index"
    first_two_sentences_index_dir = tmp_path / "whoosh_cole_first_two_sentences_index"
    first_paragraph_index_dir = tmp_path / "whoosh_cole_first_paragraph_index"
    redirect_index_dir = tmp_path / "whoosh_cole_redirect_index"
    redirect_db_path = tmp_path / "wiki_redirects.sqlite3"
    initialize_clean_articles_database(db_path)
    initialize_redirect_database(redirect_db_path)

    materialize_whoosh_cole_index(input_db_path=db_path, index_dir=cole_index_dir)
    materialize_whoosh_cole_first_sentence_index(
        input_db_path=db_path,
        index_dir=first_sentence_index_dir,
    )
    materialize_whoosh_cole_first_two_sentences_index(
        input_db_path=db_path,
        index_dir=first_two_sentences_index_dir,
    )
    materialize_whoosh_cole_first_paragraph_index(
        input_db_path=db_path,
        index_dir=first_paragraph_index_dir,
    )
    materialize_whoosh_cole_redirect_index(
        input_db_path=db_path,
        index_dir=redirect_index_dir,
    )

    def fake_search_dpr_faiss(
        query_text,
        limit,
        dpr_faiss_index_dir,
        query_embedding=None,
        cache_namespace="default",
        variant_names=None,
    ):
        assert query_text == "dense-only"
        return [
            {
                "title": "Core Article",
                "body": "",
                "source_file": "a.txt",
                "article_index": 0,
                "is_redirect": 0,
                "score": 1.0,
            }
        ]

    monkeypatch.setattr("src.search.search_dpr_faiss", fake_search_dpr_faiss)

    results = search_whoosh_weighted_cole(
        "dense-only",
        index_dir=cole_index_dir,
        first_sentence_index_dir=first_sentence_index_dir,
        first_two_sentences_index_dir=first_two_sentences_index_dir,
        first_paragraph_index_dir=first_paragraph_index_dir,
        redirect_index_dir=redirect_index_dir,
        redirect_db_path=redirect_db_path,
        dpr_faiss_index_dir=tmp_path / "missing_dpr_faiss",
    )

    assert [result["title"] for result in results] == ["Core Article"]
    assert results[0]["faiss_score"] == 1.0
    assert results[0]["score"] == WEIGHTED_FAISS_WEIGHT
    assert results[0]["body_score"] == 0


def test_weighted_cole_search_includes_qwen_summary_faiss_score_component(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "wiki_articles_step1_clean.sqlite3"
    cole_index_dir = tmp_path / "whoosh_cole_index"
    first_sentence_index_dir = tmp_path / "whoosh_cole_first_sentence_index"
    first_two_sentences_index_dir = tmp_path / "whoosh_cole_first_two_sentences_index"
    first_paragraph_index_dir = tmp_path / "whoosh_cole_first_paragraph_index"
    redirect_index_dir = tmp_path / "whoosh_cole_redirect_index"
    redirect_db_path = tmp_path / "wiki_redirects.sqlite3"
    initialize_clean_articles_database(db_path)
    initialize_redirect_database(redirect_db_path)

    materialize_whoosh_cole_index(input_db_path=db_path, index_dir=cole_index_dir)
    materialize_whoosh_cole_first_sentence_index(
        input_db_path=db_path,
        index_dir=first_sentence_index_dir,
    )
    materialize_whoosh_cole_first_two_sentences_index(
        input_db_path=db_path,
        index_dir=first_two_sentences_index_dir,
    )
    materialize_whoosh_cole_first_paragraph_index(
        input_db_path=db_path,
        index_dir=first_paragraph_index_dir,
    )
    materialize_whoosh_cole_redirect_index(
        input_db_path=db_path,
        index_dir=redirect_index_dir,
    )

    def fake_search_dpr_faiss(
        query_text,
        limit,
        dpr_faiss_index_dir,
        query_embedding=None,
        cache_namespace="default",
        variant_names=None,
    ):
        assert query_text == "dense-qwen-only"
        assert cache_namespace == "qwen_summary"
        assert variant_names == ("title_qwen_summary",)
        return [
            {
                "title": "Core Article",
                "body": "",
                "source_file": "a.txt",
                "article_index": 0,
                "is_redirect": 0,
                "score": 1.0,
            }
        ]

    monkeypatch.setattr("src.search.search_dpr_faiss", fake_search_dpr_faiss)

    qwen_weight = WEIGHTED_QWEN_SUMMARY_FAISS_WEIGHT or 1.0
    results = search_whoosh_weighted_cole(
        "dense-qwen-only",
        index_dir=cole_index_dir,
        first_sentence_index_dir=first_sentence_index_dir,
        first_two_sentences_index_dir=first_two_sentences_index_dir,
        first_paragraph_index_dir=first_paragraph_index_dir,
        redirect_index_dir=redirect_index_dir,
        redirect_db_path=redirect_db_path,
        dpr_faiss_index_dir=tmp_path / "missing_dpr_faiss",
        faiss_weight=0.0,
        qwen_summary_faiss_weight=qwen_weight,
        natural_questions_avg_faiss_weight=0.0,
        natural_questions_concat_faiss_weight=0.0,
    )

    assert [result["title"] for result in results] == ["Core Article"]
    assert results[0]["qwen_summary_faiss_score"] == 1.0
    assert results[0]["score"] == qwen_weight
    assert results[0]["body_score"] == 0


def test_weighted_cole_search_includes_concat_qwen_summary_natural_question_component(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "wiki_articles_step1_clean.sqlite3"
    cole_index_dir = tmp_path / "whoosh_cole_index"
    first_sentence_index_dir = tmp_path / "whoosh_cole_first_sentence_index"
    first_two_sentences_index_dir = tmp_path / "whoosh_cole_first_two_sentences_index"
    first_paragraph_index_dir = tmp_path / "whoosh_cole_first_paragraph_index"
    redirect_index_dir = tmp_path / "whoosh_cole_redirect_index"
    redirect_db_path = tmp_path / "wiki_redirects.sqlite3"
    initialize_clean_articles_database(db_path)
    initialize_redirect_database(redirect_db_path)

    materialize_whoosh_cole_index(input_db_path=db_path, index_dir=cole_index_dir)
    materialize_whoosh_cole_first_sentence_index(
        input_db_path=db_path,
        index_dir=first_sentence_index_dir,
    )
    materialize_whoosh_cole_first_two_sentences_index(
        input_db_path=db_path,
        index_dir=first_two_sentences_index_dir,
    )
    materialize_whoosh_cole_first_paragraph_index(
        input_db_path=db_path,
        index_dir=first_paragraph_index_dir,
    )
    materialize_whoosh_cole_redirect_index(
        input_db_path=db_path,
        index_dir=redirect_index_dir,
    )

    def fake_search_dpr_faiss(
        query_text,
        limit,
        dpr_faiss_index_dir,
        query_embedding=None,
        cache_namespace="default",
        variant_names=None,
    ):
        assert query_text == "dense-qwen-concat"
        assert cache_namespace == "natural_questions_concat_qwen_summary"
        assert variant_names == ("title_qwen_summary",)
        return [
            {
                "title": "Core Article",
                "body": "",
                "source_file": "a.txt",
                "article_index": 0,
                "is_redirect": 0,
                "score": 1.0,
            }
        ]

    monkeypatch.setattr("src.search.search_dpr_faiss", fake_search_dpr_faiss)

    concat_weight = WEIGHTED_NATURAL_QUESTIONS_CONCAT_QWEN_SUMMARY_FAISS_WEIGHT or 1.0
    results = search_whoosh_weighted_cole(
        "dense-qwen-concat",
        index_dir=cole_index_dir,
        first_sentence_index_dir=first_sentence_index_dir,
        first_two_sentences_index_dir=first_two_sentences_index_dir,
        first_paragraph_index_dir=first_paragraph_index_dir,
        redirect_index_dir=redirect_index_dir,
        redirect_db_path=redirect_db_path,
        dpr_faiss_index_dir=tmp_path / "missing_dpr_faiss",
        faiss_weight=0.0,
        qwen_summary_faiss_weight=0.0,
        natural_questions_avg_faiss_weight=0.0,
        natural_questions_concat_faiss_weight=0.0,
        natural_questions_concat_qwen_summary_faiss_weight=concat_weight,
        natural_questions_concat_embedding=[0.1],
    )

    assert [result["title"] for result in results] == ["Core Article"]
    assert results[0]["natural_questions_concat_qwen_summary_faiss_score"] == 1.0
    assert results[0]["score"] == concat_weight
