from src.run_test_100 import (
    JeopardyQuestion,
    evaluate_questions,
    get_precomputed_dense_query_embeddings,
    merge_ranked_results,
    question_queries,
    search_questions,
)
import numpy as np


def make_hit(title: str, score: float = 1.0, article_index: int = 0) -> dict:
    return {
        "title": title,
        "body": "",
        "source_file": "sample.txt",
        "article_index": article_index,
        "is_redirect": 0,
        "score": score,
    }


def test_question_queries_adds_paraphrases_and_deduplicates(monkeypatch):
    question = JeopardyQuestion(category="History", clue="Simple raw clue", answer="Alpha")
    monkeypatch.setattr(
        "src.run_test_100.generate_paraphrases",
        lambda sentence, count=4: [
            "Simple raw clue",
            "Alt clue one",
            "Alt clue one",
            "Alt clue two",
        ],
    )

    queries = question_queries(
        question,
        query_mode="full",
        include_category=False,
        paraphrase=True,
        paraphrase_count=4,
    )

    assert queries == ["Simple raw clue", "Alt clue one", "Alt clue two"]


def test_merge_ranked_results_uses_rank_fusion_ordering():
    searches = [
        {
            "query": "raw",
            "results": [
                make_hit("Alpha", score=2.0, article_index=1),
                make_hit("Beta", score=1.5, article_index=2),
            ],
        },
        {
            "query": "paraphrase",
            "results": [
                make_hit("Beta", score=3.0, article_index=2),
                make_hit("Gamma", score=1.0, article_index=3),
            ],
        },
    ]

    merged = merge_ranked_results(searches, limit=3)

    assert [result["title"] for result in merged] == ["Beta", "Alpha", "Gamma"]
    assert merged[0]["matched_queries"] == 2


def test_search_questions_expands_queries_and_merges_results(monkeypatch):
    question = JeopardyQuestion(category="Science", clue="Raw clue", answer="Beta")
    captured = {}

    monkeypatch.setattr(
        "src.run_test_100.generate_paraphrases",
        lambda sentence, count=4: ["Alt one", "Alt two", "Alt three", "Alt four"],
    )

    def fake_run_search_batch(
        queries,
        limit,
        mode,
        categories=None,
        weighted=False,
        weight_equal=False,
        weighted_cole=False,
        skip_redirects=False,
        dense_query_embeddings=None,
        natural_questions_embeddings=None,
        natural_questions_concat_embeddings=None,
        search_progress_every=0,
    ):
        captured["queries"] = queries
        captured["categories"] = categories
        captured["limit"] = limit
        captured["mode"] = mode
        captured["weighted"] = weighted
        captured["weight_equal"] = weight_equal
        captured["weighted_cole"] = weighted_cole
        captured["skip_redirects"] = skip_redirects
        captured["dense_query_embeddings"] = dense_query_embeddings
        captured["natural_questions_embeddings"] = natural_questions_embeddings
        captured["natural_questions_concat_embeddings"] = natural_questions_concat_embeddings
        captured["search_progress_every"] = search_progress_every
        query_to_results = {
            "Raw clue": [make_hit("Alpha", article_index=1), make_hit("Beta", article_index=2)],
            "Alt one": [make_hit("Beta", article_index=2), make_hit("Gamma", article_index=3)],
            "Alt two": [make_hit("Gamma", article_index=3)],
            "Alt three": [],
            "Alt four": [make_hit("Beta", article_index=2)],
        }
        return [{"query": query, "results": query_to_results[query]} for query in queries]

    monkeypatch.setattr("src.run_test_100.run_search_batch", fake_run_search_batch)

    searches = search_questions(
        [question],
        limit=3,
        mode="whoosh",
        query_mode="full",
        include_category=False,
        paraphrase=True,
        paraphrase_count=4,
        paraphrase_fetch_limit=1000,
    )

    assert captured == {
        "queries": ["Raw clue", "Alt one", "Alt two", "Alt three", "Alt four"],
        "categories": ["Science", "Science", "Science", "Science", "Science"],
        "limit": 1000,
        "mode": "whoosh",
        "weighted": False,
        "weight_equal": False,
        "weighted_cole": False,
        "skip_redirects": False,
        "dense_query_embeddings": None,
        "natural_questions_embeddings": None,
        "natural_questions_concat_embeddings": None,
        "search_progress_every": 0,
    }
    assert searches[0]["queries"] == ["Raw clue", "Alt one", "Alt two", "Alt three", "Alt four"]
    assert [result["title"] for result in searches[0]["results"]] == ["Beta", "Gamma", "Alpha"]


def test_search_questions_passes_weight_equal_to_search_batch(monkeypatch):
    question = JeopardyQuestion(category="Science", clue="Raw clue", answer="Beta")
    captured = {}

    def fake_run_search_batch(
        queries,
        limit,
        mode,
        categories=None,
        weighted=False,
        weight_equal=False,
        weighted_cole=False,
        skip_redirects=False,
        dense_query_embeddings=None,
        natural_questions_embeddings=None,
        natural_questions_concat_embeddings=None,
        search_progress_every=0,
    ):
        captured["queries"] = queries
        captured["categories"] = categories
        captured["limit"] = limit
        captured["mode"] = mode
        captured["weighted"] = weighted
        captured["weight_equal"] = weight_equal
        captured["weighted_cole"] = weighted_cole
        captured["skip_redirects"] = skip_redirects
        captured["dense_query_embeddings"] = dense_query_embeddings
        captured["natural_questions_embeddings"] = natural_questions_embeddings
        captured["natural_questions_concat_embeddings"] = natural_questions_concat_embeddings
        captured["search_progress_every"] = search_progress_every
        return [{"query": queries[0], "results": [make_hit("Beta", article_index=2)]}]

    monkeypatch.setattr("src.run_test_100.run_search_batch", fake_run_search_batch)
    monkeypatch.setattr(
        "src.run_test_100.get_precomputed_dense_query_embeddings",
        lambda questions, include_category: [np.array([1.0, 2.0], dtype=np.float32)],
    )
    monkeypatch.setattr(
        "src.run_test_100.get_precomputed_natural_question_dense_embeddings",
        lambda questions: (
            [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)],
            [np.array([5.0, 6.0], dtype=np.float32)],
        ),
    )

    searches = search_questions(
        [question],
        limit=5,
        mode="whoosh_title_body",
        query_mode="full",
        include_category=False,
        weighted=False,
        weight_equal=True,
        skip_redirects=True,
        paraphrase=False,
    )

    assert {
        key: value
        for key, value in captured.items()
        if key not in {"dense_query_embeddings", "natural_questions_embeddings", "natural_questions_concat_embeddings"}
    } == {
        "queries": ["Raw clue"],
        "categories": ["Science"],
        "limit": 5,
        "mode": "whoosh_title_body",
        "weighted": False,
        "weight_equal": True,
        "weighted_cole": False,
        "skip_redirects": True,
        "search_progress_every": 0,
    }
    assert len(captured["dense_query_embeddings"]) == 1
    assert captured["dense_query_embeddings"][0].tolist() == [1.0, 2.0]
    assert len(captured["natural_questions_embeddings"]) == 1
    assert captured["natural_questions_embeddings"][0].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert len(captured["natural_questions_concat_embeddings"]) == 1
    assert captured["natural_questions_concat_embeddings"][0].tolist() == [5.0, 6.0]
    assert [result["title"] for result in searches[0]["results"]] == ["Beta"]


def test_get_precomputed_dense_query_embeddings_uses_saved_clue_only(monkeypatch):
    question = JeopardyQuestion(category="Science", clue="Raw clue", answer="Beta")
    saved = {
        "clue_only_texts": np.array(["The clue is: Raw clue"], dtype=object),
        "clue_only_embeddings": np.array([[1.0, 2.0]], dtype=np.float32),
        "category_plus_clue_texts": np.array(
            ["The category is: Science. The clue is: Raw clue"], dtype=object
        ),
        "category_plus_clue_embeddings": np.array([[3.0, 4.0]], dtype=np.float32),
    }
    monkeypatch.setattr(
        "src.run_test_100.load_precomputed_question_embeddings",
        lambda path=None: saved,
    )
    monkeypatch.setattr(
        "src.run_test_100.DEFAULT_QUESTION_DPR_EMBEDDINGS_PATH",
        type("P", (), {"exists": lambda self: True})(),
    )

    embeddings = get_precomputed_dense_query_embeddings([question], include_category=False)

    assert len(embeddings) == 1
    assert embeddings[0].tolist() == [1.0, 2.0]


def test_search_questions_passes_precomputed_dense_embeddings_to_weighted_batch(monkeypatch):
    question = JeopardyQuestion(category="Science", clue="Raw clue", answer="Beta")
    captured = {}

    monkeypatch.setattr(
        "src.run_test_100.get_precomputed_dense_query_embeddings",
        lambda questions, include_category: [np.array([9.0, 8.0], dtype=np.float32)],
    )

    def fake_run_search_batch(
        queries,
        categories,
        limit,
        mode,
        weighted=False,
        weight_equal=False,
        weighted_cole=False,
        skip_redirects=False,
        dense_query_embeddings=None,
        natural_questions_embeddings=None,
        natural_questions_concat_embeddings=None,
        search_progress_every=0,
    ):
        captured["dense_query_embeddings"] = dense_query_embeddings
        captured["natural_questions_embeddings"] = natural_questions_embeddings
        captured["natural_questions_concat_embeddings"] = natural_questions_concat_embeddings
        captured["search_progress_every"] = search_progress_every
        return [{"query": queries[0], "results": [make_hit("Beta", article_index=2)]}]

    monkeypatch.setattr("src.run_test_100.run_search_batch", fake_run_search_batch)
    monkeypatch.setattr(
        "src.run_test_100.get_precomputed_natural_question_dense_embeddings",
        lambda questions: (
            [np.array([[7.0, 6.0], [5.0, 4.0]], dtype=np.float32)],
            [np.array([3.0, 2.0], dtype=np.float32)],
        ),
    )

    searches = search_questions(
        [question],
        limit=5,
        mode="whoosh_title_body",
        query_mode="full",
        include_category=False,
        weighted=True,
        paraphrase=False,
    )

    assert len(captured["dense_query_embeddings"]) == 1
    assert captured["dense_query_embeddings"][0].tolist() == [9.0, 8.0]
    assert len(captured["natural_questions_embeddings"]) == 1
    assert captured["natural_questions_embeddings"][0].tolist() == [[7.0, 6.0], [5.0, 4.0]]
    assert len(captured["natural_questions_concat_embeddings"]) == 1
    assert captured["natural_questions_concat_embeddings"][0].tolist() == [3.0, 2.0]
    assert captured["search_progress_every"] == 0
    assert [result["title"] for result in searches[0]["results"]] == ["Beta"]


def test_evaluate_questions_prints_question_results_only_when_enabled(monkeypatch, capsys):
    question = JeopardyQuestion(category="Science", clue="Raw clue", answer="Beta")

    monkeypatch.setattr(
        "src.run_test_100.search_questions",
        lambda *args, **kwargs: [
            {
                "query": "Raw clue",
                "results": [make_hit("Beta", article_index=2), make_hit("Gamma", article_index=3)],
            }
        ],
    )

    evaluate_questions([question], print_question_results=False, top_k_values=[1, 5])
    output_without_flag = capsys.readouterr().out
    assert "Question results grouped by first Top-k:" not in output_without_flag

    evaluate_questions([question], print_question_results=True, top_k_values=[1, 5])
    output_with_flag = capsys.readouterr().out
    assert "Question results grouped by first Top-k:" in output_with_flag
    assert "Top-1:" in output_with_flag
    assert "Clue: Raw clue" in output_with_flag
