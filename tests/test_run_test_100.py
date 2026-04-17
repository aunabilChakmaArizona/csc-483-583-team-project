from src.run_test_100 import (
    JeopardyQuestion,
    merge_ranked_results,
    question_queries,
    search_questions,
)


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

    def fake_run_search_batch(queries, limit, mode):
        captured["queries"] = queries
        captured["limit"] = limit
        captured["mode"] = mode
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
        "limit": 1000,
        "mode": "whoosh",
    }
    assert searches[0]["queries"] == ["Raw clue", "Alt one", "Alt two", "Alt three", "Alt four"]
    assert [result["title"] for result in searches[0]["results"]] == ["Beta", "Gamma", "Alpha"]
