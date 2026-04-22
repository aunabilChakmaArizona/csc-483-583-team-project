from src.retrieval_cache import get_cache_stats, get_or_compute_component_results, reset_cache_stats


def test_component_cache_reuses_results_for_identical_retrieval(tmp_path):
    reset_cache_stats()
    cache_db_path = tmp_path / "retrieval_component_cache.sqlite3"
    index_file = tmp_path / "index.faiss"
    index_file.write_text("index", encoding="utf-8")
    calls = {"count": 0}

    def compute_results():
        calls["count"] += 1
        return [{"title": "Alpha", "score": 1.0}]

    first = get_or_compute_component_results(
        component_name="faiss_title_first_sentence",
        query_text="example clue",
        query_category=None,
        component_limit=10000,
        index_paths=[index_file],
        extra_config={"variant": "title_first_sentence"},
        compute_results=compute_results,
        cache_db_path=cache_db_path,
    )
    second = get_or_compute_component_results(
        component_name="faiss_title_first_sentence",
        query_text="example clue",
        query_category=None,
        component_limit=10000,
        index_paths=[index_file],
        extra_config={"variant": "title_first_sentence"},
        compute_results=compute_results,
        cache_db_path=cache_db_path,
    )

    assert first == [{"title": "Alpha", "score": 1.0}]
    assert second == first
    assert calls["count"] == 1
    assert get_cache_stats() == {"hits": 1, "misses": 1}


def test_component_cache_misses_when_retrieval_relevant_input_changes(tmp_path):
    reset_cache_stats()
    cache_db_path = tmp_path / "retrieval_component_cache.sqlite3"
    index_file = tmp_path / "index.faiss"
    index_file.write_text("index", encoding="utf-8")
    calls = {"count": 0}

    def compute_results():
        calls["count"] += 1
        return [{"title": f"Alpha-{calls['count']}", "score": 1.0}]

    first = get_or_compute_component_results(
        component_name="faiss_title_first_sentence",
        query_text="example clue",
        query_category=None,
        component_limit=10000,
        index_paths=[index_file],
        extra_config={"variant": "title_first_sentence"},
        compute_results=compute_results,
        query_embedding_digest="digest-a",
        cache_db_path=cache_db_path,
    )
    second = get_or_compute_component_results(
        component_name="faiss_title_first_sentence",
        query_text="example clue",
        query_category=None,
        component_limit=10000,
        index_paths=[index_file],
        extra_config={"variant": "title_first_sentence"},
        compute_results=compute_results,
        query_embedding_digest="digest-b",
        cache_db_path=cache_db_path,
    )

    assert first == [{"title": "Alpha-1", "score": 1.0}]
    assert second == [{"title": "Alpha-2", "score": 1.0}]
    assert calls["count"] == 2
    assert get_cache_stats() == {"hits": 0, "misses": 2}
