from src.search import search


def test_search_returns_list():
    assert search("example query") == []
