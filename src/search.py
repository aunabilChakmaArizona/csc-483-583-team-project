"""Search entry points."""

from functools import lru_cache
from pathlib import Path

from whoosh import query as whoosh_query
from whoosh.scoring import BM25F

try:
    from src.processor4_index import DEFAULT_INDEX_DIR, open_index
    from src.processor3_tokenize import load_stop_words, tokenize_body
except ModuleNotFoundError:
    from processor4_index import DEFAULT_INDEX_DIR, open_index
    from processor3_tokenize import load_stop_words, tokenize_body


@lru_cache(maxsize=1)
def get_stop_words() -> frozenset[str]:
    return frozenset(load_stop_words())


def normalize_query_terms(query_text: str) -> list[str]:
    return tokenize_body(query_text, set(get_stop_words()))


def build_terms_query(terms: list[str]):
    if not terms:
        return None

    return whoosh_query.Or([whoosh_query.Term("body", term) for term in terms])


def serialize_results(results) -> list[dict]:
    return [
        {
            "title": hit["title"],
            "body": hit["body"],
            "source_file": hit["source_file"],
            "article_index": hit["article_index"],
            "is_redirect": hit["is_redirect"],
            "score": hit.score,
        }
        for hit in results
    ]


def search_with_searcher(searcher, query_text: str, limit: int = 10) -> list[dict]:
    terms = normalize_query_terms(query_text)
    compiled_query = build_terms_query(terms)

    if compiled_query is None:
        return []

    results = searcher.search(compiled_query, limit=limit)
    return serialize_results(results)


def search(query: str, limit: int = 10, index_dir: Path = DEFAULT_INDEX_DIR) -> list[dict]:
    """Return the top matching documents for a space-separated query."""
    index = open_index(index_dir)
    with index.searcher(weighting=BM25F()) as searcher:
        return search_with_searcher(searcher, query, limit=limit)


def multi_search(
    queries: list[str],
    limit: int = 10,
    index_dir: Path = DEFAULT_INDEX_DIR,
    progress_every: int = 10,
) -> list[dict]:
    """Return search results for multiple queries while loading the index once."""
    index = open_index(index_dir)
    total = len(queries)
    all_results = []

    with index.searcher(weighting=BM25F()) as searcher:
        for index_number, query in enumerate(queries, start=1):
            all_results.append(
                {
                    "query": query,
                    "results": search_with_searcher(searcher, query, limit=limit),
                }
            )

            if progress_every and index_number % progress_every == 0:
                print(f"[multi_search] Queries: {index_number}/{total}")

    print(f"[multi_search] Finished queries: {total}/{total}")
    return all_results
