"""Search entry points."""

from functools import lru_cache
from pathlib import Path
import sqlite3

from whoosh import query as whoosh_query
from whoosh.qparser import MultifieldParser, OrGroup, QueryParser
from whoosh.scoring import BM25F

try:
    from src.processor_redirect import DEFAULT_OUTPUT_DB_PATH as DEFAULT_REDIRECT_DB_PATH
    from src.processor4_cole_index import DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_COLE_INDEX_DIR
    from src.processor4_cole_index import open_index as open_whoosh_cole_index
    from src.processor4_whoosh_title_body_index import (
        DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_TITLE_BODY_INDEX_DIR,
    )
    from src.processor4_whoosh_title_body_index import open_index as open_whoosh_title_body_index
    from src.processor4_whoosh_index import DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_INDEX_DIR
    from src.processor4_whoosh_index import open_index as open_whoosh_index
    from src.processor4_index import DEFAULT_INDEX_DIR, open_index
    from src.processor3_tokenize import load_stop_words, tokenize_body
except ModuleNotFoundError:
    from processor_redirect import DEFAULT_OUTPUT_DB_PATH as DEFAULT_REDIRECT_DB_PATH
    from processor4_cole_index import DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_COLE_INDEX_DIR
    from processor4_cole_index import open_index as open_whoosh_cole_index
    from processor4_whoosh_title_body_index import (
        DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_TITLE_BODY_INDEX_DIR,
    )
    from processor4_whoosh_title_body_index import open_index as open_whoosh_title_body_index
    from processor4_whoosh_index import DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_INDEX_DIR
    from processor4_whoosh_index import open_index as open_whoosh_index
    from processor4_index import DEFAULT_INDEX_DIR, open_index
    from processor3_tokenize import load_stop_words, tokenize_body

# top5-40%
# WEIGHTED_TITLE_WEIGHT = 2.0
# WEIGHTED_REDIRECT_WEIGHT = 2.0 or 0.0 or 8.0
# WEIGHTED_BODY_WEIGHT = 4.0

WEIGHTED_TITLE_WEIGHT = 0.0
WEIGHTED_REDIRECT_WEIGHT = 8.0
WEIGHTED_BODY_WEIGHT = 4.0

EQUAL_TITLE_WEIGHT = 1.0
EQUAL_REDIRECT_WEIGHT = 1.0
EQUAL_BODY_WEIGHT = 1.0
WEIGHTED_COMPONENT_LIMIT = 1000


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
            "body": hit["body"] if "body" in hit else "",
            "source_file": hit["source_file"],
            "article_index": hit["article_index"],
            "is_redirect": hit["is_redirect"],
            "score": hit.score,
        }
        for hit in results
    ]


@lru_cache(maxsize=4)
def load_redirect_lookup(
    redirect_db_path: Path = DEFAULT_REDIRECT_DB_PATH,
) -> dict[tuple[str, int], dict]:
    with sqlite3.connect(redirect_db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                redirect_source_file,
                redirect_article_index,
                resolved_title,
                resolved_source_file,
                resolved_article_index,
                resolution_status
            FROM redirects
            """
        ).fetchall()

    return {
        (redirect_source_file, redirect_article_index): {
            "resolved_title": resolved_title,
            "resolved_source_file": resolved_source_file,
            "resolved_article_index": resolved_article_index,
            "resolution_status": resolution_status,
        }
        for (
            redirect_source_file,
            redirect_article_index,
            resolved_title,
            resolved_source_file,
            resolved_article_index,
            resolution_status,
        ) in rows
    }


def search_with_searcher(
    searcher,
    query_text: str,
    limit: int = 10,
    skip_redirects: bool = False,
) -> list[dict]:
    terms = normalize_query_terms(query_text)
    compiled_query = build_terms_query(terms)

    if compiled_query is None:
        return []

    result_filter = whoosh_query.NumericRange("is_redirect", 0, 0) if skip_redirects else None
    results = searcher.search(compiled_query, limit=limit, filter=result_filter)
    return serialize_results(results)


def search(
    query: str,
    limit: int = 10,
    index_dir: Path = DEFAULT_INDEX_DIR,
    skip_redirects: bool = False,
) -> list[dict]:
    """Return the top matching documents for a space-separated query."""
    index = open_index(index_dir)
    with index.searcher(weighting=BM25F()) as searcher:
        return search_with_searcher(searcher, query, limit=limit, skip_redirects=skip_redirects)


def multi_search(
    queries: list[str],
    limit: int = 10,
    index_dir: Path = DEFAULT_INDEX_DIR,
    progress_every: int = 10,
    skip_redirects: bool = False,
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
                    "results": search_with_searcher(
                        searcher,
                        query,
                        limit=limit,
                        skip_redirects=skip_redirects,
                    ),
                }
            )

            if progress_every and index_number % progress_every == 0:
                print(f"[multi_search] Queries: {index_number}/{total}")

    print(f"[multi_search] Finished queries: {total}/{total}")
    return all_results


def fetch_document_by_key(searcher, source_file: str, article_index: int) -> dict | None:
    lookup_query = whoosh_query.And(
        [
            whoosh_query.Term("source_file", source_file),
            whoosh_query.NumericRange("article_index", article_index, article_index),
        ]
    )
    results = searcher.search(lookup_query, limit=1)
    if not results:
        return None

    return serialize_results(results)[0]


def build_stored_document_lookup(searcher) -> dict[tuple[str, int], dict]:
    lookup = {}

    for docnum in range(searcher.doc_count_all()):
        stored_fields = searcher.stored_fields(docnum)
        if not stored_fields:
            continue

        key = (stored_fields["source_file"], stored_fields["article_index"])
        lookup[key] = {
            "title": stored_fields["title"],
            "body": stored_fields.get("body", ""),
            "source_file": stored_fields["source_file"],
            "article_index": stored_fields["article_index"],
            "is_redirect": stored_fields["is_redirect"],
            "score": 0.0,
        }

    return lookup


def weighted_result_key(result: dict) -> tuple[str, int]:
    return (result["source_file"], result["article_index"])


def search_whoosh_weighted_with_searcher(
    searcher,
    query_text: str,
    title_parser,
    body_parser,
    redirect_lookup: dict[tuple[str, int], dict],
    limit: int = 10,
    component_limit: int = WEIGHTED_COMPONENT_LIMIT,
    skip_redirects: bool = False,
    title_weight: float = WEIGHTED_TITLE_WEIGHT,
    redirect_weight: float = WEIGHTED_REDIRECT_WEIGHT,
    body_weight: float = WEIGHTED_BODY_WEIGHT,
    redirect_searcher=None,
    redirect_title_parser=None,
    filter_main_redirects: bool = True,
    document_lookup: dict[tuple[str, int], dict] | None = None,
) -> list[dict]:
    if not query_text.strip():
        return []

    redirect_searcher = redirect_searcher or searcher
    redirect_title_parser = redirect_title_parser or title_parser
    title_query = title_parser.parse(query_text)
    body_query = body_parser.parse(query_text)
    redirect_title_query = redirect_title_parser.parse(query_text)
    non_redirect_filter = (
        whoosh_query.NumericRange("is_redirect", 0, 0) if filter_main_redirects else None
    )
    redirect_filter = whoosh_query.NumericRange("is_redirect", 1, 1)

    title_results = serialize_results(
        searcher.search(title_query, limit=component_limit, filter=non_redirect_filter)
    )
    body_results = serialize_results(
        searcher.search(body_query, limit=component_limit, filter=non_redirect_filter)
    )
    redirect_results = []
    if not skip_redirects:
        redirect_results = serialize_results(
            redirect_searcher.search(
                redirect_title_query,
                limit=component_limit,
                filter=redirect_filter,
            )
        )

    aggregated_results: dict[tuple[str, int], dict] = {}
    document_cache: dict[tuple[str, int], dict | None] = {}

    def ensure_entry(result: dict) -> dict:
        key = weighted_result_key(result)
        entry = aggregated_results.get(key)
        if entry is None:
            entry = {
                **result,
                "title_score": 0.0,
                "redirect_score": 0.0,
                "body_score": 0.0,
                "score": 0.0,
            }
            aggregated_results[key] = entry
        return entry

    for result in title_results:
        ensure_entry(result)["title_score"] = max(ensure_entry(result)["title_score"], result["score"])

    for result in body_results:
        ensure_entry(result)["body_score"] = max(ensure_entry(result)["body_score"], result["score"])

    for redirect_result in redirect_results:
        redirect_mapping = redirect_lookup.get(weighted_result_key(redirect_result))
        if redirect_mapping is None or redirect_mapping["resolution_status"] != "resolved":
            continue

        resolved_key = (
            redirect_mapping["resolved_source_file"],
            redirect_mapping["resolved_article_index"],
        )

        canonical_result = aggregated_results.get(resolved_key)
        if canonical_result is None:
            if document_lookup is not None:
                canonical_result = document_lookup.get(resolved_key)
            else:
                if resolved_key not in document_cache:
                    document_cache[resolved_key] = fetch_document_by_key(
                        searcher,
                        source_file=redirect_mapping["resolved_source_file"],
                        article_index=redirect_mapping["resolved_article_index"],
                    )
                canonical_result = document_cache[resolved_key]
            if canonical_result is None or canonical_result["is_redirect"]:
                continue

        ensure_entry(canonical_result)["redirect_score"] = max(
            ensure_entry(canonical_result)["redirect_score"],
            redirect_result["score"],
        )

    for result in aggregated_results.values():
        result["score"] = (
            title_weight * result["title_score"]
            + redirect_weight * result["redirect_score"]
            + body_weight * result["body_score"]
        )

    ranked_results = sorted(
        aggregated_results.values(),
        key=lambda result: (
            -result["score"],
            -result["title_score"],
            -result["redirect_score"],
            -result["body_score"],
            result["article_index"],
        ),
    )
    return ranked_results[:limit]


def search_whoosh_weighted(
    query: str,
    limit: int = 10,
    component_limit: int = WEIGHTED_COMPONENT_LIMIT,
    index_dir: Path = DEFAULT_WHOOSH_TITLE_BODY_INDEX_DIR,
    redirect_db_path: Path = DEFAULT_REDIRECT_DB_PATH,
    skip_redirects: bool = False,
    title_weight: float = WEIGHTED_TITLE_WEIGHT,
    redirect_weight: float = WEIGHTED_REDIRECT_WEIGHT,
    body_weight: float = WEIGHTED_BODY_WEIGHT,
) -> list[dict]:
    index = open_whoosh_title_body_index(index_dir)
    redirect_lookup = load_redirect_lookup(redirect_db_path)

    with index.searcher(weighting=BM25F()) as searcher:
        title_parser = QueryParser("title", schema=index.schema, group=OrGroup)
        body_parser = QueryParser("body", schema=index.schema, group=OrGroup)
        return search_whoosh_weighted_with_searcher(
            searcher,
            query_text=query,
            title_parser=title_parser,
            body_parser=body_parser,
            redirect_lookup=redirect_lookup,
            limit=limit,
            component_limit=component_limit,
            skip_redirects=skip_redirects,
            title_weight=title_weight,
            redirect_weight=redirect_weight,
            body_weight=body_weight,
            filter_main_redirects=True,
        )


def search_whoosh_weighted_cole(
    query: str,
    limit: int = 10,
    component_limit: int = WEIGHTED_COMPONENT_LIMIT,
    index_dir: Path = DEFAULT_WHOOSH_COLE_INDEX_DIR,
    redirect_index_dir: Path = DEFAULT_WHOOSH_TITLE_BODY_INDEX_DIR,
    redirect_db_path: Path = DEFAULT_REDIRECT_DB_PATH,
    skip_redirects: bool = False,
    title_weight: float = WEIGHTED_TITLE_WEIGHT,
    redirect_weight: float = WEIGHTED_REDIRECT_WEIGHT,
    body_weight: float = WEIGHTED_BODY_WEIGHT,
) -> list[dict]:
    index = open_whoosh_cole_index(index_dir)
    redirect_index = open_whoosh_title_body_index(redirect_index_dir)
    redirect_lookup = load_redirect_lookup(redirect_db_path)

    with index.searcher(weighting=BM25F()) as searcher:
        with redirect_index.searcher(weighting=BM25F()) as redirect_searcher:
            document_lookup = build_stored_document_lookup(searcher)
            title_parser = QueryParser("title", schema=index.schema, group=OrGroup)
            body_parser = QueryParser("body", schema=index.schema, group=OrGroup)
            redirect_title_parser = QueryParser("title", schema=redirect_index.schema, group=OrGroup)
            return search_whoosh_weighted_with_searcher(
                searcher,
                query_text=query,
                title_parser=title_parser,
                body_parser=body_parser,
                redirect_lookup=redirect_lookup,
                limit=limit,
                component_limit=component_limit,
                skip_redirects=skip_redirects,
                title_weight=title_weight,
                redirect_weight=redirect_weight,
                body_weight=body_weight,
                redirect_searcher=redirect_searcher,
                redirect_title_parser=redirect_title_parser,
                filter_main_redirects=False,
                document_lookup=document_lookup,
            )


def multi_search_whoosh_weighted(
    queries: list[str],
    limit: int = 10,
    component_limit: int = WEIGHTED_COMPONENT_LIMIT,
    index_dir: Path = DEFAULT_WHOOSH_TITLE_BODY_INDEX_DIR,
    redirect_db_path: Path = DEFAULT_REDIRECT_DB_PATH,
    progress_every: int = 10,
    skip_redirects: bool = False,
    title_weight: float = WEIGHTED_TITLE_WEIGHT,
    redirect_weight: float = WEIGHTED_REDIRECT_WEIGHT,
    body_weight: float = WEIGHTED_BODY_WEIGHT,
) -> list[dict]:
    index = open_whoosh_title_body_index(index_dir)
    redirect_lookup = load_redirect_lookup(redirect_db_path)
    total = len(queries)
    all_results = []

    with index.searcher(weighting=BM25F()) as searcher:
        title_parser = QueryParser("title", schema=index.schema, group=OrGroup)
        body_parser = QueryParser("body", schema=index.schema, group=OrGroup)

        for index_number, query_text in enumerate(queries, start=1):
            all_results.append(
                {
                    "query": query_text,
                    "results": search_whoosh_weighted_with_searcher(
                        searcher,
                        query_text=query_text,
                        title_parser=title_parser,
                        body_parser=body_parser,
                        redirect_lookup=redirect_lookup,
                        limit=limit,
                        component_limit=component_limit,
                        skip_redirects=skip_redirects,
                        title_weight=title_weight,
                        redirect_weight=redirect_weight,
                        body_weight=body_weight,
                        filter_main_redirects=True,
                    ),
                }
            )

            if progress_every and index_number % progress_every == 0:
                print(f"[multi_search_whoosh_weighted] Queries: {index_number}/{total}")

    print(f"[multi_search_whoosh_weighted] Finished queries: {total}/{total}")
    return all_results


def multi_search_whoosh_weighted_cole(
    queries: list[str],
    limit: int = 10,
    component_limit: int = WEIGHTED_COMPONENT_LIMIT,
    index_dir: Path = DEFAULT_WHOOSH_COLE_INDEX_DIR,
    redirect_index_dir: Path = DEFAULT_WHOOSH_TITLE_BODY_INDEX_DIR,
    redirect_db_path: Path = DEFAULT_REDIRECT_DB_PATH,
    progress_every: int = 10,
    skip_redirects: bool = False,
    title_weight: float = WEIGHTED_TITLE_WEIGHT,
    redirect_weight: float = WEIGHTED_REDIRECT_WEIGHT,
    body_weight: float = WEIGHTED_BODY_WEIGHT,
) -> list[dict]:
    index = open_whoosh_cole_index(index_dir)
    redirect_index = open_whoosh_title_body_index(redirect_index_dir)
    redirect_lookup = load_redirect_lookup(redirect_db_path)
    total = len(queries)
    all_results = []

    with index.searcher(weighting=BM25F()) as searcher:
        with redirect_index.searcher(weighting=BM25F()) as redirect_searcher:
            document_lookup = build_stored_document_lookup(searcher)
            title_parser = QueryParser("title", schema=index.schema, group=OrGroup)
            body_parser = QueryParser("body", schema=index.schema, group=OrGroup)
            redirect_title_parser = QueryParser(
                "title",
                schema=redirect_index.schema,
                group=OrGroup,
            )

            for index_number, query_text in enumerate(queries, start=1):
                all_results.append(
                    {
                        "query": query_text,
                        "results": search_whoosh_weighted_with_searcher(
                            searcher,
                            query_text=query_text,
                            title_parser=title_parser,
                            body_parser=body_parser,
                            redirect_lookup=redirect_lookup,
                            limit=limit,
                            component_limit=component_limit,
                            skip_redirects=skip_redirects,
                            title_weight=title_weight,
                            redirect_weight=redirect_weight,
                            body_weight=body_weight,
                            redirect_searcher=redirect_searcher,
                            redirect_title_parser=redirect_title_parser,
                            filter_main_redirects=False,
                            document_lookup=document_lookup,
                        ),
                    }
                )

                if progress_every and index_number % progress_every == 0:
                    print(f"[multi_search_whoosh_weighted_cole] Queries: {index_number}/{total}")

    print(f"[multi_search_whoosh_weighted_cole] Finished queries: {total}/{total}")
    return all_results


def search_whoosh_default(
    query: str,
    limit: int = 10,
    index_dir: Path = DEFAULT_WHOOSH_INDEX_DIR,
    skip_redirects: bool = False,
) -> list[dict]:
    """Search the index built from cleaned text using Whoosh's default analyzer."""
    index = open_whoosh_index(index_dir)
    with index.searcher(weighting=BM25F()) as searcher:
        parsed_query = QueryParser("body", schema=index.schema).parse(query)
        result_filter = whoosh_query.NumericRange("is_redirect", 0, 0) if skip_redirects else None
        results = searcher.search(parsed_query, limit=limit, filter=result_filter)
        return serialize_results(results)


def multi_search_whoosh_default(
    queries: list[str],
    limit: int = 10,
    index_dir: Path = DEFAULT_WHOOSH_INDEX_DIR,
    progress_every: int = 10,
    skip_redirects: bool = False,
) -> list[dict]:
    """Search multiple queries against the Whoosh-default index with one index open."""
    index = open_whoosh_index(index_dir)
    total = len(queries)
    all_results = []

    with index.searcher(weighting=BM25F()) as searcher:
        parser = QueryParser("body", schema=index.schema)

        for index_number, query_text in enumerate(queries, start=1):
            parsed_query = parser.parse(query_text)
            result_filter = whoosh_query.NumericRange("is_redirect", 0, 0) if skip_redirects else None
            results = searcher.search(parsed_query, limit=limit, filter=result_filter)
            all_results.append(
                {
                    "query": query_text,
                    "results": serialize_results(results),
                }
            )

            if progress_every and index_number % progress_every == 0:
                print(f"[multi_search_whoosh] Queries: {index_number}/{total}")

    print(f"[multi_search_whoosh] Finished queries: {total}/{total}")
    return all_results


def search_whoosh_title_body(
    query: str,
    limit: int = 10,
    index_dir: Path = DEFAULT_WHOOSH_TITLE_BODY_INDEX_DIR,
    skip_redirects: bool = False,
) -> list[dict]:
    """Search title and body fields with equal field weight."""
    index = open_whoosh_title_body_index(index_dir)
    with index.searcher(weighting=BM25F()) as searcher:
        parser = MultifieldParser(["title", "body"], schema=index.schema, group=OrGroup)
        parsed_query = parser.parse(query)
        result_filter = whoosh_query.NumericRange("is_redirect", 0, 0) if skip_redirects else None
        results = searcher.search(parsed_query, limit=limit, filter=result_filter)
        return serialize_results(results)


def multi_search_whoosh_title_body(
    queries: list[str],
    limit: int = 10,
    index_dir: Path = DEFAULT_WHOOSH_TITLE_BODY_INDEX_DIR,
    progress_every: int = 10,
    skip_redirects: bool = False,
) -> list[dict]:
    """Search multiple queries against title and body fields with one index open."""
    index = open_whoosh_title_body_index(index_dir)
    total = len(queries)
    all_results = []

    with index.searcher(weighting=BM25F()) as searcher:
        parser = MultifieldParser(["title", "body"], schema=index.schema, group=OrGroup)

        for index_number, query_text in enumerate(queries, start=1):
            parsed_query = parser.parse(query_text)
            result_filter = whoosh_query.NumericRange("is_redirect", 0, 0) if skip_redirects else None
            results = searcher.search(parsed_query, limit=limit, filter=result_filter)
            all_results.append(
                {
                    "query": query_text,
                    "results": serialize_results(results),
                }
            )

            if progress_every and index_number % progress_every == 0:
                print(f"[multi_search_whoosh_title_body] Queries: {index_number}/{total}")

    print(f"[multi_search_whoosh_title_body] Finished queries: {total}/{total}")
    return all_results


def search_whoosh_cole(
    query: str,
    limit: int = 10,
    index_dir: Path = DEFAULT_WHOOSH_COLE_INDEX_DIR,
    skip_redirects: bool = False,
) -> list[dict]:
    """Search Cole's Whoosh index across title and body fields."""
    del skip_redirects
    index = open_whoosh_cole_index(index_dir)
    with index.searcher(weighting=BM25F()) as searcher:
        parser = MultifieldParser(["title", "body"], schema=index.schema, group=OrGroup)
        parsed_query = parser.parse(query)
        results = searcher.search(parsed_query, limit=limit)
        return serialize_results(results)


def multi_search_whoosh_cole(
    queries: list[str],
    limit: int = 10,
    index_dir: Path = DEFAULT_WHOOSH_COLE_INDEX_DIR,
    progress_every: int = 10,
    skip_redirects: bool = False,
) -> list[dict]:
    """Search multiple queries against Cole's Whoosh index with one index open."""
    del skip_redirects
    index = open_whoosh_cole_index(index_dir)
    total = len(queries)
    all_results = []

    with index.searcher(weighting=BM25F()) as searcher:
        parser = MultifieldParser(["title", "body"], schema=index.schema, group=OrGroup)

        for index_number, query_text in enumerate(queries, start=1):
            parsed_query = parser.parse(query_text)
            results = searcher.search(parsed_query, limit=limit)
            all_results.append(
                {
                    "query": query_text,
                    "results": serialize_results(results),
                }
            )

            if progress_every and index_number % progress_every == 0:
                print(f"[multi_search_whoosh_cole] Queries: {index_number}/{total}")

    print(f"[multi_search_whoosh_cole] Finished queries: {total}/{total}")
    return all_results
