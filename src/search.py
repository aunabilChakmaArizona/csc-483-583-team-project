"""Search entry points."""

from functools import lru_cache
import json
from pathlib import Path
import re
import sqlite3

from whoosh import query as whoosh_query
from whoosh.qparser import MultifieldParser, OrGroup, QueryParser
from whoosh.scoring import BM25F

try:
    from src.processor_redirect import DEFAULT_OUTPUT_DB_PATH as DEFAULT_REDIRECT_DB_PATH
    from src.processor4_cole_first_paragraph_index import (
        DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_COLE_FIRST_PARAGRAPH_INDEX_DIR,
    )
    from src.processor4_cole_first_paragraph_index import (
        open_index as open_whoosh_cole_first_paragraph_index,
    )
    from src.processor4_cole_first_sentence_index import (
        DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_COLE_FIRST_SENTENCE_INDEX_DIR,
    )
    from src.processor4_cole_first_sentence_index import (
        open_index as open_whoosh_cole_first_sentence_index,
    )
    from src.processor4_cole_first_two_sentences_index import (
        DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_COLE_FIRST_TWO_SENTENCES_INDEX_DIR,
    )
    from src.processor4_cole_first_two_sentences_index import (
        open_index as open_whoosh_cole_first_two_sentences_index,
    )
    from src.processor4_cole_index import DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_COLE_INDEX_DIR
    from src.processor4_cole_index import open_index as open_whoosh_cole_index
    from src.processor4_cole_redirect_index import (
        DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_COLE_REDIRECT_INDEX_DIR,
    )
    from src.processor4_cole_redirect_index import open_index as open_whoosh_cole_redirect_index
    from src.processor4_whoosh_title_body_index import (
        DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_TITLE_BODY_INDEX_DIR,
    )
    from src.processor4_whoosh_title_body_index import open_index as open_whoosh_title_body_index
    from src.processor4_whoosh_title_body_category_index import (
        DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_TITLE_BODY_CATEGORY_INDEX_DIR,
    )
    from src.processor4_whoosh_title_body_category_index import (
        open_index as open_whoosh_title_body_category_index,
    )
    from src.processor4_whoosh_index import DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_INDEX_DIR
    from src.processor4_whoosh_index import open_index as open_whoosh_index
    from src.processor4_index import DEFAULT_INDEX_DIR, open_index
    from src.processor3_tokenize import load_stop_words, tokenize_body
    from src.retrieval_cache import (
        digest_query_embedding,
        get_or_compute_component_results,
    )
except ModuleNotFoundError:
    from processor_redirect import DEFAULT_OUTPUT_DB_PATH as DEFAULT_REDIRECT_DB_PATH
    from processor4_cole_first_paragraph_index import (
        DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_COLE_FIRST_PARAGRAPH_INDEX_DIR,
    )
    from processor4_cole_first_paragraph_index import (
        open_index as open_whoosh_cole_first_paragraph_index,
    )
    from processor4_cole_first_sentence_index import (
        DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_COLE_FIRST_SENTENCE_INDEX_DIR,
    )
    from processor4_cole_first_sentence_index import (
        open_index as open_whoosh_cole_first_sentence_index,
    )
    from processor4_cole_first_two_sentences_index import (
        DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_COLE_FIRST_TWO_SENTENCES_INDEX_DIR,
    )
    from processor4_cole_first_two_sentences_index import (
        open_index as open_whoosh_cole_first_two_sentences_index,
    )
    from processor4_cole_index import DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_COLE_INDEX_DIR
    from processor4_cole_index import open_index as open_whoosh_cole_index
    from processor4_cole_redirect_index import (
        DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_COLE_REDIRECT_INDEX_DIR,
    )
    from processor4_cole_redirect_index import open_index as open_whoosh_cole_redirect_index
    from processor4_whoosh_title_body_index import (
        DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_TITLE_BODY_INDEX_DIR,
    )
    from processor4_whoosh_title_body_index import open_index as open_whoosh_title_body_index
    from processor4_whoosh_title_body_category_index import (
        DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_TITLE_BODY_CATEGORY_INDEX_DIR,
    )
    from processor4_whoosh_title_body_category_index import (
        open_index as open_whoosh_title_body_category_index,
    )
    from processor4_whoosh_index import DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_INDEX_DIR
    from processor4_whoosh_index import open_index as open_whoosh_index
    from processor4_index import DEFAULT_INDEX_DIR, open_index
    from processor3_tokenize import load_stop_words, tokenize_body
    from retrieval_cache import (
        digest_query_embedding,
        get_or_compute_component_results,
    )

#################################### ** Most Important ** Weight tuning

# Best 1 so far
# WEIGHTED_TITLE_WEIGHT = 0.0
# WEIGHTED_REDIRECT_WEIGHT = 0.0 # so far it has no impact
# WEIGHTED_BODY_WEIGHT = 15.0
# WEIGHTED_FIRST_SENTENCE_WEIGHT = 0.0
# WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT = 0.0
# WEIGHTED_FIRST_PARAGRAPH_WEIGHT = 0.0
# WEIGHTED_FAISS_WEIGHT = 1.0
# WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT = 1.0
# WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT = 1.0
# WEIGHTED_YEAR_MATCH_WEIGHT = 0.0
# WEIGHTED_QUOTE_MATCH_WEIGHT = 0.0
# WEIGHTED_CATEGORY_FIRST_SENTENCE_WEIGHT = 0.0
# WEIGHTED_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT = 0.0

# Best 2 so far
# WEIGHTED_TITLE_WEIGHT = 0.0
# WEIGHTED_REDIRECT_WEIGHT = 0.0 # so far it has no impact
# WEIGHTED_BODY_WEIGHT = 5.0
# WEIGHTED_FIRST_SENTENCE_WEIGHT = 0.0
# WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT = 0.0
# WEIGHTED_FIRST_PARAGRAPH_WEIGHT = 0.0
# WEIGHTED_FAISS_WEIGHT = 0.0
# WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT = 1.0
# WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT = 0.0
# WEIGHTED_YEAR_MATCH_WEIGHT = 0.0
# WEIGHTED_QUOTE_MATCH_WEIGHT = 0.0
# WEIGHTED_CATEGORY_FIRST_SENTENCE_WEIGHT = 0.0
# WEIGHTED_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT = 0.0

## Tune weights below

WEIGHTED_TITLE_WEIGHT = 0.0
WEIGHTED_REDIRECT_WEIGHT = 0.0 # so far it has no impact
WEIGHTED_BODY_WEIGHT = 20.0
WEIGHTED_FIRST_SENTENCE_WEIGHT = 0.0
WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT = 0.0
WEIGHTED_FIRST_PARAGRAPH_WEIGHT = 0.0
WEIGHTED_FAISS_WEIGHT = 1.0
WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT = 1.0
WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT = 1.0
WEIGHTED_YEAR_MATCH_WEIGHT = 1.0
WEIGHTED_QUOTE_MATCH_WEIGHT = 0.0 #todo: it will be best to make it a ngram match
WEIGHTED_CATEGORY_FIRST_SENTENCE_WEIGHT = 0.0
WEIGHTED_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT = 0.0

#################################### 

EQUAL_TITLE_WEIGHT = 1.0
EQUAL_REDIRECT_WEIGHT = 1.0
EQUAL_BODY_WEIGHT = 1.0
EQUAL_FIRST_SENTENCE_WEIGHT = 1.0
EQUAL_FIRST_TWO_SENTENCES_WEIGHT = 1.0
EQUAL_FIRST_PARAGRAPH_WEIGHT = 1.0
EQUAL_FAISS_WEIGHT = 1.0
EQUAL_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT = 1.0
EQUAL_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT = 1.0
EQUAL_YEAR_MATCH_WEIGHT = 1.0
EQUAL_QUOTE_MATCH_WEIGHT = 1.0
EQUAL_CATEGORY_FIRST_SENTENCE_WEIGHT = 1.0
EQUAL_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT = 1.0
WEIGHTED_COMPONENT_LIMIT = 10000
YEAR_RE = re.compile(r"\b(?:1\d{3}|20\d{2})\b")
QUOTE_RE = re.compile(r'["“”]([^"“”]+)["“”]')
INLINE_WHITESPACE_RE = re.compile(r"\s+")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DPR_FAISS_INDEX_DIR = PROJECT_ROOT / "index/dpr_faiss"
DPR_FAISS_VARIANT_NAMES = (
    "title_first_sentence",
    "title_first_two_sentences",
    "title_first_paragraph",
    "title_entire_article",
)


@lru_cache(maxsize=None)
def warn_once(message: str) -> None:
    print(message)


@lru_cache(maxsize=1)
def get_stop_words() -> frozenset[str]:
    return frozenset(load_stop_words())


def normalize_query_terms(query_text: str) -> list[str]:
    return tokenize_body(query_text, set(get_stop_words()))


def build_terms_query(terms: list[str]):
    if not terms:
        return None

    return whoosh_query.Or([whoosh_query.Term("body", term) for term in terms])


def extract_query_years(query_text: str) -> list[str]:
    return list(dict.fromkeys(YEAR_RE.findall(query_text)))


def build_year_match_query(query_text: str):
    years = extract_query_years(query_text)
    if not years:
        return None

    return whoosh_query.And(
        [
            whoosh_query.Or(
                [
                    whoosh_query.Term("title", year),
                    whoosh_query.Term("body", year),
                ]
            )
            for year in years
        ]
    )


def extract_query_quotes(query_text: str) -> list[str]:
    quotes = []
    seen = set()

    for match in QUOTE_RE.findall(query_text):
        quote = " ".join(match.split()).strip()
        if not quote or quote in seen:
            continue
        seen.add(quote)
        quotes.append(quote)

    return quotes


def build_quote_match_query(query_text: str):
    quotes = extract_query_quotes(query_text)
    if not quotes:
        return None

    quote_queries = []
    for quote in quotes:
        quote_terms = normalize_query_terms(quote)
        if not quote_terms:
            continue
        quote_queries.append(whoosh_query.And([whoosh_query.Term("body", term) for term in quote_terms]))

    if not quote_queries:
        return None

    if len(quote_queries) == 1:
        return quote_queries[0]

    return whoosh_query.And(quote_queries)


def normalize_category_keyword(category_text: str) -> str:
    return INLINE_WHITESPACE_RE.sub(" ", category_text.casefold()).strip()


def build_category_match_query(field_name: str, category_text: str):
    normalized_category = normalize_category_keyword(category_text)
    if not normalized_category:
        return None

    return whoosh_query.Term(field_name, normalized_category)


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


def normalize_results_to_unit_interval(results: list[dict]) -> list[dict]:
    if not results:
        return []

    max_score = max(result["score"] for result in results)
    if max_score <= 0:
        return [{**result, "raw_score": result["score"], "score": 0.0} for result in results]

    return [
        {
            **result,
            "raw_score": result["score"],
            "score": result["score"] / max_score,
        }
        for result in results
    ]


def normalize_dense_results_to_unit_interval(results: list[dict]) -> list[dict]:
    if not results:
        return []

    min_score = min(result["score"] for result in results)
    max_score = max(result["score"] for result in results)
    if max_score == min_score:
        normalized_score = 1.0 if max_score > 0 else 0.0
        return [
            {
                **result,
                "raw_score": result["score"],
                "score": normalized_score,
            }
            for result in results
        ]

    score_range = max_score - min_score
    return [
        {
            **result,
            "raw_score": result["score"],
            "score": (result["score"] - min_score) / score_range,
        }
        for result in results
    ]


def get_cached_component_results(
    *,
    component_name: str,
    query_text: str,
    query_category: str | None,
    component_limit: int,
    index_paths: list[Path] | tuple[Path, ...],
    extra_config: dict | None,
    compute_results,
    query_embedding=None,
) -> list[dict]:
    return get_or_compute_component_results(
        component_name=component_name,
        query_text=query_text,
        query_category=query_category,
        component_limit=component_limit,
        index_paths=index_paths,
        extra_config=extra_config,
        compute_results=compute_results,
        query_embedding_digest=digest_query_embedding(query_embedding),
    )


@lru_cache(maxsize=1)
def load_dpr_question_encoder():
    try:
        import torch
        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "transformers and torch are required for FAISS-backed DPR scoring."
        ) from error

    model_name = "facebook/dpr-question_encoder-single-nq-base"
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
    model = DPRQuestionEncoder.from_pretrained(model_name, use_safetensors=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


@lru_cache(maxsize=8)
def load_dpr_faiss_variant(variant_dir: str):
    try:
        import faiss
    except ModuleNotFoundError as error:
        raise RuntimeError("faiss is required for FAISS-backed DPR scoring.") from error

    variant_path = Path(variant_dir)
    index = faiss.read_index(str(variant_path / "index.faiss"))
    metadata_path = variant_path / "metadata.jsonl"
    metadata_rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return index, metadata_rows


def encode_dpr_query(query_text: str):
    import numpy as np
    import torch

    tokenizer, model, device = load_dpr_question_encoder()
    encoded_inputs = tokenizer(
        query_text,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}

    with torch.no_grad():
        outputs = model(**encoded_inputs).pooler_output

    return outputs.detach().cpu().numpy().astype(np.float32)


def search_dpr_faiss(
    query_text: str,
    limit: int = WEIGHTED_COMPONENT_LIMIT,
    dpr_faiss_index_dir: Path = DEFAULT_DPR_FAISS_INDEX_DIR,
    query_embedding=None,
    cache_namespace: str = "default",
) -> list[dict]:
    import numpy as np

    if not query_text.strip() or not dpr_faiss_index_dir.exists():
        return []

    if query_embedding is None:
        try:
            query_embedding = encode_dpr_query(query_text)
        except Exception as error:
            warn_once(f"[search_dpr_faiss] Skipping dense scoring: {error}")
            return []
    else:
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        elif query_embedding.ndim != 2:
            raise ValueError(
                f"Expected query_embedding to have 1 or 2 dimensions, got shape {query_embedding.shape}."
            )

    aggregated_results: dict[tuple[str, int], dict] = {}

    for variant_name in DPR_FAISS_VARIANT_NAMES:
        variant_dir = dpr_faiss_index_dir / variant_name
        if not variant_dir.exists():
            continue

        try:
            index, metadata_rows = load_dpr_faiss_variant(str(variant_dir))
        except Exception as error:
            warn_once(f"[search_dpr_faiss] Skipping dense scoring: {error}")
            return []
        search_limit = min(limit, index.ntotal)
        if search_limit <= 0:
            continue

        def compute_dense_results(
            current_index=index,
            current_metadata_rows=metadata_rows,
            current_search_limit=search_limit,
        ):
            scores, row_indexes = current_index.search(query_embedding, current_search_limit)
            return normalize_dense_results_to_unit_interval(
                [
                    {
                        "title": current_metadata_rows[row_index]["title"],
                        "body": "",
                        "source_file": current_metadata_rows[row_index]["source_file"],
                        "article_index": current_metadata_rows[row_index]["article_index"],
                        "is_redirect": 0,
                        "score": float(score),
                    }
                    for score, row_index in zip(scores[0], row_indexes[0])
                    if row_index >= 0
                ]
            )

        dense_results = get_cached_component_results(
            component_name=f"faiss_{variant_name}",
            query_text=query_text,
            query_category=None,
            component_limit=limit,
            index_paths=[variant_dir / "index.faiss", variant_dir / "metadata.jsonl"],
            extra_config={
                "metric": "inner_product",
                "doc_variant": variant_name,
                "query_source": cache_namespace,
            },
            query_embedding=query_embedding,
            compute_results=compute_dense_results,
        )

        for result in dense_results:
            key = weighted_result_key(result)
            existing = aggregated_results.get(key)
            if existing is None or result["score"] > existing["score"]:
                aggregated_results[key] = result

    ranked_results = sorted(
        aggregated_results.values(),
        key=lambda result: (-result["score"], result["article_index"]),
    )
    return ranked_results[:limit]


def average_dense_component_results(component_results_sets: list[list[dict]]) -> list[dict]:
    if not component_results_sets:
        return []

    aggregated_results: dict[tuple[str, int], dict] = {}
    total_components = len(component_results_sets)

    for component_results in component_results_sets:
        for result in component_results:
            key = weighted_result_key(result)
            entry = aggregated_results.get(key)
            if entry is None:
                entry = {
                    **result,
                    "score": 0.0,
                }
                aggregated_results[key] = entry
            entry["score"] += result["score"]

    for result in aggregated_results.values():
        result["score"] /= total_components

    return sorted(
        aggregated_results.values(),
        key=lambda result: (-result["score"], result["article_index"]),
    )


def build_faiss_precomputed_components(
    *,
    query_text: str,
    component_limit: int,
    dpr_faiss_index_dir: Path,
    faiss_weight: float,
    dense_query_embedding=None,
    natural_questions_avg_faiss_weight: float = 0.0,
    natural_questions_embeddings=None,
    natural_questions_concat_faiss_weight: float = 0.0,
    natural_questions_concat_embedding=None,
) -> list[tuple[str, float, list[dict]]]:
    precomputed_components = []

    if faiss_weight > 0:
        precomputed_components.append(
            (
                "faiss_score",
                faiss_weight,
                search_dpr_faiss(
                    query_text,
                    limit=component_limit,
                    dpr_faiss_index_dir=dpr_faiss_index_dir,
                    query_embedding=dense_query_embedding,
                    cache_namespace="default",
                ),
            )
        )

    if natural_questions_avg_faiss_weight > 0 and natural_questions_embeddings is not None:
        natural_question_results = [
            search_dpr_faiss(
                query_text,
                limit=component_limit,
                dpr_faiss_index_dir=dpr_faiss_index_dir,
                query_embedding=natural_question_embedding,
                cache_namespace=f"natural_question_{index_number}",
            )
            for index_number, natural_question_embedding in enumerate(
                natural_questions_embeddings,
                start=1,
            )
        ]
        precomputed_components.append(
            (
                "natural_questions_avg_faiss_score",
                natural_questions_avg_faiss_weight,
                average_dense_component_results(natural_question_results),
            )
        )

    if (
        natural_questions_concat_faiss_weight > 0
        and natural_questions_concat_embedding is not None
    ):
        precomputed_components.append(
            (
                "natural_questions_concat_faiss_score",
                natural_questions_concat_faiss_weight,
                search_dpr_faiss(
                    query_text,
                    limit=component_limit,
                    dpr_faiss_index_dir=dpr_faiss_index_dir,
                    query_embedding=natural_questions_concat_embedding,
                    cache_namespace="natural_questions_concat",
                ),
            )
        )

    return precomputed_components


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
    query_category: str | None,
    title_parser,
    body_parser,
    redirect_lookup: dict[tuple[str, int], dict],
    limit: int = 10,
    component_limit: int = WEIGHTED_COMPONENT_LIMIT,
    skip_redirects: bool = False,
    title_weight: float = WEIGHTED_TITLE_WEIGHT,
    redirect_weight: float = WEIGHTED_REDIRECT_WEIGHT,
    body_weight: float = WEIGHTED_BODY_WEIGHT,
    extra_body_components: list[tuple[str, float, object, object]] | None = None,
    extra_components: list[tuple[str, float, object, object]] | None = None,
    category_components: list[tuple[str, float, object, str]] | None = None,
    precomputed_components: list[tuple[str, float, list[dict]]] | None = None,
    component_cache_context: dict[str, dict] | None = None,
    redirect_searcher=None,
    redirect_title_parser=None,
    filter_main_redirects: bool = True,
    filter_redirect_results: bool = True,
    document_lookup: dict[tuple[str, int], dict] | None = None,
) -> list[dict]:
    if not query_text.strip():
        return []

    active_title = title_weight > 0
    active_body = body_weight > 0
    active_redirect = redirect_weight > 0 and not skip_redirects
    extra_body_components = [
        component for component in (extra_body_components or []) if component[1] > 0
    ]
    extra_components = [component for component in (extra_components or []) if component[1] > 0]
    category_components = [
        component for component in (category_components or []) if component[1] > 0
    ]
    precomputed_components = [
        component for component in (precomputed_components or []) if component[1] > 0
    ]
    if not any(
        [
            active_title,
            active_body,
            active_redirect,
            extra_body_components,
            extra_components,
            category_components,
            precomputed_components,
        ]
    ):
        return []
    component_cache_context = component_cache_context or {}
    redirect_searcher = redirect_searcher or searcher
    redirect_title_parser = redirect_title_parser or title_parser
    title_query = title_parser.parse(query_text) if active_title else None
    body_query = body_parser.parse(query_text) if active_body else None
    redirect_title_query = redirect_title_parser.parse(query_text) if active_redirect else None
    non_redirect_filter = (
        whoosh_query.NumericRange("is_redirect", 0, 0) if filter_main_redirects else None
    )
    redirect_filter = whoosh_query.NumericRange("is_redirect", 1, 1) if filter_redirect_results else None

    def cached_or_compute_results(
        score_name: str,
        *,
        compute_results,
        cache_query_text: str | None = None,
        cache_query_category: str | None = None,
    ) -> list[dict]:
        cache_context = component_cache_context.get(score_name)
        if cache_context is None:
            return compute_results()

        return get_cached_component_results(
            component_name=cache_context["component_name"],
            query_text=cache_query_text if cache_query_text is not None else query_text,
            query_category=cache_query_category,
            component_limit=component_limit,
            index_paths=cache_context["index_paths"],
            extra_config=cache_context.get("extra_config"),
            compute_results=compute_results,
        )

    title_results = []
    if active_title:
        title_results = cached_or_compute_results(
            "title_score",
            compute_results=lambda: normalize_results_to_unit_interval(
                serialize_results(
                    searcher.search(title_query, limit=component_limit, filter=non_redirect_filter)
                )
            ),
            cache_query_category=query_category,
        )
    body_results = []
    if active_body:
        body_results = cached_or_compute_results(
            "body_score",
            compute_results=lambda: normalize_results_to_unit_interval(
                serialize_results(
                    searcher.search(body_query, limit=component_limit, filter=non_redirect_filter)
                )
            ),
            cache_query_category=query_category,
        )
    extra_component_results = {}
    for score_name, _, component_searcher, component_parser in extra_body_components:
        extra_component_results[score_name] = cached_or_compute_results(
            score_name,
            compute_results=lambda current_searcher=component_searcher, current_parser=component_parser: normalize_results_to_unit_interval(
                serialize_results(
                    current_searcher.search(
                        current_parser.parse(query_text),
                        limit=component_limit,
                    )
                )
            ),
            cache_query_category=query_category,
        )
    for score_name, _, component_searcher, component_query_builder in extra_components:
        component_query = component_query_builder(query_text)
        if component_query is None:
            extra_component_results[score_name] = cached_or_compute_results(
                score_name,
                compute_results=lambda: [],
                cache_query_category=query_category,
            )
            continue

        extra_component_results[score_name] = cached_or_compute_results(
            score_name,
            compute_results=lambda current_searcher=component_searcher, current_query=component_query: normalize_results_to_unit_interval(
                serialize_results(
                    current_searcher.search(
                        current_query,
                        limit=component_limit,
                    )
                )
            ),
            cache_query_category=query_category,
        )
    for score_name, _, component_searcher, field_name in category_components:
        component_query = build_category_match_query(field_name, query_category or "")
        if component_query is None:
            extra_component_results[score_name] = cached_or_compute_results(
                score_name,
                compute_results=lambda: [],
                cache_query_text=query_category or "",
                cache_query_category=query_category,
            )
            continue

        extra_component_results[score_name] = cached_or_compute_results(
            score_name,
            compute_results=lambda current_searcher=component_searcher, current_query=component_query: normalize_results_to_unit_interval(
                serialize_results(
                    current_searcher.search(
                        current_query,
                        limit=component_limit,
                    )
                )
            ),
            cache_query_text=query_category or "",
            cache_query_category=query_category,
        )
    redirect_results = []
    if active_redirect:
        redirect_results = cached_or_compute_results(
            "redirect_score",
            compute_results=lambda: normalize_results_to_unit_interval(
                serialize_results(
                    redirect_searcher.search(
                        redirect_title_query,
                        limit=component_limit,
                        filter=redirect_filter,
                    )
                )
            ),
            cache_query_category=query_category,
        )

    aggregated_results: dict[tuple[str, int], dict] = {}
    document_cache: dict[tuple[str, int], dict | None] = {}
    score_names = [
        "title_score",
        "redirect_score",
        "body_score",
        *[score_name for score_name, _, _, _ in extra_body_components],
        *[score_name for score_name, _, _, _ in extra_components],
        *[score_name for score_name, _, _, _ in category_components],
        *[score_name for score_name, _, _ in precomputed_components],
    ]

    def resolve_base_result(result: dict) -> dict:
        if document_lookup is None:
            return result

        document = document_lookup.get(weighted_result_key(result))
        if document is None:
            return result

        return document

    def ensure_entry(result: dict) -> dict:
        base_result = resolve_base_result(result)
        key = weighted_result_key(base_result)
        entry = aggregated_results.get(key)
        if entry is None:
            entry = {
                **base_result,
                "score": 0.0,
            }
            for score_name in score_names:
                entry[score_name] = 0.0
            aggregated_results[key] = entry
        return entry

    for result in title_results:
        ensure_entry(result)["title_score"] = max(ensure_entry(result)["title_score"], result["score"])

    for result in body_results:
        ensure_entry(result)["body_score"] = max(ensure_entry(result)["body_score"], result["score"])

    for score_name, _, _, _ in extra_body_components:
        for result in extra_component_results[score_name]:
            ensure_entry(result)[score_name] = max(ensure_entry(result)[score_name], result["score"])
    for score_name, _, _, _ in extra_components:
        for result in extra_component_results[score_name]:
            ensure_entry(result)[score_name] = max(ensure_entry(result)[score_name], result["score"])
    for score_name, _, _, _ in category_components:
        for result in extra_component_results[score_name]:
            ensure_entry(result)[score_name] = max(ensure_entry(result)[score_name], result["score"])
    for score_name, _, component_results in precomputed_components:
        for result in component_results:
            ensure_entry(result)[score_name] = max(ensure_entry(result)[score_name], result["score"])

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
        result["score"] = title_weight * result["title_score"]
        result["score"] += redirect_weight * result["redirect_score"]
        result["score"] += body_weight * result["body_score"]
        for score_name, component_weight, _, _ in extra_body_components:
            result["score"] += component_weight * result[score_name]
        for score_name, component_weight, _, _ in extra_components:
            result["score"] += component_weight * result[score_name]
        for score_name, component_weight, _, _ in category_components:
            result["score"] += component_weight * result[score_name]
        for score_name, component_weight, _ in precomputed_components:
            result["score"] += component_weight * result[score_name]

    ranked_results = sorted(
        aggregated_results.values(),
        key=lambda result: (
            -result["score"],
            -result["title_score"],
            -result["redirect_score"],
            -result["body_score"],
            *[-result[score_name] for score_name, _, _, _ in extra_body_components],
            *[-result[score_name] for score_name, _, _, _ in extra_components],
            *[-result[score_name] for score_name, _, _, _ in category_components],
            *[-result[score_name] for score_name, _, _ in precomputed_components],
            result["article_index"],
        ),
    )
    return ranked_results[:limit]


def search_whoosh_weighted(
    query: str,
    query_category: str | None = None,
    limit: int = 10,
    component_limit: int = WEIGHTED_COMPONENT_LIMIT,
    index_dir: Path = DEFAULT_WHOOSH_TITLE_BODY_INDEX_DIR,
    category_index_dir: Path = DEFAULT_WHOOSH_TITLE_BODY_CATEGORY_INDEX_DIR,
    first_sentence_index_dir: Path = DEFAULT_WHOOSH_COLE_FIRST_SENTENCE_INDEX_DIR,
    first_two_sentences_index_dir: Path = DEFAULT_WHOOSH_COLE_FIRST_TWO_SENTENCES_INDEX_DIR,
    first_paragraph_index_dir: Path = DEFAULT_WHOOSH_COLE_FIRST_PARAGRAPH_INDEX_DIR,
    redirect_index_dir: Path = DEFAULT_WHOOSH_COLE_REDIRECT_INDEX_DIR,
    redirect_db_path: Path = DEFAULT_REDIRECT_DB_PATH,
    skip_redirects: bool = False,
    title_weight: float = WEIGHTED_TITLE_WEIGHT,
    redirect_weight: float = WEIGHTED_REDIRECT_WEIGHT,
    body_weight: float = WEIGHTED_BODY_WEIGHT,
    first_sentence_weight: float = WEIGHTED_FIRST_SENTENCE_WEIGHT,
    first_two_sentences_weight: float = WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT,
    first_paragraph_weight: float = WEIGHTED_FIRST_PARAGRAPH_WEIGHT,
    faiss_weight: float = WEIGHTED_FAISS_WEIGHT,
    natural_questions_avg_faiss_weight: float = WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT,
    natural_questions_concat_faiss_weight: float = WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT,
    category_first_sentence_weight: float = WEIGHTED_CATEGORY_FIRST_SENTENCE_WEIGHT,
    category_first_two_sentences_weight: float = WEIGHTED_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT,
    dpr_faiss_index_dir: Path = DEFAULT_DPR_FAISS_INDEX_DIR,
    dense_query_embedding=None,
    natural_questions_embeddings=None,
    natural_questions_concat_embedding=None,
) -> list[dict]:
    index = open_whoosh_title_body_index(index_dir)
    category_index = open_whoosh_title_body_category_index(category_index_dir)
    first_sentence_index = open_whoosh_cole_first_sentence_index(first_sentence_index_dir)
    first_two_sentences_index = open_whoosh_cole_first_two_sentences_index(
        first_two_sentences_index_dir
    )
    first_paragraph_index = open_whoosh_cole_first_paragraph_index(first_paragraph_index_dir)
    redirect_index = open_whoosh_cole_redirect_index(redirect_index_dir)
    redirect_lookup = load_redirect_lookup(redirect_db_path)

    with index.searcher(weighting=BM25F()) as searcher:
        with category_index.searcher(weighting=BM25F()) as category_searcher:
            with first_sentence_index.searcher(weighting=BM25F()) as first_sentence_searcher:
                with first_two_sentences_index.searcher(
                    weighting=BM25F()
                ) as first_two_sentences_searcher:
                    with first_paragraph_index.searcher(
                        weighting=BM25F()
                    ) as first_paragraph_searcher:
                        with redirect_index.searcher(weighting=BM25F()) as redirect_searcher:
                            document_lookup = build_stored_document_lookup(searcher)
                            title_parser = QueryParser("title", schema=index.schema, group=OrGroup)
                            body_parser = QueryParser("body", schema=index.schema, group=OrGroup)
                            first_sentence_parser = QueryParser(
                                "body",
                                schema=first_sentence_index.schema,
                                group=OrGroup,
                            )
                            first_two_sentences_parser = QueryParser(
                                "body",
                                schema=first_two_sentences_index.schema,
                                group=OrGroup,
                            )
                            first_paragraph_parser = QueryParser(
                                "body",
                                schema=first_paragraph_index.schema,
                                group=OrGroup,
                            )
                            redirect_title_parser = QueryParser(
                                "title",
                                schema=redirect_index.schema,
                                group=OrGroup,
                            )
                            component_cache_context = {
                                "title_score": {
                                    "component_name": "whoosh_title",
                                    "index_paths": [index_dir],
                                    "extra_config": {
                                        "field": "title",
                                        "filter_main_redirects": True,
                                    },
                                },
                                "body_score": {
                                    "component_name": "whoosh_body",
                                    "index_paths": [index_dir],
                                    "extra_config": {
                                        "field": "body",
                                        "filter_main_redirects": True,
                                    },
                                },
                                "first_sentence_score": {
                                    "component_name": "whoosh_first_sentence",
                                    "index_paths": [first_sentence_index_dir],
                                    "extra_config": {"field": "body"},
                                },
                                "first_two_sentences_score": {
                                    "component_name": "whoosh_first_two_sentences",
                                    "index_paths": [first_two_sentences_index_dir],
                                    "extra_config": {"field": "body"},
                                },
                                "first_paragraph_score": {
                                    "component_name": "whoosh_first_paragraph",
                                    "index_paths": [first_paragraph_index_dir],
                                    "extra_config": {"field": "body"},
                                },
                                "category_first_sentence_score": {
                                    "component_name": "whoosh_category_first_sentence",
                                    "index_paths": [category_index_dir],
                                    "extra_config": {"field": "title_first_sentence_categories"},
                                },
                                "category_first_two_sentences_score": {
                                    "component_name": "whoosh_category_first_two_sentences",
                                    "index_paths": [category_index_dir],
                                    "extra_config": {"field": "title_first_two_sentences_categories"},
                                },
                                "redirect_score": {
                                    "component_name": "whoosh_redirect_title",
                                    "index_paths": [redirect_index_dir],
                                    "extra_config": {"field": "title", "filter_redirect_results": False},
                                },
                            }
                            return search_whoosh_weighted_with_searcher(
                                searcher,
                                query_text=query,
                                query_category=query_category,
                                title_parser=title_parser,
                                body_parser=body_parser,
                                redirect_lookup=redirect_lookup,
                                limit=limit,
                                component_limit=component_limit,
                                skip_redirects=skip_redirects,
                                title_weight=title_weight,
                                redirect_weight=redirect_weight,
                                body_weight=body_weight,
                                extra_body_components=[
                                    (
                                        "first_sentence_score",
                                        first_sentence_weight,
                                        first_sentence_searcher,
                                        first_sentence_parser,
                                    ),
                                    (
                                        "first_two_sentences_score",
                                        first_two_sentences_weight,
                                        first_two_sentences_searcher,
                                        first_two_sentences_parser,
                                    ),
                                    (
                                        "first_paragraph_score",
                                        first_paragraph_weight,
                                        first_paragraph_searcher,
                                        first_paragraph_parser,
                                    ),
                                ],
                                category_components=[
                                    (
                                        "category_first_sentence_score",
                                        category_first_sentence_weight,
                                        category_searcher,
                                        "title_first_sentence_categories",
                                    ),
                                    (
                                        "category_first_two_sentences_score",
                                        category_first_two_sentences_weight,
                                        category_searcher,
                                        "title_first_two_sentences_categories",
                                    ),
                                ],
                                precomputed_components=build_faiss_precomputed_components(
                                    query_text=query,
                                    component_limit=component_limit,
                                    dpr_faiss_index_dir=dpr_faiss_index_dir,
                                    faiss_weight=faiss_weight,
                                    dense_query_embedding=dense_query_embedding,
                                    natural_questions_avg_faiss_weight=natural_questions_avg_faiss_weight,
                                    natural_questions_embeddings=natural_questions_embeddings,
                                    natural_questions_concat_faiss_weight=natural_questions_concat_faiss_weight,
                                    natural_questions_concat_embedding=natural_questions_concat_embedding,
                                ),
                                component_cache_context=component_cache_context,
                                redirect_searcher=redirect_searcher,
                                redirect_title_parser=redirect_title_parser,
                                filter_main_redirects=True,
                                filter_redirect_results=False,
                                document_lookup=document_lookup,
                            )


def search_whoosh_weighted_cole(
    query: str,
    query_category: str | None = None,
    limit: int = 10,
    component_limit: int = WEIGHTED_COMPONENT_LIMIT,
    index_dir: Path = DEFAULT_WHOOSH_COLE_INDEX_DIR,
    category_index_dir: Path = DEFAULT_WHOOSH_TITLE_BODY_CATEGORY_INDEX_DIR,
    first_sentence_index_dir: Path = DEFAULT_WHOOSH_COLE_FIRST_SENTENCE_INDEX_DIR,
    first_two_sentences_index_dir: Path = DEFAULT_WHOOSH_COLE_FIRST_TWO_SENTENCES_INDEX_DIR,
    first_paragraph_index_dir: Path = DEFAULT_WHOOSH_COLE_FIRST_PARAGRAPH_INDEX_DIR,
    redirect_index_dir: Path = DEFAULT_WHOOSH_COLE_REDIRECT_INDEX_DIR,
    redirect_db_path: Path = DEFAULT_REDIRECT_DB_PATH,
    skip_redirects: bool = False,
    title_weight: float = WEIGHTED_TITLE_WEIGHT,
    redirect_weight: float = WEIGHTED_REDIRECT_WEIGHT,
    body_weight: float = WEIGHTED_BODY_WEIGHT,
    first_sentence_weight: float = WEIGHTED_FIRST_SENTENCE_WEIGHT,
    first_two_sentences_weight: float = WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT,
    first_paragraph_weight: float = WEIGHTED_FIRST_PARAGRAPH_WEIGHT,
    faiss_weight: float = WEIGHTED_FAISS_WEIGHT,
    natural_questions_avg_faiss_weight: float = WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT,
    natural_questions_concat_faiss_weight: float = WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT,
    year_match_weight: float = WEIGHTED_YEAR_MATCH_WEIGHT,
    quote_match_weight: float = WEIGHTED_QUOTE_MATCH_WEIGHT,
    category_first_sentence_weight: float = WEIGHTED_CATEGORY_FIRST_SENTENCE_WEIGHT,
    category_first_two_sentences_weight: float = WEIGHTED_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT,
    dpr_faiss_index_dir: Path = DEFAULT_DPR_FAISS_INDEX_DIR,
    dense_query_embedding=None,
    natural_questions_embeddings=None,
    natural_questions_concat_embedding=None,
) -> list[dict]:
    index = open_whoosh_cole_index(index_dir)
    category_index = open_whoosh_title_body_category_index(category_index_dir)
    first_sentence_index = open_whoosh_cole_first_sentence_index(first_sentence_index_dir)
    first_two_sentences_index = open_whoosh_cole_first_two_sentences_index(
        first_two_sentences_index_dir
    )
    first_paragraph_index = open_whoosh_cole_first_paragraph_index(first_paragraph_index_dir)
    redirect_index = open_whoosh_cole_redirect_index(redirect_index_dir)
    redirect_lookup = load_redirect_lookup(redirect_db_path)

    with index.searcher(weighting=BM25F()) as searcher:
        with category_index.searcher(weighting=BM25F()) as category_searcher:
            with first_sentence_index.searcher(weighting=BM25F()) as first_sentence_searcher:
                with first_two_sentences_index.searcher(
                    weighting=BM25F()
                ) as first_two_sentences_searcher:
                    with first_paragraph_index.searcher(
                        weighting=BM25F()
                    ) as first_paragraph_searcher:
                        with redirect_index.searcher(weighting=BM25F()) as redirect_searcher:
                            document_lookup = build_stored_document_lookup(searcher)
                            title_parser = QueryParser("title", schema=index.schema, group=OrGroup)
                            body_parser = QueryParser("body", schema=index.schema, group=OrGroup)
                            first_sentence_parser = QueryParser(
                                "body",
                                schema=first_sentence_index.schema,
                                group=OrGroup,
                            )
                            first_two_sentences_parser = QueryParser(
                                "body",
                                schema=first_two_sentences_index.schema,
                                group=OrGroup,
                            )
                            first_paragraph_parser = QueryParser(
                                "body",
                                schema=first_paragraph_index.schema,
                                group=OrGroup,
                            )
                            redirect_title_parser = QueryParser(
                                "title",
                                schema=redirect_index.schema,
                                group=OrGroup,
                            )
                            quote_match_query_builder = build_quote_match_query
                            component_cache_context = {
                                "title_score": {
                                    "component_name": "cole_title",
                                    "index_paths": [index_dir],
                                    "extra_config": {
                                        "field": "title",
                                        "filter_main_redirects": False,
                                    },
                                },
                                "body_score": {
                                    "component_name": "cole_body",
                                    "index_paths": [index_dir],
                                    "extra_config": {
                                        "field": "body",
                                        "filter_main_redirects": False,
                                    },
                                },
                                "first_sentence_score": {
                                    "component_name": "whoosh_first_sentence",
                                    "index_paths": [first_sentence_index_dir],
                                    "extra_config": {"field": "body"},
                                },
                                "first_two_sentences_score": {
                                    "component_name": "whoosh_first_two_sentences",
                                    "index_paths": [first_two_sentences_index_dir],
                                    "extra_config": {"field": "body"},
                                },
                                "first_paragraph_score": {
                                    "component_name": "whoosh_first_paragraph",
                                    "index_paths": [first_paragraph_index_dir],
                                    "extra_config": {"field": "body"},
                                },
                                "year_match_score": {
                                    "component_name": "cole_year_match",
                                    "index_paths": [index_dir],
                                    "extra_config": {"query_builder": "year_match"},
                                },
                                "quote_match_score": {
                                    "component_name": "cole_quote_match",
                                    "index_paths": [index_dir],
                                    "extra_config": {"query_builder": "quote_match"},
                                },
                                "category_first_sentence_score": {
                                    "component_name": "whoosh_category_first_sentence",
                                    "index_paths": [category_index_dir],
                                    "extra_config": {"field": "title_first_sentence_categories"},
                                },
                                "category_first_two_sentences_score": {
                                    "component_name": "whoosh_category_first_two_sentences",
                                    "index_paths": [category_index_dir],
                                    "extra_config": {"field": "title_first_two_sentences_categories"},
                                },
                                "redirect_score": {
                                    "component_name": "whoosh_redirect_title",
                                    "index_paths": [redirect_index_dir],
                                    "extra_config": {"field": "title", "filter_redirect_results": False},
                                },
                            }
                            return search_whoosh_weighted_with_searcher(
                                searcher,
                                query_text=query,
                                query_category=query_category,
                                title_parser=title_parser,
                                body_parser=body_parser,
                                redirect_lookup=redirect_lookup,
                                limit=limit,
                                component_limit=component_limit,
                                skip_redirects=skip_redirects,
                                title_weight=title_weight,
                                redirect_weight=redirect_weight,
                                body_weight=body_weight,
                                extra_body_components=[
                                    (
                                        "first_sentence_score",
                                        first_sentence_weight,
                                        first_sentence_searcher,
                                        first_sentence_parser,
                                    ),
                                    (
                                        "first_two_sentences_score",
                                        first_two_sentences_weight,
                                        first_two_sentences_searcher,
                                        first_two_sentences_parser,
                                    ),
                                    (
                                        "first_paragraph_score",
                                        first_paragraph_weight,
                                        first_paragraph_searcher,
                                        first_paragraph_parser,
                                    ),
                                ],
                                extra_components=[
                                    (
                                        "year_match_score",
                                        year_match_weight,
                                        searcher,
                                        build_year_match_query,
                                    ),
                                    (
                                        "quote_match_score",
                                        quote_match_weight,
                                        searcher,
                                        quote_match_query_builder,
                                    ),
                                ],
                                category_components=[
                                    (
                                        "category_first_sentence_score",
                                        category_first_sentence_weight,
                                        category_searcher,
                                        "title_first_sentence_categories",
                                    ),
                                    (
                                        "category_first_two_sentences_score",
                                        category_first_two_sentences_weight,
                                        category_searcher,
                                        "title_first_two_sentences_categories",
                                    ),
                                ],
                                precomputed_components=build_faiss_precomputed_components(
                                    query_text=query,
                                    component_limit=component_limit,
                                    dpr_faiss_index_dir=dpr_faiss_index_dir,
                                    faiss_weight=faiss_weight,
                                    dense_query_embedding=dense_query_embedding,
                                    natural_questions_avg_faiss_weight=natural_questions_avg_faiss_weight,
                                    natural_questions_embeddings=natural_questions_embeddings,
                                    natural_questions_concat_faiss_weight=natural_questions_concat_faiss_weight,
                                    natural_questions_concat_embedding=natural_questions_concat_embedding,
                                ),
                                component_cache_context=component_cache_context,
                                redirect_searcher=redirect_searcher,
                                redirect_title_parser=redirect_title_parser,
                                filter_main_redirects=False,
                                filter_redirect_results=False,
                                document_lookup=document_lookup,
                            )


def multi_search_whoosh_weighted(
    queries: list[str],
    query_categories: list[str] | None = None,
    limit: int = 10,
    component_limit: int = WEIGHTED_COMPONENT_LIMIT,
    index_dir: Path = DEFAULT_WHOOSH_TITLE_BODY_INDEX_DIR,
    category_index_dir: Path = DEFAULT_WHOOSH_TITLE_BODY_CATEGORY_INDEX_DIR,
    first_sentence_index_dir: Path = DEFAULT_WHOOSH_COLE_FIRST_SENTENCE_INDEX_DIR,
    first_two_sentences_index_dir: Path = DEFAULT_WHOOSH_COLE_FIRST_TWO_SENTENCES_INDEX_DIR,
    first_paragraph_index_dir: Path = DEFAULT_WHOOSH_COLE_FIRST_PARAGRAPH_INDEX_DIR,
    redirect_index_dir: Path = DEFAULT_WHOOSH_COLE_REDIRECT_INDEX_DIR,
    redirect_db_path: Path = DEFAULT_REDIRECT_DB_PATH,
    progress_every: int = 10,
    skip_redirects: bool = False,
    title_weight: float = WEIGHTED_TITLE_WEIGHT,
    redirect_weight: float = WEIGHTED_REDIRECT_WEIGHT,
    body_weight: float = WEIGHTED_BODY_WEIGHT,
    first_sentence_weight: float = WEIGHTED_FIRST_SENTENCE_WEIGHT,
    first_two_sentences_weight: float = WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT,
    first_paragraph_weight: float = WEIGHTED_FIRST_PARAGRAPH_WEIGHT,
    faiss_weight: float = WEIGHTED_FAISS_WEIGHT,
    natural_questions_avg_faiss_weight: float = WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT,
    natural_questions_concat_faiss_weight: float = WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT,
    category_first_sentence_weight: float = WEIGHTED_CATEGORY_FIRST_SENTENCE_WEIGHT,
    category_first_two_sentences_weight: float = WEIGHTED_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT,
    dpr_faiss_index_dir: Path = DEFAULT_DPR_FAISS_INDEX_DIR,
    dense_query_embeddings: list | None = None,
    natural_questions_embeddings: list | None = None,
    natural_questions_concat_embeddings: list | None = None,
) -> list[dict]:
    index = open_whoosh_title_body_index(index_dir)
    category_index = open_whoosh_title_body_category_index(category_index_dir)
    first_sentence_index = open_whoosh_cole_first_sentence_index(first_sentence_index_dir)
    first_two_sentences_index = open_whoosh_cole_first_two_sentences_index(
        first_two_sentences_index_dir
    )
    first_paragraph_index = open_whoosh_cole_first_paragraph_index(first_paragraph_index_dir)
    redirect_index = open_whoosh_cole_redirect_index(redirect_index_dir)
    redirect_lookup = load_redirect_lookup(redirect_db_path)
    total = len(queries)
    all_results = []
    query_categories = query_categories or [""] * total

    with index.searcher(weighting=BM25F()) as searcher:
        with category_index.searcher(weighting=BM25F()) as category_searcher:
            with first_sentence_index.searcher(weighting=BM25F()) as first_sentence_searcher:
                with first_two_sentences_index.searcher(
                    weighting=BM25F()
                ) as first_two_sentences_searcher:
                    with first_paragraph_index.searcher(
                        weighting=BM25F()
                    ) as first_paragraph_searcher:
                        with redirect_index.searcher(weighting=BM25F()) as redirect_searcher:
                            document_lookup = build_stored_document_lookup(searcher)
                            title_parser = QueryParser("title", schema=index.schema, group=OrGroup)
                            body_parser = QueryParser("body", schema=index.schema, group=OrGroup)
                            first_sentence_parser = QueryParser(
                                "body",
                                schema=first_sentence_index.schema,
                                group=OrGroup,
                            )
                            first_two_sentences_parser = QueryParser(
                                "body",
                                schema=first_two_sentences_index.schema,
                                group=OrGroup,
                            )
                            first_paragraph_parser = QueryParser(
                                "body",
                                schema=first_paragraph_index.schema,
                                group=OrGroup,
                            )
                            redirect_title_parser = QueryParser(
                                "title",
                                schema=redirect_index.schema,
                                group=OrGroup,
                            )
                            component_cache_context = {
                                "title_score": {
                                    "component_name": "whoosh_title",
                                    "index_paths": [index_dir],
                                    "extra_config": {
                                        "field": "title",
                                        "filter_main_redirects": True,
                                    },
                                },
                                "body_score": {
                                    "component_name": "whoosh_body",
                                    "index_paths": [index_dir],
                                    "extra_config": {
                                        "field": "body",
                                        "filter_main_redirects": True,
                                    },
                                },
                                "first_sentence_score": {
                                    "component_name": "whoosh_first_sentence",
                                    "index_paths": [first_sentence_index_dir],
                                    "extra_config": {"field": "body"},
                                },
                                "first_two_sentences_score": {
                                    "component_name": "whoosh_first_two_sentences",
                                    "index_paths": [first_two_sentences_index_dir],
                                    "extra_config": {"field": "body"},
                                },
                                "first_paragraph_score": {
                                    "component_name": "whoosh_first_paragraph",
                                    "index_paths": [first_paragraph_index_dir],
                                    "extra_config": {"field": "body"},
                                },
                                "category_first_sentence_score": {
                                    "component_name": "whoosh_category_first_sentence",
                                    "index_paths": [category_index_dir],
                                    "extra_config": {"field": "title_first_sentence_categories"},
                                },
                                "category_first_two_sentences_score": {
                                    "component_name": "whoosh_category_first_two_sentences",
                                    "index_paths": [category_index_dir],
                                    "extra_config": {"field": "title_first_two_sentences_categories"},
                                },
                                "redirect_score": {
                                    "component_name": "whoosh_redirect_title",
                                    "index_paths": [redirect_index_dir],
                                    "extra_config": {"field": "title", "filter_redirect_results": False},
                                },
                            }

                            for index_number, query_text in enumerate(queries, start=1):
                                dense_query_embedding = None
                                if dense_query_embeddings is not None:
                                    dense_query_embedding = dense_query_embeddings[index_number - 1]
                                natural_questions_embedding = None
                                if natural_questions_embeddings is not None:
                                    natural_questions_embedding = natural_questions_embeddings[
                                        index_number - 1
                                    ]
                                natural_questions_concat_embedding = None
                                if natural_questions_concat_embeddings is not None:
                                    natural_questions_concat_embedding = (
                                        natural_questions_concat_embeddings[index_number - 1]
                                    )
                                all_results.append(
                                    {
                                        "query": query_text,
                                        "results": search_whoosh_weighted_with_searcher(
                                            searcher,
                                            query_text=query_text,
                                            query_category=query_categories[index_number - 1],
                                            title_parser=title_parser,
                                            body_parser=body_parser,
                                            redirect_lookup=redirect_lookup,
                                            limit=limit,
                                            component_limit=component_limit,
                                            skip_redirects=skip_redirects,
                                            title_weight=title_weight,
                                            redirect_weight=redirect_weight,
                                            body_weight=body_weight,
                                            extra_body_components=[
                                                (
                                                    "first_sentence_score",
                                                    first_sentence_weight,
                                                    first_sentence_searcher,
                                                    first_sentence_parser,
                                                ),
                                                (
                                                    "first_two_sentences_score",
                                                    first_two_sentences_weight,
                                                    first_two_sentences_searcher,
                                                    first_two_sentences_parser,
                                                ),
                                                (
                                                    "first_paragraph_score",
                                                    first_paragraph_weight,
                                                    first_paragraph_searcher,
                                                    first_paragraph_parser,
                                                ),
                                            ],
                                            category_components=[
                                                (
                                                    "category_first_sentence_score",
                                                    category_first_sentence_weight,
                                                    category_searcher,
                                                    "title_first_sentence_categories",
                                                ),
                                                (
                                                    "category_first_two_sentences_score",
                                                    category_first_two_sentences_weight,
                                                    category_searcher,
                                                    "title_first_two_sentences_categories",
                                                ),
                                            ],
                                            precomputed_components=build_faiss_precomputed_components(
                                                query_text=query_text,
                                                component_limit=component_limit,
                                                dpr_faiss_index_dir=dpr_faiss_index_dir,
                                                faiss_weight=faiss_weight,
                                                dense_query_embedding=dense_query_embedding,
                                                natural_questions_avg_faiss_weight=natural_questions_avg_faiss_weight,
                                                natural_questions_embeddings=natural_questions_embedding,
                                                natural_questions_concat_faiss_weight=natural_questions_concat_faiss_weight,
                                                natural_questions_concat_embedding=natural_questions_concat_embedding,
                                            ),
                                            component_cache_context=component_cache_context,
                                            redirect_searcher=redirect_searcher,
                                            redirect_title_parser=redirect_title_parser,
                                            filter_main_redirects=True,
                                            filter_redirect_results=False,
                                            document_lookup=document_lookup,
                                        ),
                                    }
                                )

                                if progress_every and index_number % progress_every == 0:
                                    print(
                                        f"[multi_search_whoosh_weighted] Queries: {index_number}/{total}"
                                    )

    print(f"[multi_search_whoosh_weighted] Finished queries: {total}/{total}")
    return all_results


def multi_search_whoosh_weighted_cole(
    queries: list[str],
    query_categories: list[str] | None = None,
    limit: int = 10,
    component_limit: int = WEIGHTED_COMPONENT_LIMIT,
    index_dir: Path = DEFAULT_WHOOSH_COLE_INDEX_DIR,
    category_index_dir: Path = DEFAULT_WHOOSH_TITLE_BODY_CATEGORY_INDEX_DIR,
    first_sentence_index_dir: Path = DEFAULT_WHOOSH_COLE_FIRST_SENTENCE_INDEX_DIR,
    first_two_sentences_index_dir: Path = DEFAULT_WHOOSH_COLE_FIRST_TWO_SENTENCES_INDEX_DIR,
    first_paragraph_index_dir: Path = DEFAULT_WHOOSH_COLE_FIRST_PARAGRAPH_INDEX_DIR,
    redirect_index_dir: Path = DEFAULT_WHOOSH_COLE_REDIRECT_INDEX_DIR,
    redirect_db_path: Path = DEFAULT_REDIRECT_DB_PATH,
    progress_every: int = 10,
    skip_redirects: bool = False,
    title_weight: float = WEIGHTED_TITLE_WEIGHT,
    redirect_weight: float = WEIGHTED_REDIRECT_WEIGHT,
    body_weight: float = WEIGHTED_BODY_WEIGHT,
    first_sentence_weight: float = WEIGHTED_FIRST_SENTENCE_WEIGHT,
    first_two_sentences_weight: float = WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT,
    first_paragraph_weight: float = WEIGHTED_FIRST_PARAGRAPH_WEIGHT,
    faiss_weight: float = WEIGHTED_FAISS_WEIGHT,
    natural_questions_avg_faiss_weight: float = WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT,
    natural_questions_concat_faiss_weight: float = WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT,
    year_match_weight: float = WEIGHTED_YEAR_MATCH_WEIGHT,
    quote_match_weight: float = WEIGHTED_QUOTE_MATCH_WEIGHT,
    category_first_sentence_weight: float = WEIGHTED_CATEGORY_FIRST_SENTENCE_WEIGHT,
    category_first_two_sentences_weight: float = WEIGHTED_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT,
    dpr_faiss_index_dir: Path = DEFAULT_DPR_FAISS_INDEX_DIR,
    dense_query_embeddings: list | None = None,
    natural_questions_embeddings: list | None = None,
    natural_questions_concat_embeddings: list | None = None,
) -> list[dict]:
    index = open_whoosh_cole_index(index_dir)
    category_index = open_whoosh_title_body_category_index(category_index_dir)
    first_sentence_index = open_whoosh_cole_first_sentence_index(first_sentence_index_dir)
    first_two_sentences_index = open_whoosh_cole_first_two_sentences_index(
        first_two_sentences_index_dir
    )
    first_paragraph_index = open_whoosh_cole_first_paragraph_index(first_paragraph_index_dir)
    redirect_index = open_whoosh_cole_redirect_index(redirect_index_dir)
    redirect_lookup = load_redirect_lookup(redirect_db_path)
    total = len(queries)
    all_results = []
    query_categories = query_categories or [""] * total

    with index.searcher(weighting=BM25F()) as searcher:
        with category_index.searcher(weighting=BM25F()) as category_searcher:
            with first_sentence_index.searcher(weighting=BM25F()) as first_sentence_searcher:
                with first_two_sentences_index.searcher(
                    weighting=BM25F()
                ) as first_two_sentences_searcher:
                    with first_paragraph_index.searcher(
                        weighting=BM25F()
                    ) as first_paragraph_searcher:
                        with redirect_index.searcher(weighting=BM25F()) as redirect_searcher:
                            document_lookup = build_stored_document_lookup(searcher)
                            title_parser = QueryParser("title", schema=index.schema, group=OrGroup)
                            body_parser = QueryParser("body", schema=index.schema, group=OrGroup)
                            first_sentence_parser = QueryParser(
                                "body",
                                schema=first_sentence_index.schema,
                                group=OrGroup,
                            )
                            first_two_sentences_parser = QueryParser(
                                "body",
                                schema=first_two_sentences_index.schema,
                                group=OrGroup,
                            )
                            first_paragraph_parser = QueryParser(
                                "body",
                                schema=first_paragraph_index.schema,
                                group=OrGroup,
                            )
                            redirect_title_parser = QueryParser(
                                "title",
                                schema=redirect_index.schema,
                                group=OrGroup,
                            )
                            quote_match_query_builder = build_quote_match_query
                            component_cache_context = {
                                "title_score": {
                                    "component_name": "cole_title",
                                    "index_paths": [index_dir],
                                    "extra_config": {
                                        "field": "title",
                                        "filter_main_redirects": False,
                                    },
                                },
                                "body_score": {
                                    "component_name": "cole_body",
                                    "index_paths": [index_dir],
                                    "extra_config": {
                                        "field": "body",
                                        "filter_main_redirects": False,
                                    },
                                },
                                "first_sentence_score": {
                                    "component_name": "whoosh_first_sentence",
                                    "index_paths": [first_sentence_index_dir],
                                    "extra_config": {"field": "body"},
                                },
                                "first_two_sentences_score": {
                                    "component_name": "whoosh_first_two_sentences",
                                    "index_paths": [first_two_sentences_index_dir],
                                    "extra_config": {"field": "body"},
                                },
                                "first_paragraph_score": {
                                    "component_name": "whoosh_first_paragraph",
                                    "index_paths": [first_paragraph_index_dir],
                                    "extra_config": {"field": "body"},
                                },
                                "year_match_score": {
                                    "component_name": "cole_year_match",
                                    "index_paths": [index_dir],
                                    "extra_config": {"query_builder": "year_match"},
                                },
                                "quote_match_score": {
                                    "component_name": "cole_quote_match",
                                    "index_paths": [index_dir],
                                    "extra_config": {"query_builder": "quote_match"},
                                },
                                "category_first_sentence_score": {
                                    "component_name": "whoosh_category_first_sentence",
                                    "index_paths": [category_index_dir],
                                    "extra_config": {"field": "title_first_sentence_categories"},
                                },
                                "category_first_two_sentences_score": {
                                    "component_name": "whoosh_category_first_two_sentences",
                                    "index_paths": [category_index_dir],
                                    "extra_config": {"field": "title_first_two_sentences_categories"},
                                },
                                "redirect_score": {
                                    "component_name": "whoosh_redirect_title",
                                    "index_paths": [redirect_index_dir],
                                    "extra_config": {"field": "title", "filter_redirect_results": False},
                                },
                            }

                            for index_number, query_text in enumerate(queries, start=1):
                                dense_query_embedding = None
                                if dense_query_embeddings is not None:
                                    dense_query_embedding = dense_query_embeddings[index_number - 1]
                                natural_questions_embedding = None
                                if natural_questions_embeddings is not None:
                                    natural_questions_embedding = natural_questions_embeddings[
                                        index_number - 1
                                    ]
                                natural_questions_concat_embedding = None
                                if natural_questions_concat_embeddings is not None:
                                    natural_questions_concat_embedding = (
                                        natural_questions_concat_embeddings[index_number - 1]
                                    )
                                all_results.append(
                                    {
                                        "query": query_text,
                                        "results": search_whoosh_weighted_with_searcher(
                                            searcher,
                                            query_text=query_text,
                                            query_category=query_categories[index_number - 1],
                                            title_parser=title_parser,
                                            body_parser=body_parser,
                                            redirect_lookup=redirect_lookup,
                                            limit=limit,
                                            component_limit=component_limit,
                                            skip_redirects=skip_redirects,
                                            title_weight=title_weight,
                                            redirect_weight=redirect_weight,
                                            body_weight=body_weight,
                                            extra_body_components=[
                                                (
                                                    "first_sentence_score",
                                                    first_sentence_weight,
                                                    first_sentence_searcher,
                                                    first_sentence_parser,
                                                ),
                                                (
                                                    "first_two_sentences_score",
                                                    first_two_sentences_weight,
                                                    first_two_sentences_searcher,
                                                    first_two_sentences_parser,
                                                ),
                                                (
                                                    "first_paragraph_score",
                                                    first_paragraph_weight,
                                                    first_paragraph_searcher,
                                                    first_paragraph_parser,
                                                ),
                                            ],
                                            extra_components=[
                                                (
                                                    "year_match_score",
                                                    year_match_weight,
                                                    searcher,
                                                    build_year_match_query,
                                                ),
                                                (
                                                    "quote_match_score",
                                                    quote_match_weight,
                                                    searcher,
                                                    quote_match_query_builder,
                                                ),
                                            ],
                                            category_components=[
                                                (
                                                    "category_first_sentence_score",
                                                    category_first_sentence_weight,
                                                    category_searcher,
                                                    "title_first_sentence_categories",
                                                ),
                                                (
                                                    "category_first_two_sentences_score",
                                                    category_first_two_sentences_weight,
                                                    category_searcher,
                                                    "title_first_two_sentences_categories",
                                                ),
                                            ],
                                            precomputed_components=build_faiss_precomputed_components(
                                                query_text=query_text,
                                                component_limit=component_limit,
                                                dpr_faiss_index_dir=dpr_faiss_index_dir,
                                                faiss_weight=faiss_weight,
                                                dense_query_embedding=dense_query_embedding,
                                                natural_questions_avg_faiss_weight=natural_questions_avg_faiss_weight,
                                                natural_questions_embeddings=natural_questions_embedding,
                                                natural_questions_concat_faiss_weight=natural_questions_concat_faiss_weight,
                                                natural_questions_concat_embedding=natural_questions_concat_embedding,
                                            ),
                                            component_cache_context=component_cache_context,
                                            redirect_searcher=redirect_searcher,
                                            redirect_title_parser=redirect_title_parser,
                                            filter_main_redirects=False,
                                            filter_redirect_results=False,
                                            document_lookup=document_lookup,
                                        ),
                                    }
                                )

                                if progress_every and index_number % progress_every == 0:
                                    print(
                                        f"[multi_search_whoosh_weighted_cole] Queries: {index_number}/{total}"
                                    )

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
