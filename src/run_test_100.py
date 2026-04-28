"""Evaluate search quality on the first 100 Jeopardy questions."""

# python -m src.run_test_100 --mode whoosh --query-mode entity

import argparse
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import json
import numpy as np
from pathlib import Path
import re
import sqlite3
import time

try:
    from src.processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from src.processor_natural_question_dpr_embeddings import (
        DEFAULT_OUTPUT_PATH as DEFAULT_NATURAL_QUESTION_DPR_EMBEDDINGS_PATH,
    )
    from src.processor_question_dpr_embeddings import (
        DEFAULT_OUTPUT_PATH as DEFAULT_QUESTION_DPR_EMBEDDINGS_PATH,
        category_plus_clue_text,
        clue_only_text,
    )
    from src.retrieval_cache import (
        build_component_cache_key,
        get_cache_stats,
        load_cached_component_results,
        reset_cache_stats,
        sha256_text,
        store_component_results,
    )
    from src.search import (
        DEFAULT_DPR_FAISS_INDEX_DIR,
        EQUAL_BODY_WEIGHT,
        EQUAL_CATEGORY_FIRST_SENTENCE_WEIGHT,
        EQUAL_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT,
        EQUAL_FAISS_WEIGHT,
        EQUAL_FIRST_PARAGRAPH_WEIGHT,
        EQUAL_FIRST_SENTENCE_WEIGHT,
        EQUAL_FIRST_TWO_SENTENCES_WEIGHT,
        EQUAL_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT,
        EQUAL_NATURAL_QUESTIONS_AVG_QWEN_SUMMARY_FAISS_WEIGHT,
        EQUAL_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT,
        EQUAL_NATURAL_QUESTIONS_CONCAT_QWEN_SUMMARY_FAISS_WEIGHT,
        EQUAL_QWEN_SUMMARY_FAISS_WEIGHT,
        EQUAL_QUOTE_MATCH_WEIGHT,
        EQUAL_REDIRECT_WEIGHT,
        EQUAL_TITLE_WEIGHT,
        EQUAL_YEAR_MATCH_WEIGHT,
        DPR_FAISS_VARIANT_NAMES,
        DPR_QWEN_SUMMARY_FAISS_VARIANT_NAMES,
        WEIGHTED_BODY_WEIGHT,
        WEIGHTED_CATEGORY_FIRST_SENTENCE_WEIGHT,
        WEIGHTED_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT,
        WEIGHTED_FAISS_WEIGHT,
        WEIGHTED_FIRST_PARAGRAPH_WEIGHT,
        WEIGHTED_FIRST_SENTENCE_WEIGHT,
        WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT,
        WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT,
        WEIGHTED_NATURAL_QUESTIONS_AVG_QWEN_SUMMARY_FAISS_WEIGHT,
        WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT,
        WEIGHTED_NATURAL_QUESTIONS_CONCAT_QWEN_SUMMARY_FAISS_WEIGHT,
        WEIGHTED_QWEN_SUMMARY_FAISS_WEIGHT,
        WEIGHTED_QUOTE_MATCH_WEIGHT,
        WEIGHTED_REDIRECT_WEIGHT,
        WEIGHTED_TITLE_WEIGHT,
        WEIGHTED_YEAR_MATCH_WEIGHT,
        multi_search,
        multi_search_whoosh_cole,
        multi_search_whoosh_default,
        multi_search_whoosh_weighted,
        multi_search_whoosh_weighted_cole,
        multi_search_whoosh_title_body,
        load_dpr_faiss_variant,
        load_dpr_question_encoder,
        load_redirect_lookup,
        open_index,
        open_whoosh_cole_first_paragraph_index,
        open_whoosh_cole_first_sentence_index,
        open_whoosh_cole_first_two_sentences_index,
        open_whoosh_cole_index,
        open_whoosh_cole_redirect_index,
        open_whoosh_index,
        open_whoosh_title_body_category_index,
        open_whoosh_title_body_index,
    )
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from processor_natural_question_dpr_embeddings import (
        DEFAULT_OUTPUT_PATH as DEFAULT_NATURAL_QUESTION_DPR_EMBEDDINGS_PATH,
    )
    from processor_question_dpr_embeddings import (
        DEFAULT_OUTPUT_PATH as DEFAULT_QUESTION_DPR_EMBEDDINGS_PATH,
        category_plus_clue_text,
        clue_only_text,
    )
    from retrieval_cache import (
        build_component_cache_key,
        get_cache_stats,
        load_cached_component_results,
        reset_cache_stats,
        sha256_text,
        store_component_results,
    )
    from search import (
        DEFAULT_DPR_FAISS_INDEX_DIR,
        EQUAL_BODY_WEIGHT,
        EQUAL_CATEGORY_FIRST_SENTENCE_WEIGHT,
        EQUAL_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT,
        EQUAL_FAISS_WEIGHT,
        EQUAL_FIRST_PARAGRAPH_WEIGHT,
        EQUAL_FIRST_SENTENCE_WEIGHT,
        EQUAL_FIRST_TWO_SENTENCES_WEIGHT,
        EQUAL_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT,
        EQUAL_NATURAL_QUESTIONS_AVG_QWEN_SUMMARY_FAISS_WEIGHT,
        EQUAL_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT,
        EQUAL_NATURAL_QUESTIONS_CONCAT_QWEN_SUMMARY_FAISS_WEIGHT,
        EQUAL_QWEN_SUMMARY_FAISS_WEIGHT,
        EQUAL_QUOTE_MATCH_WEIGHT,
        EQUAL_REDIRECT_WEIGHT,
        EQUAL_TITLE_WEIGHT,
        EQUAL_YEAR_MATCH_WEIGHT,
        DPR_FAISS_VARIANT_NAMES,
        DPR_QWEN_SUMMARY_FAISS_VARIANT_NAMES,
        WEIGHTED_BODY_WEIGHT,
        WEIGHTED_CATEGORY_FIRST_SENTENCE_WEIGHT,
        WEIGHTED_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT,
        WEIGHTED_FAISS_WEIGHT,
        WEIGHTED_FIRST_PARAGRAPH_WEIGHT,
        WEIGHTED_FIRST_SENTENCE_WEIGHT,
        WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT,
        WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT,
        WEIGHTED_NATURAL_QUESTIONS_AVG_QWEN_SUMMARY_FAISS_WEIGHT,
        WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT,
        WEIGHTED_NATURAL_QUESTIONS_CONCAT_QWEN_SUMMARY_FAISS_WEIGHT,
        WEIGHTED_QWEN_SUMMARY_FAISS_WEIGHT,
        WEIGHTED_QUOTE_MATCH_WEIGHT,
        WEIGHTED_REDIRECT_WEIGHT,
        WEIGHTED_TITLE_WEIGHT,
        WEIGHTED_YEAR_MATCH_WEIGHT,
        multi_search,
        multi_search_whoosh_cole,
        multi_search_whoosh_default,
        multi_search_whoosh_weighted,
        multi_search_whoosh_weighted_cole,
        multi_search_whoosh_title_body,
        load_dpr_faiss_variant,
        load_dpr_question_encoder,
        load_redirect_lookup,
        open_index,
        open_whoosh_cole_first_paragraph_index,
        open_whoosh_cole_first_sentence_index,
        open_whoosh_cole_first_two_sentences_index,
        open_whoosh_cole_index,
        open_whoosh_cole_redirect_index,
        open_whoosh_index,
        open_whoosh_title_body_category_index,
        open_whoosh_title_body_index,
    )


TOP_K_VALUES = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 10000]
CROSS_ENCODER_TOP_K_VALUES = [1, 5, 10, 20, 50, 100]
DEFAULT_PARAPHRASE_COUNT = 4
DEFAULT_PARAPHRASE_FETCH_LIMIT = 1000
PARAPHRASE_MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"
RANK_FUSION_K = 60
WHITESPACE_RE = re.compile(r"\s+")
QUESTION_RESULTS_PREVIEW_LIMIT = 10
QUESTION_PROGRESS_EVERY = 10
CROSS_ENCODER_PROGRESS_EVERY = 1
DEFAULT_CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2" #"cross-encoder/ms-marco-roberta-base" #"cross-encoder/ms-marco-MiniLM-L-12-v2"
CROSS_ENCODER_CANDIDATE_LIMIT = 100
CROSS_ENCODER_CHUNK_TOKEN_LIMIT = 256
CROSS_ENCODER_SMALL_CHUNK_TOKEN_LIMIT = 100
CROSS_ENCODER_BATCH_SIZE = 16
CROSS_ENCODER_FUSION_INITIAL_WEIGHT = 1.0
CROSS_ENCODER_FUSION_CROSS_ENCODER_WEIGHT = 7.0
FINAL_RANKING_CACHE_SCHEMA_VERSION = "final_ranking_cache_v1"
CROSS_ENCODER_CACHE_SCHEMA_VERSION = "cross_encoder_cache_v1"
FINAL_RANKING_CACHE_STATS = {"hits": 0, "misses": 0}
CROSS_ENCODER_CACHE_STATS = {"hits": 0, "misses": 0}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROSS_ENCODER_NATURAL_QUESTIONS_PATH = (
    PROJECT_ROOT / "data/processed/questions_natural_qwen3_8b.json"
)
DEFAULT_CROSS_ENCODER_ARTICLE_DB_PATH = PROJECT_ROOT / "data/processed/wiki_articles_step1_clean.sqlite3"


@dataclass(frozen=True)
class JeopardyQuestion:
    category: str
    clue: str
    answer: str


def format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def format_elapsed_minutes(seconds: float) -> str:
    return f"{seconds / 60.0:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the first 100 Jeopardy questions.")
    parser.add_argument(
        "--mode",
        choices=["token", "whoosh", "whoosh_title_body", "cole"],
        default="token",
        help=(
            "token uses processor3 tokens; whoosh searches cleaned body; "
            "whoosh_title_body searches cleaned title and body; "
            "cole mirrors Cole's processor4 on cleaned text."
        ),
    )
    parser.add_argument(
        "--query-mode",
        choices=["full", "entity"],
        default="full",
        help="full uses the whole query text; entity keeps entities, nouns, and numbers.",
    )
    parser.add_argument(
        "--include-category",
        action="store_true",
        help="Add the question category before the clue.",
    )
    weight_group = parser.add_mutually_exclusive_group()
    weight_group.add_argument(
        "--weighted",
        action="store_true",
        help=(
            "Use weighted retrieval with title/body plus Cole lead indexes and redirect-only "
            "evidence. By default all weighted components currently start with the same weight."
        ),
    )
    weight_group.add_argument(
        "--weight-equal",
        action="store_true",
        help=(
            "Force equal weights across title, redirects, full body, first sentence, "
            "first two sentences, and first paragraph."
        ),
    )
    weight_group.add_argument(
        "--weighted-cole",
        action="store_true",
        help=(
            "Use the weighted formula on Cole for canonical title/body scoring while also "
            "using the Cole lead indexes and the redirect-only index."
        ),
    )
    parser.add_argument(
        "--skip-redirects",
        action="store_true",
        help="Filter redirect documents out of the retrieved results.",
    )
    parser.add_argument(
        "--paraphrase",
        action="store_true",
        help=(
            "Expand each raw query with model-generated paraphrases, retrieve candidates "
            "for every variant, and merge the rankings."
        ),
    )
    parser.add_argument(
        "--paraphrase-count",
        type=int,
        default=DEFAULT_PARAPHRASE_COUNT,
        help=(
            "Number of generated paraphrases per raw query when --paraphrase is enabled. "
            f"Default: {DEFAULT_PARAPHRASE_COUNT}."
        ),
    )
    parser.add_argument(
        "--paraphrase-fetch-limit",
        type=int,
        default=DEFAULT_PARAPHRASE_FETCH_LIMIT,
        help=(
            "Number of documents to retrieve for each paraphrase-expanded query before "
            f"rank fusion. Default: {DEFAULT_PARAPHRASE_FETCH_LIMIT}."
        ),
    )
    parser.add_argument(
        "--print-question-results",
        action="store_true",
        help=(
            "Print each question grouped by its first matching Top-k bucket, plus a "
            "preview of the top retrieved titles."
        ),
    )
    parser.add_argument(
        "--cross-encoder",
        action="store_true",
        help=(
            "Rerank the first 100 retrieved candidates with a cross-encoder after the "
            "BM25/FAISS combined ranking."
        ),
    )
    parser.add_argument(
        "--cross-encoder-mode",
        choices=["full_article", "chunks_256", "chunks_100"],
        default="chunks_256",
        help=(
            "full_article scores category+clue against the whole article text; "
            "chunks_256 and chunks_100 score article chunks and keep the max score."
        ),
    )
    parser.add_argument(
        "--cross-encoder-query-mode",
        choices=["category_clue", "natural_questions_avg"],
        default="category_clue",
        help=(
            "category_clue scores Category+Clue against each article; "
            "natural_questions_avg scores each generated natural question against the article "
            "and averages the five scores."
        ),
    )
    parser.add_argument(
        "--cross-encoder-natural-questions",
        type=Path,
        default=DEFAULT_CROSS_ENCODER_NATURAL_QUESTIONS_PATH,
        help=(
            "Generated natural-questions JSON used by "
            "--cross-encoder-query-mode natural_questions_avg."
        ),
    )
    parser.add_argument(
        "--cross-encoder-model",
        default=DEFAULT_CROSS_ENCODER_MODEL_NAME,
        help=f"Cross-encoder model name. Default: {DEFAULT_CROSS_ENCODER_MODEL_NAME}.",
    )
    return parser.parse_args()


def load_questions(path: Path = DEFAULT_QUESTIONS_JSON_PATH) -> list[JeopardyQuestion]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [JeopardyQuestion(**row) for row in rows]


def normalize_label(text: str) -> str:
    text = text.casefold().strip()
    return WHITESPACE_RE.sub(" ", text)


def valid_answers(answer: str) -> set[str]:
    return {normalize_label(part) for part in answer.split("|") if part.strip()}


def reset_stage_cache_stats() -> None:
    FINAL_RANKING_CACHE_STATS["hits"] = 0
    FINAL_RANKING_CACHE_STATS["misses"] = 0
    CROSS_ENCODER_CACHE_STATS["hits"] = 0
    CROSS_ENCODER_CACHE_STATS["misses"] = 0


def active_weight_config(weight_equal: bool, weighted: bool, weighted_cole: bool) -> dict:
    if weight_equal:
        config = {
            "title_weight": EQUAL_TITLE_WEIGHT,
            "redirect_weight": EQUAL_REDIRECT_WEIGHT,
            "body_weight": EQUAL_BODY_WEIGHT,
            "first_sentence_weight": EQUAL_FIRST_SENTENCE_WEIGHT,
            "first_two_sentences_weight": EQUAL_FIRST_TWO_SENTENCES_WEIGHT,
            "first_paragraph_weight": EQUAL_FIRST_PARAGRAPH_WEIGHT,
            "faiss_weight": EQUAL_FAISS_WEIGHT,
            "qwen_summary_faiss_weight": EQUAL_QWEN_SUMMARY_FAISS_WEIGHT,
            "natural_questions_avg_faiss_weight": EQUAL_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT,
            "natural_questions_concat_faiss_weight": EQUAL_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT,
            "natural_questions_avg_qwen_summary_faiss_weight": (
                EQUAL_NATURAL_QUESTIONS_AVG_QWEN_SUMMARY_FAISS_WEIGHT
            ),
            "natural_questions_concat_qwen_summary_faiss_weight": (
                EQUAL_NATURAL_QUESTIONS_CONCAT_QWEN_SUMMARY_FAISS_WEIGHT
            ),
            "category_first_sentence_weight": EQUAL_CATEGORY_FIRST_SENTENCE_WEIGHT,
            "category_first_two_sentences_weight": EQUAL_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT,
        }
        if weighted_cole:
            config["year_match_weight"] = EQUAL_YEAR_MATCH_WEIGHT
            config["quote_match_weight"] = EQUAL_QUOTE_MATCH_WEIGHT
        return config

    if not (weighted or weighted_cole):
        return {}

    config = {
        "title_weight": WEIGHTED_TITLE_WEIGHT,
        "redirect_weight": WEIGHTED_REDIRECT_WEIGHT,
        "body_weight": WEIGHTED_BODY_WEIGHT,
        "first_sentence_weight": WEIGHTED_FIRST_SENTENCE_WEIGHT,
        "first_two_sentences_weight": WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT,
        "first_paragraph_weight": WEIGHTED_FIRST_PARAGRAPH_WEIGHT,
        "faiss_weight": WEIGHTED_FAISS_WEIGHT,
        "qwen_summary_faiss_weight": WEIGHTED_QWEN_SUMMARY_FAISS_WEIGHT,
        "natural_questions_avg_faiss_weight": WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT,
        "natural_questions_concat_faiss_weight": WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT,
        "natural_questions_avg_qwen_summary_faiss_weight": (
            WEIGHTED_NATURAL_QUESTIONS_AVG_QWEN_SUMMARY_FAISS_WEIGHT
        ),
        "natural_questions_concat_qwen_summary_faiss_weight": (
            WEIGHTED_NATURAL_QUESTIONS_CONCAT_QWEN_SUMMARY_FAISS_WEIGHT
        ),
        "category_first_sentence_weight": WEIGHTED_CATEGORY_FIRST_SENTENCE_WEIGHT,
        "category_first_two_sentences_weight": WEIGHTED_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT,
    }
    if weighted_cole:
        config["year_match_weight"] = WEIGHTED_YEAR_MATCH_WEIGHT
        config["quote_match_weight"] = WEIGHTED_QUOTE_MATCH_WEIGHT
    return config


def final_ranking_cache_config(
    *,
    mode: str,
    query_mode: str,
    include_category: bool,
    weighted: bool,
    weight_equal: bool,
    weighted_cole: bool,
    skip_redirects: bool,
    paraphrase: bool,
    paraphrase_count: int,
    paraphrase_fetch_limit: int,
) -> dict:
    return {
        "cache_schema_version": FINAL_RANKING_CACHE_SCHEMA_VERSION,
        "mode": mode,
        "query_mode": query_mode,
        "include_category": include_category,
        "weighted": weighted,
        "weight_equal": weight_equal,
        "weighted_cole": weighted_cole,
        "skip_redirects": skip_redirects,
        "paraphrase": paraphrase,
        "paraphrase_count": paraphrase_count,
        "paraphrase_fetch_limit": paraphrase_fetch_limit,
        "weights": active_weight_config(weight_equal, weighted, weighted_cole),
    }


def final_ranking_cache_key(
    *,
    query_text: str,
    query_category: str,
    limit: int,
    cache_config: dict,
) -> str:
    return build_component_cache_key(
        component_name="final_combined_ranking",
        query_text=query_text,
        query_category=query_category,
        component_limit=limit,
        index_paths=[],
        extra_config=cache_config,
    )


def load_cached_final_ranking(
    *,
    query_text: str,
    query_category: str,
    limit: int,
    cache_config: dict,
) -> list[dict] | None:
    cached = load_cached_component_results(
        final_ranking_cache_key(
            query_text=query_text,
            query_category=query_category,
            limit=limit,
            cache_config=cache_config,
        )
    )
    if cached is None:
        FINAL_RANKING_CACHE_STATS["misses"] += 1
        return None

    FINAL_RANKING_CACHE_STATS["hits"] += 1
    return cached


def store_cached_final_ranking(
    *,
    query_text: str,
    query_category: str,
    limit: int,
    cache_config: dict,
    results: list[dict],
) -> None:
    store_component_results(
        cache_key=final_ranking_cache_key(
            query_text=query_text,
            query_category=query_category,
            limit=limit,
            cache_config=cache_config,
        ),
        component_name="final_combined_ranking",
        results=results,
    )


def is_correct_at_k(results: list[dict], answers: set[str], k: int) -> bool:
    top_titles = {normalize_label(result["title"]) for result in results[:k]}
    return bool(top_titles & answers)


def first_correct_k(results: list[dict], answers: set[str], top_k_values: list[int]) -> int | None:
    for k in top_k_values:
        if is_correct_at_k(results, answers, k):
            return k

    return None


def correct_at_k_values(results: list[dict], answers: set[str], top_k_values: list[int]) -> set[int]:
    matched_values = set()
    answer_rank = None

    for rank, result in enumerate(results, start=1):
        if normalize_label(result["title"]) in answers:
            answer_rank = rank
            break

    if answer_rank is None:
        return matched_values

    for k in top_k_values:
        if answer_rank <= k:
            matched_values.add(k)

    return matched_values


def first_correct_k_from_rank(answer_rank: int | None, top_k_values: list[int]) -> int | None:
    if answer_rank is None:
        return None

    for k in top_k_values:
        if answer_rank <= k:
            return k

    return None


def first_answer_rank(results: list[dict], answers: set[str]) -> int | None:
    for rank, result in enumerate(results, start=1):
        if normalize_label(result["title"]) in answers:
            return rank

    return None


@lru_cache(maxsize=1)
def load_entity_model():
    try:
        import spacy
    except ModuleNotFoundError as error:
        raise RuntimeError("spaCy is required for --query-mode entity.") from error

    try:
        return spacy.load("en_core_web_sm")
    except OSError as error:
        raise RuntimeError(
            "spaCy model en_core_web_sm is required for --query-mode entity. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from error


def entity_query_text(query_text: str) -> str:
    doc = load_entity_model()(query_text)
    spans = [(entity.start, entity.end, entity.text) for entity in doc.ents]
    covered_token_indexes = {
        token_index
        for start, end, _ in spans
        for token_index in range(start, end)
    }

    for token in doc:
        keep_token = token.like_num or token.pos_ in {"NOUN", "PROPN"}
        if token.i not in covered_token_indexes and keep_token and len(token.text) > 1:
            spans.append((token.i, token.i + 1, token.text))

    spans.sort(key=lambda span: span[0])
    return " ".join(text for _, _, text in spans)


@lru_cache(maxsize=1)
def load_paraphrase_model():
    print("Loading paraphrase model...")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "transformers is required for --paraphrase. Install dependencies from requirements.txt."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL_NAME)
    model.eval()
    return tokenizer, model


def generate_paraphrases(
    sentence: str,
    count: int = DEFAULT_PARAPHRASE_COUNT,
    max_length: int = 64,
) -> list[str]:
    clean_sentence = sentence.strip()
    if count <= 0 or not clean_sentence:
        return []

    tokenizer, model = load_paraphrase_model()
    input_text = f"paraphrase: {clean_sentence} </s>"
    encoded = tokenizer(input_text, return_tensors="pt", truncation=True)

    candidate_count = min(max(count * 3, count + 2), 20)
    outputs = model.generate(
        **encoded,
        max_length=max_length,
        num_beams=max(5, candidate_count),
        num_return_sequences=candidate_count,
        early_stopping=True,
    )

    paraphrases = []
    seen = {normalize_label(clean_sentence)}

    for output in outputs:
        candidate = tokenizer.decode(output, skip_special_tokens=True).strip()
        normalized_candidate = normalize_label(candidate)
        if not candidate or normalized_candidate in seen:
            continue

        seen.add(normalized_candidate)
        paraphrases.append(candidate)

        if len(paraphrases) == count:
            break

    return paraphrases


def question_text(question: JeopardyQuestion, include_category: bool) -> str:
    if include_category:
        return f"{question.category} {question.clue}"

    return question.clue


def transform_query_text(query_text: str, query_mode: str) -> str:
    if query_mode == "entity":
        return entity_query_text(query_text)

    return query_text


def question_query(question: JeopardyQuestion, query_mode: str, include_category: bool) -> str:
    return transform_query_text(question_text(question, include_category), query_mode)


@lru_cache(maxsize=1)
def load_precomputed_question_embeddings(
    path: Path = DEFAULT_QUESTION_DPR_EMBEDDINGS_PATH,
):
    return np.load(path, allow_pickle=True)


@lru_cache(maxsize=1)
def load_precomputed_natural_question_embeddings(
    path: Path = DEFAULT_NATURAL_QUESTION_DPR_EMBEDDINGS_PATH,
):
    return np.load(path, allow_pickle=True)


def get_precomputed_dense_query_embeddings(
    questions: list[JeopardyQuestion],
    include_category: bool,
) -> list[np.ndarray] | None:
    embeddings_path = DEFAULT_QUESTION_DPR_EMBEDDINGS_PATH
    if not embeddings_path.exists():
        return None

    saved = load_precomputed_question_embeddings(embeddings_path)
    if include_category:
        expected_texts = [
            category_plus_clue_text(
                {
                    "category": question.category,
                    "clue": question.clue,
                    "answer": question.answer,
                }
            )
            for question in questions
        ]
        saved_texts = saved["category_plus_clue_texts"].tolist()[: len(questions)]
        if saved_texts != expected_texts:
            return None
        return [row for row in saved["category_plus_clue_embeddings"][: len(questions)]]

    expected_texts = [
        clue_only_text(
            {
                "category": question.category,
                "clue": question.clue,
                "answer": question.answer,
            }
        )
        for question in questions
    ]
    saved_texts = saved["clue_only_texts"].tolist()[: len(questions)]
    if saved_texts != expected_texts:
        return None
    return [row for row in saved["clue_only_embeddings"][: len(questions)]]


def get_precomputed_natural_question_dense_embeddings(
    questions: list[JeopardyQuestion],
) -> tuple[list[np.ndarray], list[np.ndarray]] | tuple[None, None]:
    embeddings_path = DEFAULT_NATURAL_QUESTION_DPR_EMBEDDINGS_PATH
    if not embeddings_path.exists():
        return None, None

    saved = load_precomputed_natural_question_embeddings(embeddings_path)
    expected_categories = [question.category for question in questions]
    expected_clues = [question.clue for question in questions]
    saved_categories = saved["categories"].tolist()[: len(questions)]
    saved_clues = saved["clues"].tolist()[: len(questions)]
    if saved_categories != expected_categories or saved_clues != expected_clues:
        return None, None

    natural_question_embeddings = [
        row for row in saved["natural_questions_embeddings"][: len(questions)]
    ]
    natural_questions_concat_embeddings = [
        row for row in saved["combined_natural_questions_embeddings"][: len(questions)]
    ]
    return natural_question_embeddings, natural_questions_concat_embeddings


def warmup_retrieval_resources(
    questions: list[JeopardyQuestion],
    *,
    mode: str,
    query_mode: str,
    include_category: bool,
    weighted: bool,
    weight_equal: bool,
    weighted_cole: bool,
    paraphrase: bool,
) -> None:
    if not questions:
        return

    if weight_equal:
        active_title = True
        active_body = True
        active_first_sentence = True
        active_first_two_sentences = True
        active_first_paragraph = True
        active_redirect = True
        active_faiss = True
        active_qwen_summary_faiss = True
        active_natural_questions_avg_faiss = True
        active_natural_questions_concat_faiss = True
        active_natural_questions_avg_qwen_summary_faiss = True
        active_natural_questions_concat_qwen_summary_faiss = True
        active_category_first_sentence = True
        active_category_first_two_sentences = True
        active_year_match = weighted_cole
        active_quote_match = weighted_cole
    else:
        active_title = WEIGHTED_TITLE_WEIGHT > 0
        active_body = WEIGHTED_BODY_WEIGHT > 0
        active_first_sentence = WEIGHTED_FIRST_SENTENCE_WEIGHT > 0
        active_first_two_sentences = WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT > 0
        active_first_paragraph = WEIGHTED_FIRST_PARAGRAPH_WEIGHT > 0
        active_redirect = WEIGHTED_REDIRECT_WEIGHT > 0
        active_faiss = WEIGHTED_FAISS_WEIGHT > 0
        active_qwen_summary_faiss = WEIGHTED_QWEN_SUMMARY_FAISS_WEIGHT > 0
        active_natural_questions_avg_faiss = WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT > 0
        active_natural_questions_concat_faiss = (
            WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT > 0
        )
        active_natural_questions_avg_qwen_summary_faiss = (
            WEIGHTED_NATURAL_QUESTIONS_AVG_QWEN_SUMMARY_FAISS_WEIGHT > 0
        )
        active_natural_questions_concat_qwen_summary_faiss = (
            WEIGHTED_NATURAL_QUESTIONS_CONCAT_QWEN_SUMMARY_FAISS_WEIGHT > 0
        )
        active_category_first_sentence = WEIGHTED_CATEGORY_FIRST_SENTENCE_WEIGHT > 0
        active_category_first_two_sentences = WEIGHTED_CATEGORY_FIRST_TWO_SENTENCES_WEIGHT > 0
        active_year_match = weighted_cole and WEIGHTED_YEAR_MATCH_WEIGHT > 0
        active_quote_match = weighted_cole and WEIGHTED_QUOTE_MATCH_WEIGHT > 0

    if query_mode == "entity":
        load_entity_model()

    if paraphrase:
        load_paraphrase_model()

    if mode == "token":
        open_index()
    elif mode == "whoosh":
        open_whoosh_index()
    elif mode == "whoosh_title_body":
        open_whoosh_title_body_index()
    elif mode == "cole":
        open_whoosh_cole_index()

    if weighted or weight_equal or weighted_cole:
        if weighted_cole and (
            active_title
            or active_body
            or active_year_match
            or active_quote_match
            or active_redirect
            or active_faiss
            or active_qwen_summary_faiss
            or active_natural_questions_avg_faiss
            or active_natural_questions_concat_faiss
            or active_natural_questions_avg_qwen_summary_faiss
            or active_natural_questions_concat_qwen_summary_faiss
        ):
            open_whoosh_cole_index()
        elif not weighted_cole and (
            active_title
            or active_body
            or active_redirect
            or active_faiss
            or active_qwen_summary_faiss
            or active_natural_questions_avg_faiss
            or active_natural_questions_concat_faiss
            or active_natural_questions_avg_qwen_summary_faiss
            or active_natural_questions_concat_qwen_summary_faiss
        ):
            open_whoosh_title_body_index()
        if active_category_first_sentence or active_category_first_two_sentences:
            open_whoosh_title_body_category_index()
        if active_first_sentence:
            open_whoosh_cole_first_sentence_index()
        if active_first_two_sentences:
            open_whoosh_cole_first_two_sentences_index()
        if active_first_paragraph:
            open_whoosh_cole_first_paragraph_index()
        if active_redirect:
            open_whoosh_cole_redirect_index()
            load_redirect_lookup()

        if query_mode == "full" and (
            active_faiss
            or active_qwen_summary_faiss
            or active_natural_questions_avg_faiss
            or active_natural_questions_concat_faiss
            or active_natural_questions_avg_qwen_summary_faiss
            or active_natural_questions_concat_qwen_summary_faiss
        ):
            dense_query_embeddings = get_precomputed_dense_query_embeddings(
                questions,
                include_category=include_category,
            )
            if active_faiss and dense_query_embeddings is None:
                load_dpr_question_encoder()
        elif query_mode != "full" and (
            active_faiss
            or active_qwen_summary_faiss
            or active_natural_questions_avg_faiss
            or active_natural_questions_concat_faiss
            or active_natural_questions_avg_qwen_summary_faiss
            or active_natural_questions_concat_qwen_summary_faiss
        ):
            load_dpr_question_encoder()

        if (
            active_natural_questions_avg_faiss
            or active_natural_questions_concat_faiss
            or active_natural_questions_avg_qwen_summary_faiss
            or active_natural_questions_concat_qwen_summary_faiss
        ):
            get_precomputed_natural_question_dense_embeddings(questions)

        if (
            active_faiss
            or active_qwen_summary_faiss
            or active_natural_questions_avg_faiss
            or active_natural_questions_concat_faiss
            or active_natural_questions_avg_qwen_summary_faiss
            or active_natural_questions_concat_qwen_summary_faiss
        ):
            for variant_name in DPR_FAISS_VARIANT_NAMES:
                variant_dir = DEFAULT_DPR_FAISS_INDEX_DIR / variant_name
                if variant_dir.exists():
                    load_dpr_faiss_variant(str(variant_dir))
            for variant_name in DPR_QWEN_SUMMARY_FAISS_VARIANT_NAMES:
                variant_dir = DEFAULT_DPR_FAISS_INDEX_DIR / variant_name
                if variant_dir.exists():
                    load_dpr_faiss_variant(str(variant_dir))


def print_question_progress(
    processed_questions: int,
    total_questions: int,
    retrieval_start_time: float,
) -> None:
    elapsed_seconds = time.time() - retrieval_start_time
    cache_stats = get_cache_stats()
    print(
        f"[question-progress] Questions: {processed_questions}/{total_questions} | "
        f"elapsed: {format_elapsed_minutes(elapsed_seconds)} min | "
        f"cache misses: {cache_stats['misses']}"
    )


def question_queries(
    question: JeopardyQuestion,
    query_mode: str,
    include_category: bool,
    paraphrase: bool = False,
    paraphrase_count: int = DEFAULT_PARAPHRASE_COUNT,
) -> list[str]:
    raw_query_text = question_text(question, include_category)
    raw_queries = [raw_query_text]

    if paraphrase:
        raw_queries.extend(generate_paraphrases(raw_query_text, count=paraphrase_count))

    transformed_queries = []
    seen = set()

    for raw_query in raw_queries:
        query_text = transform_query_text(raw_query, query_mode).strip()
        normalized_query = normalize_label(query_text)
        if not query_text or normalized_query in seen:
            continue

        seen.add(normalized_query)
        transformed_queries.append(query_text)

    return transformed_queries or [""]


def run_search_batch(
    queries: list[str],
    categories: list[str] | None,
    limit: int,
    mode: str,
    weighted: bool = False,
    weight_equal: bool = False,
    weighted_cole: bool = False,
    skip_redirects: bool = False,
    dense_query_embeddings: list[np.ndarray] | None = None,
    natural_questions_embeddings: list[np.ndarray] | None = None,
    natural_questions_concat_embeddings: list[np.ndarray] | None = None,
    search_progress_every: int = 0,
) -> list[dict]:
    if weighted_cole:
        return multi_search_whoosh_weighted_cole(
            queries,
            query_categories=categories,
            limit=limit,
            skip_redirects=skip_redirects,
            dense_query_embeddings=dense_query_embeddings,
            natural_questions_embeddings=natural_questions_embeddings,
            natural_questions_concat_embeddings=natural_questions_concat_embeddings,
            progress_every=search_progress_every,
        )

    if weighted or weight_equal:
        weighted_kwargs = {
            "query_categories": categories,
            "limit": limit,
            "skip_redirects": skip_redirects,
            "dense_query_embeddings": dense_query_embeddings,
            "natural_questions_embeddings": natural_questions_embeddings,
            "natural_questions_concat_embeddings": natural_questions_concat_embeddings,
            "progress_every": search_progress_every,
        }
        if weight_equal:
            weighted_kwargs.update(
                {
                    "title_weight": 1.0,
                    "redirect_weight": 1.0,
                    "body_weight": 1.0,
                    "first_sentence_weight": 1.0,
                    "first_two_sentences_weight": 1.0,
                    "first_paragraph_weight": 1.0,
                    "faiss_weight": 1.0,
                    "qwen_summary_faiss_weight": 1.0,
                    "natural_questions_avg_faiss_weight": 1.0,
                    "natural_questions_concat_faiss_weight": 1.0,
                    "natural_questions_avg_qwen_summary_faiss_weight": 1.0,
                    "natural_questions_concat_qwen_summary_faiss_weight": 1.0,
                    "category_first_sentence_weight": 1.0,
                    "category_first_two_sentences_weight": 1.0,
                }
            )
        return multi_search_whoosh_weighted(queries, **weighted_kwargs)

    if mode == "whoosh_title_body":
        return multi_search_whoosh_title_body(
            queries,
            limit=limit,
            skip_redirects=skip_redirects,
            progress_every=search_progress_every,
        )

    if mode == "cole":
        return multi_search_whoosh_cole(
            queries,
            limit=limit,
            skip_redirects=skip_redirects,
            progress_every=search_progress_every,
        )

    if mode == "whoosh":
        return multi_search_whoosh_default(
            queries,
            limit=limit,
            skip_redirects=skip_redirects,
            progress_every=search_progress_every,
        )

    return multi_search(
        queries,
        limit=limit,
        skip_redirects=skip_redirects,
        progress_every=search_progress_every,
    )


def merge_ranked_results(searches: list[dict], limit: int) -> list[dict]:
    fused_results: dict[tuple, dict] = {}

    for search in searches:
        for rank, result in enumerate(search["results"], start=1):
            key = (
                result["source_file"],
                result["article_index"],
                normalize_label(result["title"]),
            )
            fused_result = fused_results.setdefault(
                key,
                {
                    **result,
                    "fusion_score": 0.0,
                    "best_rank": rank,
                    "matched_queries": 0,
                },
            )
            fused_result["fusion_score"] += 1.0 / (RANK_FUSION_K + rank)
            fused_result["best_rank"] = min(fused_result["best_rank"], rank)
            fused_result["matched_queries"] += 1
            fused_result["score"] = max(fused_result["score"], result["score"])

    ranked_results = sorted(
        fused_results.values(),
        key=lambda result: (
            -result["fusion_score"],
            result["best_rank"],
            -result["matched_queries"],
            -result["score"],
            normalize_label(result["title"]),
        ),
    )
    return ranked_results[:limit]


def search_questions(
    questions: list[JeopardyQuestion],
    limit: int,
    mode: str,
    query_mode: str,
    include_category: bool,
    weighted: bool = False,
    weight_equal: bool = False,
    weighted_cole: bool = False,
    skip_redirects: bool = False,
    paraphrase: bool = False,
    paraphrase_count: int = DEFAULT_PARAPHRASE_COUNT,
    paraphrase_fetch_limit: int = DEFAULT_PARAPHRASE_FETCH_LIMIT,
    progress_every: int = QUESTION_PROGRESS_EVERY,
    progress_callback=None,
) -> list[dict]:
    total_questions = len(questions)
    progress_every = max(1, progress_every)

    if not paraphrase:
        queries = [question_query(question, query_mode, include_category) for question in questions]
        categories = [question.category for question in questions]
        cache_config = final_ranking_cache_config(
            mode=mode,
            query_mode=query_mode,
            include_category=include_category,
            weighted=weighted,
            weight_equal=weight_equal,
            weighted_cole=weighted_cole,
            skip_redirects=skip_redirects,
            paraphrase=paraphrase,
            paraphrase_count=paraphrase_count,
            paraphrase_fetch_limit=paraphrase_fetch_limit,
        )
        dense_query_embeddings = None
        natural_questions_embeddings = None
        natural_questions_concat_embeddings = None
        if query_mode == "full" and (weighted or weight_equal or weighted_cole):
            dense_query_embeddings = get_precomputed_dense_query_embeddings(
                questions,
                include_category=include_category,
            )
            (
                natural_questions_embeddings,
                natural_questions_concat_embeddings,
            ) = get_precomputed_natural_question_dense_embeddings(questions)
        use_final_ranking_cache = weighted_cole
        searches: list[dict | None] = [None] * total_questions
        miss_indexes = []
        for question_index, (query, category) in enumerate(zip(queries, categories)):
            cached_results = None
            if use_final_ranking_cache:
                cached_results = load_cached_final_ranking(
                    query_text=query,
                    query_category=category,
                    limit=limit,
                    cache_config=cache_config,
                )
            if cached_results is None:
                if not use_final_ranking_cache:
                    FINAL_RANKING_CACHE_STATS["misses"] += 1
                miss_indexes.append(question_index)
            else:
                searches[question_index] = {
                    "query": query,
                    "results": cached_results,
                }

        for miss_batch_start in range(0, len(miss_indexes), progress_every):
            batch_indexes = miss_indexes[miss_batch_start : miss_batch_start + progress_every]
            batch_dense_embeddings = (
                [dense_query_embeddings[index] for index in batch_indexes]
                if dense_query_embeddings is not None
                else None
            )
            batch_natural_questions_embeddings = (
                [natural_questions_embeddings[index] for index in batch_indexes]
                if natural_questions_embeddings is not None
                else None
            )
            batch_natural_questions_concat_embeddings = (
                [natural_questions_concat_embeddings[index] for index in batch_indexes]
                if natural_questions_concat_embeddings is not None
                else None
            )
            batch_searches = run_search_batch(
                [queries[index] for index in batch_indexes],
                categories=[categories[index] for index in batch_indexes],
                limit=limit,
                mode=mode,
                weighted=weighted,
                weight_equal=weight_equal,
                weighted_cole=weighted_cole,
                skip_redirects=skip_redirects,
                dense_query_embeddings=batch_dense_embeddings,
                natural_questions_embeddings=batch_natural_questions_embeddings,
                natural_questions_concat_embeddings=batch_natural_questions_concat_embeddings,
                search_progress_every=0,
            )
            for question_index, search_result in zip(batch_indexes, batch_searches):
                searches[question_index] = search_result
                if use_final_ranking_cache:
                    store_cached_final_ranking(
                        query_text=queries[question_index],
                        query_category=categories[question_index],
                        limit=limit,
                        cache_config=cache_config,
                        results=search_result["results"],
                    )
            if progress_callback is not None:
                processed = min(
                    FINAL_RANKING_CACHE_STATS["hits"] + miss_batch_start + len(batch_indexes),
                    total_questions,
                )
                progress_callback(processed, total_questions)
        if progress_callback is not None and not miss_indexes:
            progress_callback(total_questions, total_questions)
        return [search for search in searches if search is not None]

    print("Running paraphrasing...")
    grouped_queries = []
    grouped_categories = []

    for question_number, question in enumerate(questions, start=1):
        print(f"[paraphrase] Question {question_number}/{total_questions}")
        query_group = question_queries(
            question,
            query_mode=query_mode,
            include_category=include_category,
            paraphrase=True,
            paraphrase_count=paraphrase_count,
        )
        grouped_queries.append(query_group)
        grouped_categories.append([question.category] * len(query_group))
    merged_searches = []
    for batch_start in range(0, total_questions, progress_every):
        batch_end = min(batch_start + progress_every, total_questions)
        batch_grouped_queries = grouped_queries[batch_start:batch_end]
        batch_grouped_categories = grouped_categories[batch_start:batch_end]
        flat_queries = [query for query_group in batch_grouped_queries for query in query_group]
        flat_categories = [
            category for category_group in batch_grouped_categories for category in category_group
        ]
        flat_searches = run_search_batch(
            flat_queries,
            categories=flat_categories,
            limit=max(limit, paraphrase_fetch_limit),
            mode=mode,
            weighted=weighted,
            weight_equal=weight_equal,
            weighted_cole=weighted_cole,
            skip_redirects=skip_redirects,
            search_progress_every=0,
        )
        search_offset = 0
        for query_group in batch_grouped_queries:
            group_size = len(query_group)
            search_group = flat_searches[search_offset : search_offset + group_size]
            search_offset += group_size
            merged_searches.append(
                {
                    "query": query_group[0],
                    "queries": query_group,
                    "results": merge_ranked_results(search_group, limit=limit),
                }
            )
        if progress_callback is not None:
            progress_callback(batch_end, total_questions)

    return merged_searches


def cross_encoder_query_text(question: JeopardyQuestion) -> str:
    return f"Category: {question.category}\nClue: {question.clue}"


def fallback_natural_questions_path(path: Path) -> Path:
    if path.exists():
        return path

    fallback_path = PROJECT_ROOT / "data/processed/questions_natural_qwen3_14b.json"
    if fallback_path.exists():
        return fallback_path

    return path


@lru_cache(maxsize=2)
def load_cross_encoder_natural_questions(path_value: str) -> list[dict]:
    path = fallback_natural_questions_path(Path(path_value))
    return json.loads(path.read_text(encoding="utf-8"))


def get_cross_encoder_natural_questions(
    questions: list[JeopardyQuestion],
    path: Path,
) -> list[list[str]]:
    rows = load_cross_encoder_natural_questions(str(path))
    if len(rows) < len(questions):
        raise ValueError(
            f"Natural-question file has {len(rows)} rows, but {len(questions)} questions are being evaluated."
        )

    natural_question_groups = []
    for question_index, (question, row) in enumerate(zip(questions, rows), start=1):
        if row.get("category") != question.category or row.get("clue") != question.clue:
            raise ValueError(
                "Natural-question file does not align with evaluated questions at "
                f"question {question_index}."
            )

        natural_questions = [
            str(natural_question).strip()
            for natural_question in row.get("natural_questions", [])
            if str(natural_question).strip()
        ]
        if len(natural_questions) != 5:
            raise ValueError(
                f"Question {question_index} expected 5 natural questions, found "
                f"{len(natural_questions)}."
            )
        natural_question_groups.append(natural_questions)

    return natural_question_groups


def cross_encoder_query_texts(
    question: JeopardyQuestion,
    *,
    query_mode: str,
    natural_questions: list[str] | None,
) -> list[str]:
    if query_mode == "natural_questions_avg":
        if not natural_questions:
            raise ValueError("natural_questions_avg requires generated natural questions.")
        return natural_questions

    return [cross_encoder_query_text(question)]


def cross_encoder_document_text(result: dict) -> str:
    title = result.get("title", "").strip()
    body = result.get("body", "").strip()
    if title and body:
        return f"Title: {title}\nArticle: {body}"
    if title:
        return f"Title: {title}"
    return body


@lru_cache(maxsize=1)
def get_cross_encoder_article_connection(
    db_path: Path = DEFAULT_CROSS_ENCODER_ARTICLE_DB_PATH,
) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


@lru_cache(maxsize=50000)
def fetch_cross_encoder_article_body(source_file: str, article_index: int) -> str:
    connection = get_cross_encoder_article_connection()
    row = connection.execute(
        """
        SELECT body
        FROM articles
        WHERE source_file = ? AND article_index = ?
        """,
        (source_file, article_index),
    ).fetchone()
    if row is None:
        return ""

    return row[0] or ""


def hydrate_cross_encoder_result(result: dict) -> dict:
    if result.get("body"):
        return result

    source_file = result.get("source_file")
    article_index = result.get("article_index")
    if source_file is None or article_index is None:
        return result

    body = fetch_cross_encoder_article_body(str(source_file), int(article_index))
    if not body:
        return result

    return {**result, "body": body}


def hydrate_cross_encoder_results(results: list[dict]) -> list[dict]:
    return [hydrate_cross_encoder_result(result) for result in results]


@lru_cache(maxsize=2)
def load_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "sentence-transformers is required for --cross-encoder. "
            "Install dependencies from requirements.txt."
        ) from error

    print(f"Loading cross-encoder model: {model_name}")
    return CrossEncoder(model_name)


def chunk_text_by_whitespace_tokens(text: str, chunk_token_limit: int) -> list[str]:
    clean_text = text.strip()
    if not clean_text:
        return [""]

    tokens = clean_text.split()
    if not tokens:
        return [clean_text]

    chunks = []
    for start in range(0, len(tokens), chunk_token_limit):
        token_chunk = tokens[start : start + chunk_token_limit]
        chunk_text = " ".join(token_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks or [clean_text]


def cross_encoder_chunk_token_limit(mode: str) -> int:
    if mode == "chunks_100":
        return CROSS_ENCODER_SMALL_CHUNK_TOKEN_LIMIT

    return CROSS_ENCODER_CHUNK_TOKEN_LIMIT


def cross_encoder_pair_cache_key(model_name: str, text_a: str, text_b: str) -> str:
    return build_component_cache_key(
        component_name="cross_encoder_pair_score",
        query_text=text_a,
        query_category=None,
        component_limit=1,
        index_paths=[],
        extra_config={
            "cache_schema_version": CROSS_ENCODER_CACHE_SCHEMA_VERSION,
            "model_name": model_name,
            "text_b": text_b,
            "text_b_sha256": sha256_text(text_b),
        },
    )


def load_cached_cross_encoder_score(
    model_name: str,
    text_a: str,
    text_b: str,
) -> float | None:
    cached = load_cached_component_results(cross_encoder_pair_cache_key(model_name, text_a, text_b))
    if cached is None:
        CROSS_ENCODER_CACHE_STATS["misses"] += 1
        return None

    CROSS_ENCODER_CACHE_STATS["hits"] += 1
    return float(cached[0]["score"])


def store_cached_cross_encoder_score(
    model_name: str,
    text_a: str,
    text_b: str,
    score: float,
) -> None:
    store_component_results(
        cache_key=cross_encoder_pair_cache_key(model_name, text_a, text_b),
        component_name="cross_encoder_pair_score",
        results=[{"score": score}],
    )


def score_cross_encoder_pairs(
    pairs: list[tuple[str, str]],
    *,
    model_name: str,
    batch_size: int = CROSS_ENCODER_BATCH_SIZE,
) -> list[float]:
    scores: list[float | None] = [None] * len(pairs)
    uncached_pairs = []
    uncached_indexes = []

    for pair_index, (text_a, text_b) in enumerate(pairs):
        cached_score = load_cached_cross_encoder_score(model_name, text_a, text_b)
        if cached_score is None:
            uncached_indexes.append(pair_index)
            uncached_pairs.append((text_a, text_b))
        else:
            scores[pair_index] = cached_score

    if uncached_pairs:
        model = load_cross_encoder(model_name)
        predicted_scores = model.predict(uncached_pairs, batch_size=batch_size)
        for pair_index, pair, score in zip(uncached_indexes, uncached_pairs, predicted_scores):
            score_value = float(score)
            scores[pair_index] = score_value
            store_cached_cross_encoder_score(model_name, pair[0], pair[1], score_value)

    return [float(score) for score in scores if score is not None]


def rerank_results_with_cross_encoder(
    question: JeopardyQuestion,
    results: list[dict],
    *,
    model_name: str,
    mode: str,
    query_mode: str,
    natural_questions: list[str] | None = None,
    candidate_limit: int = CROSS_ENCODER_CANDIDATE_LIMIT,
) -> list[dict]:
    candidates = hydrate_cross_encoder_results(results[:candidate_limit])
    query_texts = cross_encoder_query_texts(
        question,
        query_mode=query_mode,
        natural_questions=natural_questions,
    )
    pair_results = []
    pairs = []

    for candidate_index, result in enumerate(candidates):
        document_text = cross_encoder_document_text(result)
        if mode in {"chunks_256", "chunks_100"}:
            document_chunks = chunk_text_by_whitespace_tokens(
                document_text,
                cross_encoder_chunk_token_limit(mode),
            )
        else:
            document_chunks = [document_text]

        candidate_query_pair_indexes = []
        for query_text in query_texts:
            query_pair_indexes = []
            for chunk_text in document_chunks:
                query_pair_indexes.append(len(pairs))
                pairs.append((query_text, chunk_text))
            candidate_query_pair_indexes.append(query_pair_indexes)
        pair_results.append((candidate_index, candidate_query_pair_indexes))

    pair_scores = score_cross_encoder_pairs(pairs, model_name=model_name)
    reranked_candidates = []
    for candidate_index, candidate_query_pair_indexes in pair_results:
        query_scores = []
        for query_pair_indexes in candidate_query_pair_indexes:
            chunk_scores = [pair_scores[pair_index] for pair_index in query_pair_indexes]
            query_scores.append(max(chunk_scores) if chunk_scores else float("-inf"))
        best_score = float(np.mean(query_scores)) if query_scores else float("-inf")
        reranked_candidates.append(
            {
                **candidates[candidate_index],
                "initial_rank": candidate_index + 1,
                "initial_score": candidates[candidate_index].get("score", 0.0),
                "cross_encoder_score": best_score,
                "cross_encoder_query_count": len(query_texts),
            }
        )

    return sorted(
        reranked_candidates,
        key=lambda result: (
            -result["cross_encoder_score"],
            result["initial_rank"],
            normalize_label(result["title"]),
        ),
    )


def count_cross_encoder_pairs_for_results(
    results: list[dict],
    *,
    mode: str,
    query_count: int = 1,
    candidate_limit: int = CROSS_ENCODER_CANDIDATE_LIMIT,
) -> int:
    pair_count = 0
    for result in hydrate_cross_encoder_results(results[:candidate_limit]):
        document_text = cross_encoder_document_text(result)
        if mode in {"chunks_256", "chunks_100"}:
            pair_count += query_count * len(
                chunk_text_by_whitespace_tokens(
                    document_text,
                    cross_encoder_chunk_token_limit(mode),
                )
            )
        else:
            pair_count += query_count
    return pair_count


def rerank_searches_with_cross_encoder(
    questions: list[JeopardyQuestion],
    searches: list[dict],
    *,
    model_name: str,
    mode: str,
    query_mode: str,
    natural_question_groups: list[list[str]] | None = None,
) -> list[dict]:
    reranked_searches = []
    total = len(questions)
    print(
        f"Running cross-encoder reranking over top {CROSS_ENCODER_CANDIDATE_LIMIT} candidates..."
    )
    for question_number, (question, search_result) in enumerate(
        zip(questions, searches),
        start=1,
    ):
        natural_questions = (
            natural_question_groups[question_number - 1]
            if natural_question_groups is not None
            else None
        )
        query_count = 5 if query_mode == "natural_questions_avg" else 1
        docs_count = min(len(search_result["results"]), CROSS_ENCODER_CANDIDATE_LIMIT)
        pair_count = count_cross_encoder_pairs_for_results(
            search_result["results"],
            mode=mode,
            query_count=query_count,
        )
        hits_before = CROSS_ENCODER_CACHE_STATS["hits"]
        misses_before = CROSS_ENCODER_CACHE_STATS["misses"]
        if CROSS_ENCODER_PROGRESS_EVERY and (
            question_number % CROSS_ENCODER_PROGRESS_EVERY == 0
            or question_number == total
        ):
            print(
                f"[cross-encoder] Starting question {question_number}/{total} | "
                f"docs: {docs_count} | pairs: {pair_count}",
                flush=True,
            )
        reranked_searches.append(
            {
                **search_result,
                "initial_results": search_result["results"],
                "results": rerank_results_with_cross_encoder(
                    question,
                    search_result["results"],
                    model_name=model_name,
                    mode=mode,
                    query_mode=query_mode,
                    natural_questions=natural_questions,
                ),
            }
        )
        if (
            CROSS_ENCODER_PROGRESS_EVERY
            and question_number % CROSS_ENCODER_PROGRESS_EVERY == 0
        ) or question_number == total:
            question_hits = CROSS_ENCODER_CACHE_STATS["hits"] - hits_before
            question_misses = CROSS_ENCODER_CACHE_STATS["misses"] - misses_before
            print(
                f"[cross-encoder] Question {question_number}/{total} | "
                f"docs: {docs_count} | pairs: {pair_count} | "
                f"cache hits: {question_hits} | cache misses: {question_misses}",
                flush=True,
            )

    return reranked_searches


def min_max_normalize_values(values: list[float]) -> list[float]:
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        normalized_value = 1.0 if max_value > 0 else 0.0
        return [normalized_value for _ in values]

    value_range = max_value - min_value
    return [(value - min_value) / value_range for value in values]


def fuse_initial_and_cross_encoder_results(
    results: list[dict],
    *,
    initial_weight: float = CROSS_ENCODER_FUSION_INITIAL_WEIGHT,
    cross_encoder_weight: float = CROSS_ENCODER_FUSION_CROSS_ENCODER_WEIGHT,
) -> list[dict]:
    if not results:
        return []

    initial_scores = [float(result.get("initial_score", result.get("score", 0.0))) for result in results]
    cross_encoder_scores = [float(result.get("cross_encoder_score", 0.0)) for result in results]
    normalized_initial_scores = min_max_normalize_values(initial_scores)
    normalized_cross_encoder_scores = min_max_normalize_values(cross_encoder_scores)

    fused_results = []
    for result, initial_score, cross_encoder_score in zip(
        results,
        normalized_initial_scores,
        normalized_cross_encoder_scores,
    ):
        fused_score = initial_weight * initial_score + cross_encoder_weight * cross_encoder_score
        fused_results.append(
            {
                **result,
                "normalized_initial_score": initial_score,
                "normalized_cross_encoder_score": cross_encoder_score,
                "fused_stage2_score": fused_score,
            }
        )

    return sorted(
        fused_results,
        key=lambda result: (
            -result["fused_stage2_score"],
            -result["normalized_cross_encoder_score"],
            -result["normalized_initial_score"],
            result["initial_rank"],
            normalize_label(result["title"]),
        ),
    )


def fuse_searches_with_cross_encoder(
    searches: list[dict],
    *,
    initial_weight: float = CROSS_ENCODER_FUSION_INITIAL_WEIGHT,
    cross_encoder_weight: float = CROSS_ENCODER_FUSION_CROSS_ENCODER_WEIGHT,
) -> list[dict]:
    return [
        {
            **search,
            "cross_encoder_results": search["results"],
            "results": fuse_initial_and_cross_encoder_results(
                search["results"],
                initial_weight=initial_weight,
                cross_encoder_weight=cross_encoder_weight,
            ),
        }
        for search in searches
    ]


def compute_metrics_from_searches(
    questions: list[JeopardyQuestion],
    searches: list[dict],
    top_k_values: list[int],
    print_question_results: bool = False,
) -> dict[int, float]:
    correct_counts = {k: 0 for k in top_k_values}
    grouped_questions_by_k = {k: [] for k in top_k_values}
    unmatched_questions = []

    for question, search_result in zip(questions, searches):
        answers = valid_answers(question.answer)
        results = search_result["results"]
        answer_rank = first_answer_rank(results, answers)
        matched_k = first_correct_k_from_rank(answer_rank, top_k_values)

        if matched_k is not None:
            grouped_questions_by_k[matched_k].append((question, results))
        else:
            unmatched_questions.append((question, results))

        for k in correct_at_k_values(results, answers, top_k_values):
            correct_counts[k] += 1

    if print_question_results and (any(grouped_questions_by_k.values()) or unmatched_questions):
        print("Question results grouped by first Top-k:")
        for k in top_k_values:
            grouped_questions = grouped_questions_by_k[k]
            if not grouped_questions:
                continue

            print(f"Top-{k}:")
            for question, results in grouped_questions:
                top_titles = [result["title"] for result in results[:QUESTION_RESULTS_PREVIEW_LIMIT]]
                print(f"  Category: {question.category}")
                print(f"  Clue: {question.clue}")
                print(f"  Answer: {question.answer}")
                print(f"  Retrieved titles: {top_titles}")

        if unmatched_questions:
            print("No match in evaluated Top-k range:")
            for question, results in unmatched_questions:
                top_titles = [result["title"] for result in results[:QUESTION_RESULTS_PREVIEW_LIMIT]]
                print(f"  Category: {question.category}")
                print(f"  Clue: {question.clue}")
                print(f"  Answer: {question.answer}")
                print(f"  Retrieved titles: {top_titles}")

    total = len(questions)
    return {k: correct_counts[k] / total for k in top_k_values}


def evaluate_questions(
    questions: list[JeopardyQuestion],
    mode: str = "token",
    query_mode: str = "full",
    include_category: bool = False,
    weighted: bool = False,
    weight_equal: bool = False,
    weighted_cole: bool = False,
    skip_redirects: bool = False,
    paraphrase: bool = False,
    paraphrase_count: int = DEFAULT_PARAPHRASE_COUNT,
    paraphrase_fetch_limit: int = DEFAULT_PARAPHRASE_FETCH_LIMIT,
    print_question_results: bool = False,
    top_k_values: list[int] = TOP_K_VALUES,
    cross_encoder: bool = False,
    cross_encoder_mode: str = "chunks_256",
    cross_encoder_query_mode: str = "category_clue",
    cross_encoder_natural_questions_path: Path = DEFAULT_CROSS_ENCODER_NATURAL_QUESTIONS_PATH,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL_NAME,
    progress_every: int = QUESTION_PROGRESS_EVERY,
) -> dict[int, float] | dict[str, dict[int, float]]:
    max_k = max(top_k_values)
    reset_cache_stats()
    reset_stage_cache_stats()
    retrieval_start_time = time.time()
    print(f"Question retrieval start time: {format_timestamp(retrieval_start_time)}")
    searches = search_questions(
        questions,
        limit=max_k,
        mode=mode,
        query_mode=query_mode,
        include_category=include_category,
        weighted=weighted,
        weight_equal=weight_equal,
        weighted_cole=weighted_cole,
        skip_redirects=skip_redirects,
        paraphrase=paraphrase,
        paraphrase_count=paraphrase_count,
        paraphrase_fetch_limit=paraphrase_fetch_limit,
        progress_every=progress_every,
        progress_callback=lambda processed, total: print_question_progress(
            processed,
            total,
            retrieval_start_time,
        ),
    )
    retrieval_end_time = time.time()
    cache_stats = get_cache_stats()
    print(f"Question retrieval end time: {format_timestamp(retrieval_end_time)}")
    print(
        f"Total retrieval time: {format_elapsed_minutes(retrieval_end_time - retrieval_start_time)} min"
    )
    print(f"Total cache hits: {cache_stats['hits']}")
    print(f"Final ranking cache hits: {FINAL_RANKING_CACHE_STATS['hits']}")
    print(f"Final ranking cache misses: {FINAL_RANKING_CACHE_STATS['misses']}")

    initial_metrics = compute_metrics_from_searches(
        questions,
        searches,
        top_k_values,
        print_question_results=print_question_results and not cross_encoder,
    )
    if not cross_encoder:
        return initial_metrics

    natural_question_groups = None
    if cross_encoder_query_mode == "natural_questions_avg":
        natural_question_groups = get_cross_encoder_natural_questions(
            questions,
            cross_encoder_natural_questions_path,
        )

    cross_encoder_start_time = time.time()
    reranked_searches = rerank_searches_with_cross_encoder(
        questions,
        searches,
        model_name=cross_encoder_model,
        mode=cross_encoder_mode,
        query_mode=cross_encoder_query_mode,
        natural_question_groups=natural_question_groups,
    )
    cross_encoder_end_time = time.time()
    print(
        "Cross-encoder reranking time: "
        f"{format_elapsed_minutes(cross_encoder_end_time - cross_encoder_start_time)} min"
    )
    print(f"Cross-encoder cache hits: {CROSS_ENCODER_CACHE_STATS['hits']}")
    print(f"Cross-encoder cache misses: {CROSS_ENCODER_CACHE_STATS['misses']}")
    cross_encoder_metrics = compute_metrics_from_searches(
        questions,
        reranked_searches,
        CROSS_ENCODER_TOP_K_VALUES,
        print_question_results=print_question_results,
    )
    fused_searches = fuse_searches_with_cross_encoder(reranked_searches)
    fused_metrics = compute_metrics_from_searches(
        questions,
        fused_searches,
        CROSS_ENCODER_TOP_K_VALUES,
        print_question_results=False,
    )
    return {
        "initial": initial_metrics,
        "cross_encoder": cross_encoder_metrics,
        "stage2_fused": fused_metrics,
    }


def print_metrics(
    metrics_by_stage: dict[int, float] | dict[str, dict[int, float]],
    total: int,
    elapsed: float,
    mode: str,
    query_mode: str,
    include_category: bool,
    weighted: bool,
    weight_equal: bool,
    weighted_cole: bool,
    skip_redirects: bool,
    paraphrase: bool,
    paraphrase_count: int,
    paraphrase_fetch_limit: int,
    cross_encoder: bool = False,
    cross_encoder_mode: str = "chunks_256",
    cross_encoder_query_mode: str = "category_clue",
    cross_encoder_natural_questions_path: Path = DEFAULT_CROSS_ENCODER_NATURAL_QUESTIONS_PATH,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL_NAME,
) -> None:
    active_weights = active_weight_config(weight_equal, weighted, weighted_cole)
    if "initial" not in metrics_by_stage:
        metrics_by_stage = {"initial": metrics_by_stage}

    print(f"Mode: {mode}")
    print(f"Query mode: {query_mode}")
    print(f"Include category: {include_category}")
    print(f"Weighted: {weighted}")
    print(f"Weight equal: {weight_equal}")
    print(f"Weighted cole: {weighted_cole}")
    print(f"Skip redirects: {skip_redirects}")
    print(f"Paraphrase: {paraphrase}")
    print(f"Cross encoder: {cross_encoder}")
    if cross_encoder:
        print(f"Cross encoder mode: {cross_encoder_mode}")
        print(f"Cross encoder query mode: {cross_encoder_query_mode}")
        if cross_encoder_query_mode == "natural_questions_avg":
            print(f"Cross encoder natural questions: {cross_encoder_natural_questions_path}")
        print(f"Cross encoder model: {cross_encoder_model}")
    if paraphrase:
        print(f"Paraphrase count: {paraphrase_count}")
        print(f"Paraphrase fetch limit: {paraphrase_fetch_limit}")
    if active_weights:
        print("Weights:")
        for weight_name, weight_value in active_weights.items():
            print(f"  {weight_name}: {weight_value}")
    print(f"Questions: {total}")
    print(f"Time: {elapsed:.2f}s")

    print("Initial ranking accuracy:")
    for k, value in metrics_by_stage["initial"].items():
        print(f"  Top-{k}: {value:.4f}")

    if "cross_encoder" in metrics_by_stage:
        print("Cross-encoder ranking accuracy:")
        for k, value in metrics_by_stage["cross_encoder"].items():
            print(f"  Top-{k}: {value:.4f}")

    if "stage2_fused" in metrics_by_stage:
        print(
            "Stage-2 fused ranking accuracy "
            f"(initial={CROSS_ENCODER_FUSION_INITIAL_WEIGHT}, "
            f"cross_encoder={CROSS_ENCODER_FUSION_CROSS_ENCODER_WEIGHT}):"
        )
        for k, value in metrics_by_stage["stage2_fused"].items():
            print(f"  Top-{k}: {value:.4f}")


def main() -> None:
    args = parse_args()
    questions_path = Path(DEFAULT_QUESTIONS_JSON_PATH)
    questions = load_questions(questions_path)[:100]
    reset_stage_cache_stats()

    warmup_retrieval_resources(
        questions,
        mode=args.mode,
        query_mode=args.query_mode,
        include_category=args.include_category,
        weighted=args.weighted,
        weight_equal=args.weight_equal,
        weighted_cole=args.weighted_cole,
        paraphrase=args.paraphrase,
    )

    start_time = time.time()
    metrics = evaluate_questions(
        questions,
        mode=args.mode,
        query_mode=args.query_mode,
        include_category=args.include_category,
        weighted=args.weighted,
        weight_equal=args.weight_equal,
        weighted_cole=args.weighted_cole,
        skip_redirects=args.skip_redirects,
        paraphrase=args.paraphrase,
        paraphrase_count=args.paraphrase_count,
        paraphrase_fetch_limit=args.paraphrase_fetch_limit,
        print_question_results=args.print_question_results,
        top_k_values=TOP_K_VALUES,
        progress_every=QUESTION_PROGRESS_EVERY,
        cross_encoder=args.cross_encoder,
        cross_encoder_mode=args.cross_encoder_mode,
        cross_encoder_query_mode=args.cross_encoder_query_mode,
        cross_encoder_natural_questions_path=args.cross_encoder_natural_questions,
        cross_encoder_model=args.cross_encoder_model,
    )
    elapsed = time.time() - start_time

    print_metrics(
        metrics,
        total=len(questions),
        elapsed=elapsed,
        mode=args.mode,
        query_mode=args.query_mode,
        include_category=args.include_category,
        weighted=args.weighted,
        weight_equal=args.weight_equal,
        weighted_cole=args.weighted_cole,
        skip_redirects=args.skip_redirects,
        paraphrase=args.paraphrase,
        paraphrase_count=args.paraphrase_count,
        paraphrase_fetch_limit=args.paraphrase_fetch_limit,
        cross_encoder=args.cross_encoder,
        cross_encoder_mode=args.cross_encoder_mode,
        cross_encoder_query_mode=args.cross_encoder_query_mode,
        cross_encoder_natural_questions_path=args.cross_encoder_natural_questions,
        cross_encoder_model=args.cross_encoder_model,
    )


if __name__ == "__main__":
    main()
