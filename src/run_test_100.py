"""Evaluate search quality on the first 100 Jeopardy questions."""

# python -m src.run_test_100 --mode whoosh --query-mode entity

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
import time

try:
    from src.processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from src.search import (
        multi_search,
        multi_search_whoosh_cole,
        multi_search_whoosh_default,
        multi_search_whoosh_title_body,
    )
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from search import (
        multi_search,
        multi_search_whoosh_cole,
        multi_search_whoosh_default,
        multi_search_whoosh_title_body,
    )


TOP_K_VALUES = [1, 5, 10, 20, 50, 100]
DEFAULT_PARAPHRASE_COUNT = 4
DEFAULT_PARAPHRASE_FETCH_LIMIT = 1000
PARAPHRASE_MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"
RANK_FUSION_K = 60
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class JeopardyQuestion:
    category: str
    clue: str
    answer: str


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
    return parser.parse_args()


def load_questions(path: Path = DEFAULT_QUESTIONS_JSON_PATH) -> list[JeopardyQuestion]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [JeopardyQuestion(**row) for row in rows]


def normalize_label(text: str) -> str:
    text = text.casefold().strip()
    return WHITESPACE_RE.sub(" ", text)


def valid_answers(answer: str) -> set[str]:
    return {normalize_label(part) for part in answer.split("|") if part.strip()}


def is_correct_at_k(results: list[dict], answers: set[str], k: int) -> bool:
    top_titles = {normalize_label(result["title"]) for result in results[:k]}
    return bool(top_titles & answers)


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
    limit: int,
    mode: str,
) -> list[dict]:
    if mode == "whoosh_title_body":
        return multi_search_whoosh_title_body(queries, limit=limit)

    if mode == "cole":
        return multi_search_whoosh_cole(queries, limit=limit)

    if mode == "whoosh":
        return multi_search_whoosh_default(queries, limit=limit)

    return multi_search(queries, limit=limit)


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
    paraphrase: bool = False,
    paraphrase_count: int = DEFAULT_PARAPHRASE_COUNT,
    paraphrase_fetch_limit: int = DEFAULT_PARAPHRASE_FETCH_LIMIT,
) -> list[dict]:
    if not paraphrase:
        queries = [question_query(question, query_mode, include_category) for question in questions]
        return run_search_batch(queries, limit=limit, mode=mode)

    print("Running paraphrasing...")
    total_questions = len(questions)
    grouped_queries = []

    for question_number, question in enumerate(questions, start=1):
        print(f"[paraphrase] Question {question_number}/{total_questions}")
        grouped_queries.append(
            question_queries(
            question,
            query_mode=query_mode,
            include_category=include_category,
            paraphrase=True,
            paraphrase_count=paraphrase_count,
        )
        )
    flat_queries = [query for query_group in grouped_queries for query in query_group]
    flat_searches = run_search_batch(
        flat_queries,
        limit=max(limit, paraphrase_fetch_limit),
        mode=mode,
    )

    merged_searches = []
    search_offset = 0

    for query_group in grouped_queries:
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

    return merged_searches


def evaluate_questions(
    questions: list[JeopardyQuestion],
    mode: str = "token",
    query_mode: str = "full",
    include_category: bool = False,
    paraphrase: bool = False,
    paraphrase_count: int = DEFAULT_PARAPHRASE_COUNT,
    paraphrase_fetch_limit: int = DEFAULT_PARAPHRASE_FETCH_LIMIT,
    top_k_values: list[int] = TOP_K_VALUES,
) -> dict[int, float]:
    max_k = max(top_k_values)
    searches = search_questions(
        questions,
        limit=max_k,
        mode=mode,
        query_mode=query_mode,
        include_category=include_category,
        paraphrase=paraphrase,
        paraphrase_count=paraphrase_count,
        paraphrase_fetch_limit=paraphrase_fetch_limit,
    )
    correct_counts = {k: 0 for k in top_k_values}

    for question, search_result in zip(questions, searches):
        answers = valid_answers(question.answer)
        results = search_result["results"]

        for k in top_k_values:
            if is_correct_at_k(results, answers, k):
                correct_counts[k] += 1

    total = len(questions)
    return {k: correct_counts[k] / total for k in top_k_values}


def print_metrics(
    metrics: dict[int, float],
    total: int,
    elapsed: float,
    mode: str,
    query_mode: str,
    include_category: bool,
    paraphrase: bool,
    paraphrase_count: int,
    paraphrase_fetch_limit: int,
) -> None:
    print(f"Mode: {mode}")
    print(f"Query mode: {query_mode}")
    print(f"Include category: {include_category}")
    print(f"Paraphrase: {paraphrase}")
    if paraphrase:
        print(f"Paraphrase count: {paraphrase_count}")
        print(f"Paraphrase fetch limit: {paraphrase_fetch_limit}")
    print(f"Questions: {total}")
    print(f"Time: {elapsed:.2f}s")

    for k in TOP_K_VALUES:
        print(f"Top-{k} accuracy: {metrics[k]:.4f}")


def main() -> None:
    args = parse_args()
    questions_path = Path(DEFAULT_QUESTIONS_JSON_PATH)
    questions = load_questions(questions_path)[:100]

    start_time = time.time()
    metrics = evaluate_questions(
        questions,
        mode=args.mode,
        query_mode=args.query_mode,
        include_category=args.include_category,
        paraphrase=args.paraphrase,
        paraphrase_count=args.paraphrase_count,
        paraphrase_fetch_limit=args.paraphrase_fetch_limit,
    )
    elapsed = time.time() - start_time

    print_metrics(
        metrics,
        total=len(questions),
        elapsed=elapsed,
        mode=args.mode,
        query_mode=args.query_mode,
        include_category=args.include_category,
        paraphrase=args.paraphrase,
        paraphrase_count=args.paraphrase_count,
        paraphrase_fetch_limit=args.paraphrase_fetch_limit,
    )


if __name__ == "__main__":
    main()
