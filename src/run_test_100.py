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
    from src.search import multi_search, multi_search_whoosh_default, multi_search_whoosh_title_body
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from search import multi_search, multi_search_whoosh_default, multi_search_whoosh_title_body


TOP_K_VALUES = [1, 5, 10, 20, 50, 100]
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
        choices=["token", "whoosh", "whoosh_title_body"],
        default="token",
        help=(
            "token uses processor3 tokens; whoosh searches cleaned body; "
            "whoosh_title_body searches cleaned title and body."
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


def question_text(question: JeopardyQuestion, include_category: bool) -> str:
    if include_category:
        return f"{question.category} {question.clue}"

    return question.clue


def question_query(question: JeopardyQuestion, query_mode: str, include_category: bool) -> str:
    query_text = question_text(question, include_category)
    if query_mode == "entity":
        return entity_query_text(query_text)

    return query_text


def search_questions(
    questions: list[JeopardyQuestion],
    limit: int,
    mode: str,
    query_mode: str,
    include_category: bool,
) -> list[dict]:
    queries = [question_query(question, query_mode, include_category) for question in questions]

    if mode == "whoosh_title_body":
        return multi_search_whoosh_title_body(queries, limit=limit)

    if mode == "whoosh":
        return multi_search_whoosh_default(queries, limit=limit)

    return multi_search(queries, limit=limit)


def evaluate_questions(
    questions: list[JeopardyQuestion],
    mode: str = "token",
    query_mode: str = "full",
    include_category: bool = False,
    top_k_values: list[int] = TOP_K_VALUES,
) -> dict[int, float]:
    max_k = max(top_k_values)
    searches = search_questions(
        questions,
        limit=max_k,
        mode=mode,
        query_mode=query_mode,
        include_category=include_category,
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
) -> None:
    print(f"Mode: {mode}")
    print(f"Query mode: {query_mode}")
    print(f"Include category: {include_category}")
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
    )
    elapsed = time.time() - start_time

    print_metrics(
        metrics,
        total=len(questions),
        elapsed=elapsed,
        mode=args.mode,
        query_mode=args.query_mode,
        include_category=args.include_category,
    )


if __name__ == "__main__":
    main()
