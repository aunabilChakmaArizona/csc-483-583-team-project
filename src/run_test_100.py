"""Evaluate search quality on the first 100 Jeopardy questions."""

# python -m src.run_test_100 --mode whoosh

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time

try:
    from src.processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from src.search import multi_search, multi_search_whoosh_default
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from search import multi_search, multi_search_whoosh_default


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
        choices=["token", "whoosh"],
        default="token",
        help="token uses processor3 tokens; whoosh uses Whoosh's default analyzer index.",
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


def search_questions(questions: list[JeopardyQuestion], limit: int, mode: str) -> list[dict]:
    queries = [question.clue for question in questions]

    if mode == "whoosh":
        return multi_search_whoosh_default(queries, limit=limit)

    return multi_search(queries, limit=limit)


def evaluate_questions(
    questions: list[JeopardyQuestion],
    mode: str = "token",
    top_k_values: list[int] = TOP_K_VALUES,
) -> dict[int, float]:
    max_k = max(top_k_values)
    searches = search_questions(questions, limit=max_k, mode=mode)
    correct_counts = {k: 0 for k in top_k_values}

    for question, search_result in zip(questions, searches):
        answers = valid_answers(question.answer)
        results = search_result["results"]

        for k in top_k_values:
            if is_correct_at_k(results, answers, k):
                correct_counts[k] += 1

    total = len(questions)
    return {k: correct_counts[k] / total for k in top_k_values}


def print_metrics(metrics: dict[int, float], total: int, elapsed: float, mode: str) -> None:
    print(f"Mode: {mode}")
    print(f"Questions: {total}")
    print(f"Time: {elapsed:.2f}s")

    for k in TOP_K_VALUES:
        print(f"Top-{k} accuracy: {metrics[k]:.4f}")


def main() -> None:
    args = parse_args()
    questions_path = Path(DEFAULT_QUESTIONS_JSON_PATH)
    questions = load_questions(questions_path)[:100]

    start_time = time.time()
    metrics = evaluate_questions(questions, mode=args.mode)
    elapsed = time.time() - start_time

    print_metrics(metrics, total=len(questions), elapsed=elapsed, mode=args.mode)


if __name__ == "__main__":
    main()
