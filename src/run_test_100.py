"""Evaluate search quality on the first 100 Jeopardy questions."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time

try:
    from src.processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from src.search import multi_search
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from search import multi_search


TOP_K_VALUES = [1, 5, 10, 20, 50, 100]
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class JeopardyQuestion:
    category: str
    clue: str
    answer: str


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


def evaluate_questions(
    questions: list[JeopardyQuestion],
    top_k_values: list[int] = TOP_K_VALUES,
) -> dict[int, float]:
    max_k = max(top_k_values)
    searches = multi_search([question.clue for question in questions], limit=max_k)
    correct_counts = {k: 0 for k in top_k_values}

    for question, search_result in zip(questions, searches):
        answers = valid_answers(question.answer)
        results = search_result["results"]

        for k in top_k_values:
            if is_correct_at_k(results, answers, k):
                correct_counts[k] += 1

    total = len(questions)
    return {k: correct_counts[k] / total for k in top_k_values}


def print_metrics(metrics: dict[int, float], total: int, elapsed: float) -> None:
    print(f"Questions: {total}")
    print(f"Time: {elapsed:.2f}s")

    for k in TOP_K_VALUES:
        print(f"Top-{k} accuracy: {metrics[k]:.4f}")


def main() -> None:
    questions_path = Path(DEFAULT_QUESTIONS_JSON_PATH)
    questions = load_questions(questions_path)[:100]

    start_time = time.time()
    metrics = evaluate_questions(questions)
    elapsed = time.time() - start_time

    print_metrics(metrics, total=len(questions), elapsed=elapsed)


if __name__ == "__main__":
    main()
