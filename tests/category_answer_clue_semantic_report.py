"""Report category matches using answer/article name plus clue text."""

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from src.processor_category_semantic import (
        load_sentence_transformer,
        load_unique_categories,
    )
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from processor_category_semantic import (
        load_sentence_transformer,
        load_unique_categories,
    )


QUESTION_LIMIT = 100
TOP_K_VALUES = (1, 3, 5, 10)
REPORT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def load_questions(path: Path = DEFAULT_QUESTIONS_JSON_PATH, limit: int = QUESTION_LIMIT) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))[:limit]


def answer_variants(answer: str) -> list[str]:
    variants = []
    seen = set()

    for part in answer.split("|"):
        cleaned = part.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        variants.append(cleaned)

    return variants


def encode_texts(model, texts: list[str]) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=min(len(texts), 128),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def top_categories_for_text(
    text: str,
    categories: list[str],
    category_embeddings: np.ndarray,
    model,
    top_k: int = 5,
) -> list[dict]:
    text_embedding = encode_texts(model, [text])[0]
    scores = category_embeddings @ text_embedding
    top_indexes = np.argsort(scores)[-top_k:][::-1]
    return [
        {
            "category": categories[int(index)],
            "score": float(scores[int(index)]),
        }
        for index in top_indexes
    ]


def answer_clue_text(answer_variant: str, clue: str) -> str:
    return f"{answer_variant}. {clue}".strip()


def evaluate_answer_clue_category_matches(
    questions_path: Path = DEFAULT_QUESTIONS_JSON_PATH,
    limit: int = QUESTION_LIMIT,
) -> dict:
    questions = load_questions(questions_path, limit=limit)
    categories = load_unique_categories(questions_path, limit=limit)
    model = load_sentence_transformer(REPORT_MODEL_NAME)
    category_embeddings = encode_texts(model, categories)

    correct_counts = {k: 0 for k in TOP_K_VALUES}
    detailed_results = []

    for index, question in enumerate(questions, start=1):
        actual_category = question["category"]
        clue = question["clue"]
        variants = answer_variants(question["answer"])
        variant_matches = []

        for variant in variants:
            semantic_text = answer_clue_text(variant, clue)
            predictions = top_categories_for_text(
                semantic_text,
                categories,
                category_embeddings,
                model,
                top_k=max(TOP_K_VALUES),
            )
            top_categories = [item["category"] for item in predictions]
            variant_matches.append(
                {
                    "answer_variant": variant,
                    "semantic_text": semantic_text,
                    "predictions": predictions,
                    "top1_category": predictions[0]["category"],
                    "top1_score": predictions[0]["score"],
                    "hits": {k: actual_category in top_categories[:k] for k in TOP_K_VALUES},
                }
            )

        question_hits = {
            k: any(item["hits"][k] for item in variant_matches)
            for k in TOP_K_VALUES
        }
        for k in TOP_K_VALUES:
            if question_hits[k]:
                correct_counts[k] += 1

        detailed_results.append(
            {
                "question_number": index,
                "actual_category": actual_category,
                "clue": clue,
                "answer": question["answer"],
                "hits": question_hits,
                "variants": variant_matches,
            }
        )

    return {
        "model_name": REPORT_MODEL_NAME,
        "question_limit": limit,
        "category_count": len(categories),
        "correct_counts": correct_counts,
        "accuracies": {
            k: (correct_counts[k] / len(questions) if questions else 0.0)
            for k in TOP_K_VALUES
        },
        "results": detailed_results,
    }


def print_report(report: dict) -> None:
    print(f"Model: {report['model_name']}")
    print(f"Questions: {report['question_limit']}")
    print(f"Unique categories: {report['category_count']}")
    for k in TOP_K_VALUES:
        correct = report["correct_counts"][k]
        accuracy = report["accuracies"][k]
        print(f"Top-{k} matches: {correct}")
        print(f"Top-{k} accuracy: {accuracy:.4f}")
    print()

    for row in report["results"]:
        best_variant = max(row["variants"], key=lambda item: item["top1_score"])
        status = (
            "TOP1"
            if row["hits"][1]
            else "TOP3"
            if row["hits"][3]
            else "TOP5"
            if row["hits"][5]
            else "TOP10"
            if row["hits"][10]
            else "MISS"
        )
        top_predictions = ", ".join(
            f"{item['category']} ({item['score']:.4f})"
            for item in best_variant["predictions"]
        )
        print(
            f"[{status}] Q{row['question_number']:03d} | "
            f"Actual: {row['actual_category']} | "
            f"Best top-1: {best_variant['top1_category']} | "
            f"Actual article: {row['answer']} | "
            f"Best variant: {best_variant['answer_variant']} | "
            f"Clue: {row['clue']} | "
            f"Top-{max(TOP_K_VALUES)}: {top_predictions}"
        )


def main() -> None:
    report = evaluate_answer_clue_category_matches()
    print_report(report)


if __name__ == "__main__":
    main()
