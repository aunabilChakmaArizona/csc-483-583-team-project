"""Precompute DPR embeddings for generated natural questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from src.processor_question_dpr_embeddings import (
        DEFAULT_MODEL_NAME,
        encode_question_texts,
    )
except ModuleNotFoundError:
    from processor_question_dpr_embeddings import (
        DEFAULT_MODEL_NAME,
        encode_question_texts,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data/processed/questions_natural_qwen3_8b.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/processed/question_dpr_embeddings_qwen3_8b.npz"
EXPECTED_QUESTION_COUNT = 5


def load_natural_question_rows(path: Path = DEFAULT_INPUT_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_natural_question_rows(rows: list[dict]) -> None:
    for row_index, row in enumerate(rows):
        natural_questions = row.get("natural_questions", [])
        if len(natural_questions) != EXPECTED_QUESTION_COUNT:
            raise ValueError(
                f"Row {row_index} expected {EXPECTED_QUESTION_COUNT} natural questions, "
                f"found {len(natural_questions)}."
            )


def combined_natural_questions_text(natural_questions: list[str]) -> str:
    return " ".join(question.strip() for question in natural_questions if question.strip())


def materialize_natural_question_dpr_embeddings(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
    max_length: int = 512,
) -> int:
    rows = load_natural_question_rows(input_path)
    validate_natural_question_rows(rows)

    flat_natural_question_texts = [
        natural_question
        for row in rows
        for natural_question in row["natural_questions"]
    ]
    combined_texts = [
        combined_natural_questions_text(row["natural_questions"])
        for row in rows
    ]

    flat_natural_question_embeddings = encode_question_texts(
        flat_natural_question_texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )
    combined_embeddings = encode_question_texts(
        combined_texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )

    natural_question_embeddings = flat_natural_question_embeddings.reshape(
        len(rows),
        EXPECTED_QUESTION_COUNT,
        flat_natural_question_embeddings.shape[1],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        question_indexes=np.array([row["question_index"] for row in rows], dtype=np.int32),
        categories=np.array([row["category"] for row in rows], dtype=object),
        clues=np.array([row["clue"] for row in rows], dtype=object),
        answers=np.array([row.get("answer", "") for row in rows], dtype=object),
        natural_questions_texts=np.array(
            [row["natural_questions"] for row in rows],
            dtype=object,
        ),
        natural_questions_embeddings=natural_question_embeddings.astype(np.float32),
        combined_natural_questions_texts=np.array(combined_texts, dtype=object),
        combined_natural_questions_embeddings=combined_embeddings.astype(np.float32),
        source_model_names=np.array([row.get("model_name", "") for row in rows], dtype=object),
        dpr_model_name=np.array(model_name, dtype=object),
    )
    print(f"[processor_natural_question_dpr_embeddings] Wrote embeddings to {output_path}")
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute DPR embeddings for generated natural questions plus one combined "
            "question per clue."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input natural-questions JSON path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output .npz path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"DPR question encoder model (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding (default: 64)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length before truncation (default: 512)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    total = materialize_natural_question_dpr_embeddings(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"Stored embeddings for {total} question rows")
