"""Precompute DPR question embeddings for Jeopardy clues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from src.processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_QUESTIONS_JSON_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/processed/question_dpr_embeddings.npz"
DEFAULT_MODEL_NAME = "facebook/dpr-question_encoder-single-nq-base"


def load_questions(path: Path = DEFAULT_QUESTIONS_JSON_PATH) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def clue_only_text(question: dict[str, str]) -> str:
    clue = question["clue"].strip()
    return f"The clue is: {clue}"


def category_plus_clue_text(question: dict[str, str]) -> str:
    category = question["category"].strip()
    clue = question["clue"].strip()
    return f"The category is: {category}. The clue is: {clue}"


def encode_question_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
    max_length: int = 512,
) -> np.ndarray:
    import torch
    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
    model = DPRQuestionEncoder.from_pretrained(model_name, use_safetensors=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    embedding_batches = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded_inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}

        with torch.no_grad():
            outputs = model(**encoded_inputs).pooler_output

        embedding_batches.append(outputs.detach().cpu().numpy().astype("float32"))
        print(f"[processor_question_dpr_embeddings] Encoded {min(start + batch_size, len(texts))}/{len(texts)}")

    return np.concatenate(embedding_batches, axis=0)


def materialize_question_dpr_embeddings(
    questions_path: Path = DEFAULT_QUESTIONS_JSON_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
    max_length: int = 512,
) -> int:
    questions = load_questions(questions_path)
    clue_only_texts = [clue_only_text(question) for question in questions]
    category_plus_clue_texts = [category_plus_clue_text(question) for question in questions]

    clue_only_embeddings = encode_question_texts(
        clue_only_texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )
    category_plus_clue_embeddings = encode_question_texts(
        category_plus_clue_texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        clue_only_texts=np.array(clue_only_texts, dtype=object),
        clue_only_embeddings=clue_only_embeddings,
        category_plus_clue_texts=np.array(category_plus_clue_texts, dtype=object),
        category_plus_clue_embeddings=category_plus_clue_embeddings,
    )
    print(f"[processor_question_dpr_embeddings] Wrote embeddings to {output_path}")
    return len(questions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute DPR question embeddings for clue-only and category+clue forms."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_JSON_PATH,
        help=f"Questions JSON path (default: {DEFAULT_QUESTIONS_JSON_PATH})",
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
    total = materialize_question_dpr_embeddings(
        questions_path=args.questions,
        output_path=args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"Stored embeddings for {total} questions")
