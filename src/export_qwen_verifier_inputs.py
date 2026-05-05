"""Export Qwen verifier inputs for offline/Colab scoring.

This script does not run Qwen. It exports one JSONL row per
question/document candidate using the cached retrieval + cross-encoder stage.

Typical local usage:

    python src/export_qwen_verifier_inputs.py \
        --output data/processed/qwen_verifier_inputs.jsonl \
        --top-docs 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from src.processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from src.run_test_100 import (
        CROSS_ENCODER_TOP_K_VALUES,
        DEFAULT_CROSS_ENCODER_MODEL_NAME,
        DEFAULT_CROSS_ENCODER_NATURAL_QUESTIONS_PATH,
        TOP_K_VALUES,
        build_stage2_searches,
        cross_encoder_document_text,
        get_cross_encoder_natural_questions,
        hydrate_cross_encoder_results,
        load_questions,
        search_questions,
    )
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from run_test_100 import (
        CROSS_ENCODER_TOP_K_VALUES,
        DEFAULT_CROSS_ENCODER_MODEL_NAME,
        DEFAULT_CROSS_ENCODER_NATURAL_QUESTIONS_PATH,
        TOP_K_VALUES,
        build_stage2_searches,
        cross_encoder_document_text,
        get_cross_encoder_natural_questions,
        hydrate_cross_encoder_results,
        load_questions,
        search_questions,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/processed/qwen_verifier_inputs.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export clue/category/natural-question/top-document rows for Qwen scoring."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_JSON_PATH)
    parser.add_argument("--question-limit", type=int, default=100)
    parser.add_argument("--top-docs", type=int, default=10)
    parser.add_argument("--evidence-token-limit", type=int, default=360)
    parser.add_argument("--include-category", action="store_true", default=True)
    parser.add_argument("--no-include-category", action="store_false", dest="include_category")
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL_NAME)
    parser.add_argument(
        "--cross-encoder-mode",
        choices=["full_article", "chunks_256", "chunks_100"],
        default="chunks_100",
    )
    parser.add_argument(
        "--cross-encoder-query-mode",
        choices=["category_clue", "natural_questions_avg"],
        default="natural_questions_avg",
    )
    parser.add_argument(
        "--cross-encoder-natural-questions",
        type=Path,
        default=DEFAULT_CROSS_ENCODER_NATURAL_QUESTIONS_PATH,
    )
    return parser.parse_args()


def first_words(text: str, limit: int) -> str:
    words = text.strip().split()
    if limit <= 0 or len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit])


def candidate_key(question_index: int, result: dict) -> str:
    raw_key = json.dumps(
        {
            "question_index": question_index,
            "source_file": result.get("source_file", ""),
            "article_index": result.get("article_index"),
            "title": result.get("title", ""),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions)[: args.question_limit]
    natural_question_groups = get_cross_encoder_natural_questions(
        questions,
        args.cross_encoder_natural_questions,
    )

    searches = search_questions(
        questions,
        limit=max(TOP_K_VALUES),
        mode="token",
        query_mode="full",
        include_category=args.include_category,
        weighted=False,
        weight_equal=False,
        weighted_cole=True,
        skip_redirects=False,
        paraphrase=False,
        progress_every=10,
    )
    _, fused_searches = build_stage2_searches(
        questions,
        searches,
        cross_encoder_model=args.cross_encoder_model,
        cross_encoder_mode=args.cross_encoder_mode,
        cross_encoder_query_mode=args.cross_encoder_query_mode,
        cross_encoder_natural_questions_path=args.cross_encoder_natural_questions,
        natural_question_groups=natural_question_groups,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    top_docs = min(args.top_docs, max(CROSS_ENCODER_TOP_K_VALUES))

    with args.output.open("w", encoding="utf-8") as output_file:
        for question_index, (question, search) in enumerate(
            zip(questions, fused_searches),
            start=1,
        ):
            candidates = hydrate_cross_encoder_results(search["results"][:top_docs])
            natural_questions = natural_question_groups[question_index - 1][:5]
            for candidate_rank, result in enumerate(candidates, start=1):
                evidence = first_words(
                    cross_encoder_document_text(result),
                    args.evidence_token_limit,
                )
                row = {
                    "candidate_key": candidate_key(question_index - 1, result),
                    "question_index": question_index - 1,
                    "category": question.category,
                    "clue": question.clue,
                    "answer": question.answer,
                    "natural_questions": natural_questions,
                    "candidate_rank": candidate_rank,
                    "title": result.get("title", ""),
                    "source_file": result.get("source_file", ""),
                    "article_index": result.get("article_index"),
                    "initial_rank": result.get("initial_rank"),
                    "cross_encoder_rank": result.get("cross_encoder_rank"),
                    "initial_score": result.get("initial_score"),
                    "cross_encoder_score": result.get("cross_encoder_score"),
                    "fused_stage2_score": result.get("fused_stage2_score"),
                    "evidence": evidence,
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                exported += 1

            print(
                f"[export-qwen-verifier-inputs] Question {question_index}/{len(questions)} | "
                f"exported rows: {exported}",
                flush=True,
            )

    print(f"Wrote {exported} rows to {args.output}")


if __name__ == "__main__":
    main()
