"""Export cross-encoder pair inputs for offline/Colab scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from src.processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from src.run_test_100 import (
        CROSS_ENCODER_CANDIDATE_LIMIT,
        DEFAULT_CROSS_ENCODER_MODEL_NAME,
        DEFAULT_CROSS_ENCODER_NATURAL_QUESTIONS_PATH,
        TOP_K_VALUES,
        cross_encoder_chunk_token_limit,
        cross_encoder_document_text,
        cross_encoder_pair_cache_key,
        cross_encoder_query_texts,
        get_cross_encoder_natural_questions,
        hydrate_cross_encoder_results,
        load_questions,
        search_questions,
    )
    from src.retrieval_cache import load_cached_component_results
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from run_test_100 import (
        CROSS_ENCODER_CANDIDATE_LIMIT,
        DEFAULT_CROSS_ENCODER_MODEL_NAME,
        DEFAULT_CROSS_ENCODER_NATURAL_QUESTIONS_PATH,
        TOP_K_VALUES,
        cross_encoder_chunk_token_limit,
        cross_encoder_document_text,
        cross_encoder_pair_cache_key,
        cross_encoder_query_texts,
        get_cross_encoder_natural_questions,
        hydrate_cross_encoder_results,
        load_questions,
        search_questions,
    )
    from retrieval_cache import load_cached_component_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/processed/cross_encoder_pairs.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cross-encoder pair inputs and cache keys for Colab scoring."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_JSON_PATH)
    parser.add_argument("--question-limit", type=int, default=100)
    parser.add_argument("--model-name", default=DEFAULT_CROSS_ENCODER_MODEL_NAME)
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
    parser.add_argument("--candidate-limit", type=int, default=CROSS_ENCODER_CANDIDATE_LIMIT)
    parser.add_argument("--include-category", action="store_true", default=True)
    parser.add_argument("--no-include-category", action="store_false", dest="include_category")
    parser.add_argument("--skip-existing-cache", action="store_true")
    return parser.parse_args()


def document_chunks(document_text: str, mode: str) -> list[str]:
    if mode == "full_article":
        return [document_text]

    words = document_text.strip().split()
    if not words:
        return [""]

    chunk_size = cross_encoder_chunk_token_limit(mode)
    return [
        " ".join(words[start : start + chunk_size]).strip()
        for start in range(0, len(words), chunk_size)
    ]


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions)[: args.question_limit]
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
    natural_question_groups = None
    if args.cross_encoder_query_mode == "natural_questions_avg":
        natural_question_groups = get_cross_encoder_natural_questions(
            questions,
            args.cross_encoder_natural_questions,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    skipped = 0
    seen_cache_keys = set()

    with args.output.open("w", encoding="utf-8") as output_file:
        for question_index, (question, search_result) in enumerate(
            zip(questions, searches),
            start=1,
        ):
            natural_questions = (
                natural_question_groups[question_index - 1]
                if natural_question_groups is not None
                else None
            )
            query_texts = cross_encoder_query_texts(
                question,
                query_mode=args.cross_encoder_query_mode,
                natural_questions=natural_questions,
            )
            candidates = hydrate_cross_encoder_results(
                search_result["results"][: args.candidate_limit]
            )

            for candidate_rank, result in enumerate(candidates, start=1):
                chunks = document_chunks(cross_encoder_document_text(result), args.cross_encoder_mode)
                for query_number, text_a in enumerate(query_texts, start=1):
                    for chunk_number, text_b in enumerate(chunks, start=1):
                        cache_key = cross_encoder_pair_cache_key(args.model_name, text_a, text_b)
                        if cache_key in seen_cache_keys:
                            skipped += 1
                            continue
                        seen_cache_keys.add(cache_key)
                        if args.skip_existing_cache and load_cached_component_results(cache_key) is not None:
                            skipped += 1
                            continue

                        output_file.write(
                            json.dumps(
                                {
                                    "cache_key": cache_key,
                                    "model_name": args.model_name,
                                    "text_a": text_a,
                                    "text_b": text_b,
                                    "question_index": question_index - 1,
                                    "candidate_rank": candidate_rank,
                                    "query_number": query_number,
                                    "chunk_number": chunk_number,
                                    "title": result.get("title", ""),
                                    "source_file": result.get("source_file", ""),
                                    "article_index": result.get("article_index"),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        exported += 1

            print(
                f"[export-cross-encoder-pairs] Question {question_index}/{len(questions)} | "
                f"exported: {exported} | skipped: {skipped}"
            )

    print(f"Wrote {exported} pairs to {args.output}")
    print(f"Skipped {skipped} duplicate/existing pairs")


if __name__ == "__main__":
    main()
