"""Evaluate search quality for one Jeopardy clue selected from the questions set."""

import argparse
from pathlib import Path

try:
    from src.processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from src.run_test_100 import (
        DEFAULT_PARAPHRASE_COUNT,
        DEFAULT_PARAPHRASE_FETCH_LIMIT,
        JeopardyQuestion,
        TOP_K_VALUES,
        evaluate_questions,
        first_correct_k,
        is_correct_at_k,
        load_questions,
        normalize_label,
        print_metrics,
        search_questions,
        valid_answers,
    )
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from run_test_100 import (
        DEFAULT_PARAPHRASE_COUNT,
        DEFAULT_PARAPHRASE_FETCH_LIMIT,
        JeopardyQuestion,
        TOP_K_VALUES,
        evaluate_questions,
        first_correct_k,
        is_correct_at_k,
        load_questions,
        normalize_label,
        print_metrics,
        search_questions,
        valid_answers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one Jeopardy clue from the questions set.")
    parser.add_argument(
        "--clue",
        required=True,
        help="Exact clue text to run, matched case-insensitively with normalized whitespace.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help=(
            "Additional clue variant to search alongside the original clue. "
            "Can be passed multiple times; document scores are averaged across all variants."
        ),
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path(DEFAULT_QUESTIONS_JSON_PATH),
        help=f"Questions JSON path. Default: {DEFAULT_QUESTIONS_JSON_PATH}",
    )
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
        help="Use weighted retrieval with title/body plus Cole lead indexes and redirect evidence.",
    )
    weight_group.add_argument(
        "--weight-equal",
        action="store_true",
        help="Force equal weights across the weighted retrieval components.",
    )
    weight_group.add_argument(
        "--weighted-cole",
        action="store_true",
        help="Use the weighted formula on Cole title/body scoring with lead and redirect evidence.",
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
            "Expand the raw query with model-generated paraphrases, retrieve candidates "
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


def select_question_by_clue(clue: str, questions_path: Path):
    questions = load_questions(questions_path)
    normalized_target = normalize_label(clue)
    matches = [question for question in questions if normalize_label(question.clue) == normalized_target]

    if not matches:
        raise RuntimeError(f'No question matched clue: "{clue}"')

    if len(matches) > 1:
        print(
            f"[run_test_specific] Found {len(matches)} matching questions for the clue; "
            "running the first exact match."
        )

    return matches[0]


def average_ranked_results(searches: list[dict], limit: int) -> list[dict]:
    total_queries = len(searches)
    averaged_results: dict[tuple, dict] = {}

    for search in searches:
        for result in search["results"]:
            key = (
                result["source_file"],
                result["article_index"],
                normalize_label(result["title"]),
            )
            entry = averaged_results.setdefault(
                key,
                {
                    **result,
                    "score_sum": 0.0,
                    "matched_variants": 0,
                    "best_score": result["score"],
                },
            )
            entry["score_sum"] += result["score"]
            entry["matched_variants"] += 1
            entry["best_score"] = max(entry["best_score"], result["score"])

    for entry in averaged_results.values():
        entry["score"] = entry["score_sum"] / total_queries

    ranked_results = sorted(
        averaged_results.values(),
        key=lambda result: (
            -result["score"],
            -result["matched_variants"],
            -result["best_score"],
            normalize_label(result["title"]),
        ),
    )
    return ranked_results[:limit]


def main() -> None:
    args = parse_args()
    question = select_question_by_clue(args.clue, args.questions_path)

    print("[run_test_specific] Selected question:")
    print(f"Category: {question.category}")
    print(f"Clue: {question.clue}")
    print(f"Answer: {question.answer}")

    import time

    start_time = time.time()
    if args.variant:
        unique_variants = []
        seen_variants = {normalize_label(question.clue)}

        for variant in args.variant:
            normalized_variant = normalize_label(variant)
            if not normalized_variant or normalized_variant in seen_variants:
                continue
            seen_variants.add(normalized_variant)
            unique_variants.append(variant)

        variant_questions = [question] + [
            JeopardyQuestion(
                category=question.category,
                clue=variant,
                answer=question.answer,
            )
            for variant in unique_variants
        ]

        print("[run_test_specific] Variant mode enabled:")
        print(f"Original clue + {len(unique_variants)} additional variants")
        for variant_index, variant_question in enumerate(variant_questions, start=1):
            label = "original" if variant_index == 1 else f"variant {variant_index - 1}"
            print(f"  {label}: {variant_question.clue}")

        searches = search_questions(
            variant_questions,
            limit=max(TOP_K_VALUES),
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
        )
        averaged_results = average_ranked_results(searches, limit=max(TOP_K_VALUES))
        answers = valid_answers(question.answer)
        metrics = {
            k: 1.0 if is_correct_at_k(averaged_results, answers, k) else 0.0
            for k in TOP_K_VALUES
        }
        matched_k = first_correct_k(averaged_results, answers, TOP_K_VALUES)
        if matched_k is not None:
            print(f"[run_test_specific] Averaged ranking first matched at Top-{matched_k}")
        else:
            print("[run_test_specific] Averaged ranking did not match within Top-10000")
    else:
        metrics = evaluate_questions(
            [question],
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
            top_k_values=TOP_K_VALUES,
        )
    elapsed = time.time() - start_time

    print_metrics(
        metrics,
        total=1,
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
    )


if __name__ == "__main__":
    main()
