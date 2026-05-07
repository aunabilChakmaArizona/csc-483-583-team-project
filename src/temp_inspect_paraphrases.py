"""Inspect paraphrases and normalized query variants for a few Jeopardy questions."""

# PYTHONPATH=. python -m src.temp_inspect_paraphrases --limit 5 --paraphrase-count 4

import argparse

try:
    from src.run_test_100 import (
        DEFAULT_PARAPHRASE_COUNT,
        JeopardyQuestion,
        generate_paraphrases,
        load_questions,
        normalize_label,
        question_text,
        transform_query_text,
    )
    from src.search import normalize_query_terms
except ModuleNotFoundError:
    from run_test_100 import (
        DEFAULT_PARAPHRASE_COUNT,
        JeopardyQuestion,
        generate_paraphrases,
        load_questions,
        normalize_label,
        question_text,
        transform_query_text,
    )
    from search import normalize_query_terms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show original and paraphrased query variants plus normalized tokens."
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of questions to inspect.")
    parser.add_argument(
        "--paraphrase-count",
        type=int,
        default=DEFAULT_PARAPHRASE_COUNT,
        help="Number of paraphrases to generate per question.",
    )
    parser.add_argument(
        "--query-mode",
        choices=["full", "entity"],
        default="full",
        help="Apply the same query transformation used in run_test_100.",
    )
    parser.add_argument(
        "--include-category",
        action="store_true",
        help="Prefix the category to the clue before paraphrasing.",
    )
    return parser.parse_args()


def build_raw_variants(
    question: JeopardyQuestion,
    include_category: bool,
    paraphrase_count: int,
) -> list[str]:
    base_query = question_text(question, include_category)
    variants = [base_query]
    variants.extend(generate_paraphrases(base_query, count=paraphrase_count))

    deduplicated_variants = []
    seen = set()

    for variant in variants:
        normalized_variant = normalize_label(variant)
        if not variant.strip() or normalized_variant in seen:
            continue

        seen.add(normalized_variant)
        deduplicated_variants.append(variant)

    return deduplicated_variants


def print_question_variants(
    question_number: int,
    total_questions: int,
    question: JeopardyQuestion,
    query_mode: str,
    include_category: bool,
    paraphrase_count: int,
) -> None:
    raw_variants = build_raw_variants(
        question,
        include_category=include_category,
        paraphrase_count=paraphrase_count,
    )

    print(f"Question {question_number}/{total_questions}")
    print(f"Category: {question.category}")
    print(f"Clue: {question.clue}")
    print(f"Answer: {question.answer}")

    for variant_number, raw_variant in enumerate(raw_variants):
        transformed_variant = transform_query_text(raw_variant, query_mode).strip()
        token_terms = normalize_query_terms(transformed_variant)
        variant_label = "original" if variant_number == 0 else f"paraphrase {variant_number}"

        print(f"  Variant {variant_number} ({variant_label})")
        print(f"    raw: {raw_variant}")
        print(f"    transformed: {transformed_variant}")
        print(f"    normalized: {' '.join(token_terms)}")

    print()


def main() -> None:
    args = parse_args()
    questions = load_questions()[: args.limit]

    print(f"Inspecting {len(questions)} questions")
    print(f"Query mode: {args.query_mode}")
    print(f"Include category: {args.include_category}")
    print(f"Paraphrase count: {args.paraphrase_count}")
    print()

    for question_number, question in enumerate(questions, start=1):
        print_question_variants(
            question_number=question_number,
            total_questions=len(questions),
            question=question,
            query_mode=args.query_mode,
            include_category=args.include_category,
            paraphrase_count=args.paraphrase_count,
        )


if __name__ == "__main__":
    main()
