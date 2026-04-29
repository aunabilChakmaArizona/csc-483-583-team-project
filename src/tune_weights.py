"""
tune_weights.py — Weight optimizer for --weighted-cole retrieval.

HOW IT WORKS
============
Rather than re-running expensive BM25 / FAISS searches for every weight trial,
this script does two passes:

  Pass 1 (collect):  Run all 100 questions ONCE with every component set to
                     weight=1.  The returned result dicts already carry the
                     per-component scores (body_score, faiss_score, etc.).
                     The retrieval cache also stores these for future runs.

  Pass 2 (tune):     For each weight combination we want to try, re-rank the
                     already-retrieved results by computing a new weighted sum
                     over those stored component scores.  No index is touched.
                     Thousands of trials run in seconds.

USAGE
=====
  # grid search (default) — tries every combination of the listed values:
  python -m src.tune_weights

  # random search — samples N random weight vectors from the given ranges:
  python -m src.tune_weights --strategy random --trials 2000

  # save top results to a JSON file:
  python -m src.tune_weights --save-results tuning_results.json

  # only tune a subset of components (fix the rest at their current value):
  python -m src.tune_weights --tune body faiss natural_avg natural_concat

COMPONENTS TUNED
================
The script currently tunes these active components:
  body          WEIGHTED_BODY_WEIGHT          (BM25 full body)
  faiss         WEIGHTED_FAISS_WEIGHT         (DPR dense, default query)
  natural_avg   WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT
  natural_concat WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT
  year_match    WEIGHTED_YEAR_MATCH_WEIGHT
  first_sent    WEIGHTED_FIRST_SENTENCE_WEIGHT
  first_two     WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT
  first_para    WEIGHTED_FIRST_PARAGRAPH_WEIGHT
  title         WEIGHTED_TITLE_WEIGHT

You can add more by extending COMPONENT_NAMES and the grid in build_grid().
"""

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Imports from the project — tolerates both  python -m src.tune_weights
# and  python tune_weights.py  invocations.
# ---------------------------------------------------------------------------
try:
    from src.processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from src.processor_natural_question_dpr_embeddings import (
        DEFAULT_OUTPUT_PATH as DEFAULT_NQ_EMBEDDINGS_PATH,
    )
    from src.processor_question_dpr_embeddings import (
        DEFAULT_OUTPUT_PATH as DEFAULT_Q_EMBEDDINGS_PATH,
        category_plus_clue_text,
        clue_only_text,
    )
    from src.run_test_100 import (
        JeopardyQuestion,
        load_questions,
        normalize_label,
        valid_answers,
    )
    from src.search import (
        WEIGHTED_BODY_WEIGHT,
        WEIGHTED_FAISS_WEIGHT,
        WEIGHTED_FIRST_PARAGRAPH_WEIGHT,
        WEIGHTED_FIRST_SENTENCE_WEIGHT,
        WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT,
        WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT,
        WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT,
        WEIGHTED_TITLE_WEIGHT,
        WEIGHTED_YEAR_MATCH_WEIGHT,
        multi_search_whoosh_weighted_cole,
        open_whoosh_cole_first_paragraph_index,
        open_whoosh_cole_first_sentence_index,
        open_whoosh_cole_first_two_sentences_index,
        open_whoosh_cole_index,
        open_whoosh_cole_redirect_index,
        open_whoosh_title_body_category_index,
        load_dpr_faiss_variant,
        load_redirect_lookup,
        DEFAULT_DPR_FAISS_INDEX_DIR,
        DPR_FAISS_VARIANT_NAMES,
    )
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from processor_natural_question_dpr_embeddings import (
        DEFAULT_OUTPUT_PATH as DEFAULT_NQ_EMBEDDINGS_PATH,
    )
    from processor_question_dpr_embeddings import (
        DEFAULT_OUTPUT_PATH as DEFAULT_Q_EMBEDDINGS_PATH,
        category_plus_clue_text,
        clue_only_text,
    )
    from run_test_100 import (
        JeopardyQuestion,
        load_questions,
        normalize_label,
        valid_answers,
    )
    from search import (
        WEIGHTED_BODY_WEIGHT,
        WEIGHTED_FAISS_WEIGHT,
        WEIGHTED_FIRST_PARAGRAPH_WEIGHT,
        WEIGHTED_FIRST_SENTENCE_WEIGHT,
        WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT,
        WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT,
        WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT,
        WEIGHTED_TITLE_WEIGHT,
        WEIGHTED_YEAR_MATCH_WEIGHT,
        multi_search_whoosh_weighted_cole,
        open_whoosh_cole_first_paragraph_index,
        open_whoosh_cole_first_sentence_index,
        open_whoosh_cole_first_two_sentences_index,
        open_whoosh_cole_index,
        open_whoosh_cole_redirect_index,
        open_whoosh_title_body_category_index,
        load_dpr_faiss_variant,
        load_redirect_lookup,
        DEFAULT_DPR_FAISS_INDEX_DIR,
        DPR_FAISS_VARIANT_NAMES,
    )


# ---------------------------------------------------------------------------
# Component names that map exactly to the score keys in result dicts.
# ---------------------------------------------------------------------------
COMPONENT_NAMES = [
    "title_score",
    "body_score",
    "first_sentence_score",
    "first_two_sentences_score",
    "first_paragraph_score",
    "year_match_score",
    "faiss_score",
    "natural_questions_avg_faiss_score",
    "natural_questions_concat_faiss_score",
]

# Human-readable short names → score key mapping (for CLI --tune flag)
SHORT_TO_KEY = {
    "title":          "title_score",
    "body":           "body_score",
    "first_sent":     "first_sentence_score",
    "first_two":      "first_two_sentences_score",
    "first_para":     "first_paragraph_score",
    "year_match":     "year_match_score",
    "faiss":          "faiss_score",
    "natural_avg":    "natural_questions_avg_faiss_score",
    "natural_concat": "natural_questions_concat_faiss_score",
}

# Default weight values read from search.py — used for components NOT being tuned.
CURRENT_DEFAULTS = {
    "title_score":                          WEIGHTED_TITLE_WEIGHT,
    "body_score":                           WEIGHTED_BODY_WEIGHT,
    "first_sentence_score":                 WEIGHTED_FIRST_SENTENCE_WEIGHT,
    "first_two_sentences_score":            WEIGHTED_FIRST_TWO_SENTENCES_WEIGHT,
    "first_paragraph_score":                WEIGHTED_FIRST_PARAGRAPH_WEIGHT,
    "year_match_score":                     WEIGHTED_YEAR_MATCH_WEIGHT,
    "faiss_score":                          WEIGHTED_FAISS_WEIGHT,
    "natural_questions_avg_faiss_score":    WEIGHTED_NATURAL_QUESTIONS_AVG_FAISS_WEIGHT,
    "natural_questions_concat_faiss_score": WEIGHTED_NATURAL_QUESTIONS_CONCAT_FAISS_WEIGHT,
}

# Evaluation k-values to report
TOP_K_VALUES = [1, 5, 10]


# ---------------------------------------------------------------------------
# Grid definitions — edit these to broaden or narrow the search.
# ---------------------------------------------------------------------------
def build_grid(tune_keys: list[str]) -> dict[str, list[float]]:
    """
    Return a dict mapping each *tuned* score key to the list of values to try.

    We use a coarser grid for components that have historically had
    little impact (title, first_sentence, year_match) to keep total
    trial count manageable.
    """
    fine   = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    coarse = [0.0, 1.0, 5.0]
    binary = [0.0, 1.0]

    key_to_values = {
        "title_score":                          binary,
        "body_score":                           fine,
        "first_sentence_score":                 coarse,
        "first_two_sentences_score":            coarse,
        "first_paragraph_score":                coarse,
        "year_match_score":                     binary,
        "faiss_score":                          coarse,
        "natural_questions_avg_faiss_score":    coarse,
        "natural_questions_concat_faiss_score": coarse,
    }
    return {k: key_to_values[k] for k in tune_keys}


# ---------------------------------------------------------------------------
# Pass 1: collect per-component scores from a single full run.
# ---------------------------------------------------------------------------
def collect_component_scores(
    questions: list[JeopardyQuestion],
    include_category: bool = False,
) -> list[dict]:
    """
    Run all questions with EQUAL weights (every component active at 1.0) so
    that every component's score is populated in the returned result dicts.

    Returns a list of per-question dicts:
      {
        "answers": set[str],        # normalized valid answers
        "results": list[dict],      # result dicts with *_score fields
      }
    """
    print("[tune_weights] Pass 1: collecting per-component scores (all weights = 1.0)…")
    start = time.time()

    queries = []
    categories = []
    for q in questions:
        if include_category:
            queries.append(f"{q.category} {q.clue}")
        else:
            queries.append(q.clue)
        categories.append(q.category)

    # Load precomputed DPR embeddings if available.
    dense_embeddings = _load_dense_embeddings(questions, include_category)
    nq_embeddings, nq_concat_embeddings = _load_nq_embeddings(questions)

    # Call with all weights = 1.0 so every score field is computed.
    searches = multi_search_whoosh_weighted_cole(
        queries,
        query_categories=categories,
        limit=1000,
        title_weight=1.0,
        body_weight=1.0,
        first_sentence_weight=1.0,
        first_two_sentences_weight=1.0,
        first_paragraph_weight=1.0,
        year_match_weight=1.0,
        faiss_weight=1.0,
        natural_questions_avg_faiss_weight=1.0,
        natural_questions_concat_faiss_weight=1.0,
        category_first_sentence_weight=0.0,      
        category_first_two_sentences_weight=0.0, 
        dense_query_embeddings=dense_embeddings,
        natural_questions_embeddings=nq_embeddings,
        natural_questions_concat_embeddings=nq_concat_embeddings,
    )

    elapsed = time.time() - start
    print(f"[tune_weights] Pass 1 done in {elapsed:.1f}s")

    collected = []
    for question, search in zip(questions, searches):
        collected.append(
            {
                "answers": valid_answers(question.answer),
                "results": search["results"],
            }
        )
    return collected


# ---------------------------------------------------------------------------
# Pass 2: re-rank using a given weight dict — pure arithmetic, no index I/O.
# ---------------------------------------------------------------------------
def rerank(collected: list[dict], weights: dict[str, float], limit: int = 1000) -> list[list[dict]]:
    """
    Re-rank each question's result list using the supplied weight dict.
    Returns a list of ranked result lists (one per question).
    """
    all_reranked = []
    for item in collected:
        results = item["results"]
        # Compute new score for each candidate.
        rescored = []
        for r in results:
            new_score = sum(
                weights.get(key, 0.0) * r.get(key, 0.0)
                for key in COMPONENT_NAMES
            )
            rescored.append({**r, "score": new_score})
        # Sort descending by new score.
        rescored.sort(key=lambda x: -x["score"])
        all_reranked.append(rescored[:limit])
    return all_reranked


def evaluate(collected: list[dict], reranked: list[list[dict]]) -> dict[int, float]:
    """Compute top-k accuracy for each k in TOP_K_VALUES."""
    counts = {k: 0 for k in TOP_K_VALUES}
    total = len(collected)
    for item, ranked in zip(collected, reranked):
        answers = item["answers"]
        for k in TOP_K_VALUES:
            top_titles = {normalize_label(r["title"]) for r in ranked[:k]}
            if top_titles & answers:
                counts[k] += 1
    return {k: counts[k] / total for k in TOP_K_VALUES}


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def grid_search(
    collected: list[dict],
    tune_keys: list[str],
    fixed_weights: dict[str, float],
    primary_k: int = 5,
) -> list[dict]:
    """
    Exhaustive grid search over all combinations of tuned weight values.
    Returns results sorted by primary_k accuracy (descending).
    """
    grid = build_grid(tune_keys)
    value_lists = [grid[k] for k in tune_keys]
    combinations = list(itertools.product(*value_lists))
    total = len(combinations)
    print(f"[tune_weights] Grid search: {total:,} combinations over keys {tune_keys}")

    results = []
    best_acc = -1.0
    start = time.time()

    for trial_num, combo in enumerate(combinations, start=1):
        weights = {**fixed_weights}
        for key, val in zip(tune_keys, combo):
            weights[key] = val

        reranked = rerank(collected, weights)
        metrics  = evaluate(collected, reranked)
        acc      = metrics[primary_k]

        results.append({"weights": weights.copy(), "metrics": metrics})

        if acc > best_acc:
            best_acc = acc
            print(
                f"  [trial {trial_num}/{total}] New best top-{primary_k}: "
                f"{acc:.4f}  weights={_fmt_weights(weights, tune_keys)}"
            )

        if trial_num % max(1, total // 20) == 0:
            elapsed = time.time() - start
            eta = elapsed / trial_num * (total - trial_num)
            print(
                f"  [trial {trial_num}/{total}] elapsed {elapsed:.1f}s  ETA {eta:.0f}s"
            )

    results.sort(key=lambda x: -x["metrics"][primary_k])
    return results


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------
def random_search(
    collected: list[dict],
    tune_keys: list[str],
    fixed_weights: dict[str, float],
    n_trials: int = 2000,
    primary_k: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Sample n_trials random weight vectors from continuous uniform ranges.
    Ranges are inferred from build_grid (min, max of listed values).
    """
    rng = random.Random(seed)
    grid = build_grid(tune_keys)
    ranges = {k: (min(v), max(v)) for k, v in grid.items()}

    print(f"[tune_weights] Random search: {n_trials:,} trials over keys {tune_keys}")
    results = []
    best_acc = -1.0
    start = time.time()

    for trial_num in range(1, n_trials + 1):
        weights = {**fixed_weights}
        for key in tune_keys:
            lo, hi = ranges[key]
            weights[key] = rng.uniform(lo, hi)

        reranked = rerank(collected, weights)
        metrics  = evaluate(collected, reranked)
        acc      = metrics[primary_k]

        results.append({"weights": weights.copy(), "metrics": metrics})

        if acc > best_acc:
            best_acc = acc
            print(
                f"  [trial {trial_num}/{n_trials}] New best top-{primary_k}: "
                f"{acc:.4f}  weights={_fmt_weights(weights, tune_keys)}"
            )

    elapsed = time.time() - start
    print(f"[tune_weights] Random search done in {elapsed:.1f}s")
    results.sort(key=lambda x: -x["metrics"][primary_k])
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt_weights(weights: dict, keys: list[str]) -> str:
    return "  ".join(f"{k}={weights[k]:.2f}" for k in keys)


def _load_dense_embeddings(questions, include_category):
    """Load precomputed DPR clue embeddings if the .npz file exists."""
    if not DEFAULT_Q_EMBEDDINGS_PATH.exists():
        return None
    saved = np.load(DEFAULT_Q_EMBEDDINGS_PATH, allow_pickle=True)
    if include_category:
        return list(saved["category_plus_clue_embeddings"][: len(questions)])
    return list(saved["clue_only_embeddings"][: len(questions)])


def _load_nq_embeddings(questions):
    """Load precomputed natural-question DPR embeddings if available."""
    if not DEFAULT_NQ_EMBEDDINGS_PATH.exists():
        return None, None
    saved = np.load(DEFAULT_NQ_EMBEDDINGS_PATH, allow_pickle=True)
    nq   = list(saved["natural_questions_embeddings"][: len(questions)])
    conc = list(saved["combined_natural_questions_embeddings"][: len(questions)])
    return nq, conc


def print_top_n(results: list[dict], n: int = 10, primary_k: int = 5) -> None:
    print(f"\n{'='*70}")
    print(f"Top {n} weight configurations by top-{primary_k} accuracy:")
    print(f"{'='*70}")
    for rank, entry in enumerate(results[:n], start=1):
        m = entry["metrics"]
        acc_str = "  ".join(f"top-{k}: {m[k]:.4f}" for k in TOP_K_VALUES)
        w_str   = "  ".join(
            f"{k.replace('_score','')}={v:.2f}"
            for k, v in entry["weights"].items()
            if v != 0.0
        )
        print(f"  #{rank:2d}  {acc_str}")
        print(f"       weights: {w_str}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune --weighted-cole component weights without re-running the index."
    )
    parser.add_argument(
        "--strategy",
        choices=["grid", "random"],
        default="grid",
        help="Search strategy. 'grid' tries all combinations; 'random' samples N vectors.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=2000,
        help="Number of random trials (only used when --strategy random).",
    )
    parser.add_argument(
        "--primary-k",
        type=int,
        default=5,
        choices=[1, 5, 10],
        help="Which top-k accuracy to optimize for. Default: 5.",
    )
    parser.add_argument(
        "--tune",
        nargs="+",
        choices=list(SHORT_TO_KEY.keys()),
        default=["body", "faiss", "natural_avg", "natural_concat", "year_match"],
        help=(
            "Which components to tune. Others are held at their current defaults. "
            f"Choices: {list(SHORT_TO_KEY.keys())}"
        ),
    )
    parser.add_argument(
        "--include-category",
        action="store_true",
        help="Prepend the Jeopardy category to each clue when querying.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many top configurations to print at the end.",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save full results as JSON (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve short names to score key names
    tune_keys = [SHORT_TO_KEY[s] for s in args.tune]
    fixed_weights = {
        key: CURRENT_DEFAULTS[key]
        for key in COMPONENT_NAMES
        if key not in tune_keys
    }

    print(f"[tune_weights] Strategy:       {args.strategy}")
    print(f"[tune_weights] Optimizing:     top-{args.primary_k} accuracy")
    print(f"[tune_weights] Tuning:         {tune_keys}")
    print(f"[tune_weights] Fixed weights:  {fixed_weights}")

    # --- Load questions ---
    questions = load_questions(DEFAULT_QUESTIONS_JSON_PATH)[:100]

    # --- Pass 1: collect per-component scores ---
    collected = collect_component_scores(questions, include_category=args.include_category)

    # --- Baseline: evaluate with current configured weights ---
    baseline_reranked = rerank(collected, CURRENT_DEFAULTS)
    baseline_metrics  = evaluate(collected, baseline_reranked)
    print(
        "\n[tune_weights] Baseline (current search.py weights): "
        + "  ".join(f"top-{k}: {baseline_metrics[k]:.4f}" for k in TOP_K_VALUES)
    )

    # --- Pass 2: weight search ---
    if args.strategy == "grid":
        results = grid_search(
            collected,
            tune_keys=tune_keys,
            fixed_weights=fixed_weights,
            primary_k=args.primary_k,
        )
    else:
        results = random_search(
            collected,
            tune_keys=tune_keys,
            fixed_weights=fixed_weights,
            n_trials=args.trials,
            primary_k=args.primary_k,
        )

    # --- Print summary ---
    print_top_n(results, n=args.top_n, primary_k=args.primary_k)

    # --- Optionally save ---
    if args.save_results:
        save_path = Path(args.save_results)
        # Convert sets in answers to lists for JSON serialization
        output = {
            "baseline": {"metrics": baseline_metrics},
            "top_results": results[: args.top_n],
            "strategy": args.strategy,
            "tune_keys": tune_keys,
            "primary_k": args.primary_k,
        }
        save_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\n[tune_weights] Results saved to {save_path}")


if __name__ == "__main__":
    main()