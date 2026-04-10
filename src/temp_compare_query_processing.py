"""Print query processing differences for token and Whoosh modes."""

# python -m src.temp_compare_query_processing

import argparse
import json
from pathlib import Path

try:
    from src.processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from src.processor4_whoosh_index import DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_INDEX_DIR
    from src.processor4_whoosh_index import open_index as open_whoosh_index
    from src.run_test_100 import entity_query_text
    from src.search import normalize_query_terms
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from processor4_whoosh_index import DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_INDEX_DIR
    from processor4_whoosh_index import open_index as open_whoosh_index
    from run_test_100 import entity_query_text
    from search import normalize_query_terms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare token and Whoosh query processing.")
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def load_clues(path: Path = DEFAULT_QUESTIONS_JSON_PATH, limit: int = 100) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [row["clue"] for row in rows[:limit]]


def whoosh_query_terms(query: str, body_field) -> list[str]:
    return [token.text for token in body_field.analyzer(query)]


def main() -> None:
    args = parse_args()
    clues = load_clues(limit=args.limit)
    whoosh_index = open_whoosh_index(DEFAULT_WHOOSH_INDEX_DIR)
    body_field = whoosh_index.schema["body"]

    for question_number, clue in enumerate(clues, start=1):
        entity_query = entity_query_text(clue)
        token_terms = normalize_query_terms(clue)
        whoosh_terms = whoosh_query_terms(clue, body_field)
        token_entity_terms = normalize_query_terms(entity_query)
        whoosh_entity_terms = whoosh_query_terms(entity_query, body_field)
        print(
            f"{question_number}\n"
            f"raw={clue}\n"
            f"entity_noun={entity_query}\n"
            f"token={' '.join(token_terms)}\n"
            f"whoosh={' '.join(whoosh_terms)}\n"
            f"token_entity={' '.join(token_entity_terms)}\n"
            f"whoosh_entity={' '.join(whoosh_entity_terms)}\n\n\n"
        )


if __name__ == "__main__":
    main()
