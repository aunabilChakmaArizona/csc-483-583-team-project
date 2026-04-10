"""Print query processing differences for token and Whoosh modes."""

# python -m src.temp_compare_query_processing

import argparse
import json
from pathlib import Path

try:
    from src.processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from src.processor4_whoosh_index import DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_INDEX_DIR
    from src.processor4_whoosh_index import open_index as open_whoosh_index
    from src.search import normalize_query_terms
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_QUESTIONS_JSON_PATH
    from processor4_whoosh_index import DEFAULT_INDEX_DIR as DEFAULT_WHOOSH_INDEX_DIR
    from processor4_whoosh_index import open_index as open_whoosh_index
    from search import normalize_query_terms

def load_stop_words() -> set[str]:
    from spacy.lang.en.stop_words import STOP_WORDS
    return set(STOP_WORDS)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare token and Whoosh query processing.")
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def load_clues(path: Path = DEFAULT_QUESTIONS_JSON_PATH, limit: int = 100) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [row["clue"] for row in rows[:limit]]


def whoosh_query_terms(query: str) -> list[str]:
    index = open_whoosh_index(DEFAULT_WHOOSH_INDEX_DIR)
    body_field = index.schema["body"]
    return [token.text for token in body_field.analyzer(query)]


def main() -> None:
    args = parse_args()
    clues = load_clues(limit=args.limit)

    for question_number, clue in enumerate(clues, start=1):
        token_terms = normalize_query_terms(clue)
        whoosh_terms = whoosh_query_terms(clue)
        print(
            f"{question_number}\n"
            f"raw={clue}\n"
            f"token={' '.join(token_terms)}\n"
            f"whoosh={' '.join(whoosh_terms)}\n\n\n"
        )


if __name__ == "__main__":
    main()
    print("=================")
    sw = sorted(list(load_stop_words()))
    for w in sw:
        print(w)
