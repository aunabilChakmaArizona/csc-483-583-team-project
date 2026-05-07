"""Inspect how Whoosh analyzes and parses a query."""

import argparse

from whoosh.qparser import QueryParser

try:
    from src.processor4_whoosh_index import DEFAULT_INDEX_DIR, open_index
except ModuleNotFoundError:
    from processor4_whoosh_index import DEFAULT_INDEX_DIR, open_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Whoosh query processing.")
    parser.add_argument("query", nargs="?", default="This is, THE apple-banana test!")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = open_index(DEFAULT_INDEX_DIR)
    body_field = index.schema["body"]
    parser = QueryParser("body", schema=index.schema)

    tokens = [token.text for token in body_field.analyzer(args.query)]
    parsed_query = parser.parse(args.query)

    print(f"Raw query: {args.query}")
    print(f"Analyzer tokens: {tokens}")
    print(f"Parsed query: {parsed_query}")


if __name__ == "__main__":
    main()
