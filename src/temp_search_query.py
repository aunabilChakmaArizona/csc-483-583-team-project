"""Run a simple search query and print matching article titles."""

# python -m src.test_search_query "apple banana" --mode token --limit 10

import argparse
import time

try:
    from src.search import search, search_whoosh_default, search_whoosh_title_body
except ModuleNotFoundError:
    from search import search, search_whoosh_default, search_whoosh_title_body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple search and print article titles.")
    parser.add_argument("query", nargs="?", default="apple banana")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["token", "whoosh", "whoosh_title_body"],
        default="token",
        help=(
            "token uses processor3 tokens; whoosh searches cleaned body; "
            "whoosh_title_body searches cleaned title and body."
        ),
    )
    return parser.parse_args()


def run_search(query: str, limit: int, mode: str) -> list[dict]:
    if mode == "whoosh_title_body":
        return search_whoosh_title_body(query, limit=limit)

    if mode == "whoosh":
        return search_whoosh_default(query, limit=limit)

    return search(query, limit=limit)


def main() -> None:
    args = parse_args()

    start_time = time.time()
    results = run_search(args.query, limit=args.limit, mode=args.mode)
    elapsed = time.time() - start_time

    print(f"Mode: {args.mode}")
    print(f"Query: {args.query}")
    print(f"Time: {elapsed:.4f}s")
    for result in results:
        print(result["title"])


if __name__ == "__main__":
    main()
