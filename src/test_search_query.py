"""Run a simple search query and print matching article titles."""

import time

try:
    from src.search import search
except ModuleNotFoundError:
    from search import search


def main() -> None:
    query = "apple banana"
    start_time = time.time()
    results = search(query, limit=10)
    elapsed = time.time() - start_time

    print(f"Query: {query}")
    print(f"Time: {elapsed:.4f}s")
    for result in results:
        print(result["title"])


if __name__ == "__main__":
    main()
