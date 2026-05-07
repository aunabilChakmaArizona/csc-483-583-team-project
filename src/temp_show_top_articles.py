"""Print a few wiki articles from the SQLite database."""

from dataclasses import dataclass
from pathlib import Path
import argparse
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data/processed/wiki_articles.sqlite3"


@dataclass(frozen=True)
class WikiArticle:
    title: str
    body: str
    source_file: str
    article_index: int
    is_redirect: int


def fetch_top_articles(db_path: Path, limit: int = 5) -> list[WikiArticle]:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            SELECT title, body, source_file, article_index, is_redirect
            FROM articles
            ORDER BY source_file, article_index
            LIMIT ?
            """,
            (limit,),
        )
        return [WikiArticle(*row) for row in cursor.fetchall()]


def print_articles(articles: list[WikiArticle]) -> None:
    for i, article in enumerate(articles, start=1):
        print(f"Article {i}")
        print(f"Title: {article.title}")
        print(f"Source file: {article.source_file}")
        print(f"Article index: {article.article_index}")
        print(f"Is redirect: {bool(article.is_redirect)}")
        print("Body:")
        print(article.body)
        print("-" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the first few wiki articles from a SQLite DB.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of articles to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    articles = fetch_top_articles(args.db_path, args.limit)
    print_articles(articles)


if __name__ == "__main__":
    main()
