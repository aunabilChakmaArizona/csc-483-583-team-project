"""Resolve redirect articles to a canonical target in a separate SQLite database."""

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import time

try:
    from src.processor1_parse import DEFAULT_DB_PATH as DEFAULT_INPUT_DB_PATH
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_DB_PATH as DEFAULT_INPUT_DB_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DB_PATH = PROJECT_ROOT / "data/processed/wiki_redirects.sqlite3"
REDIRECT_TARGET_RE = re.compile(r"^\s*#redirect\s*\[\[(.+?)\]\]", re.IGNORECASE | re.MULTILINE)
WHITESPACE_RE = re.compile(r"\s+")
MAX_REDIRECT_HOPS = 20


@dataclass(frozen=True)
class WikiArticle:
    title: str
    body: str
    source_file: str
    article_index: int
    is_redirect: int


@dataclass(frozen=True)
class RedirectTarget:
    title: str
    section: str


@dataclass(frozen=True)
class RedirectMapping:
    redirect_title: str
    redirect_source_file: str
    redirect_article_index: int
    target_title: str
    target_section: str
    resolved_title: str
    resolved_source_file: str
    resolved_article_index: int
    resolution_status: str
    hops: int


def normalize_title(title: str) -> str:
    title = title.replace("_", " ").strip()
    return WHITESPACE_RE.sub(" ", title)


def normalize_title_key(title: str) -> str:
    return normalize_title(title).casefold()


def extract_redirect_target(body: str) -> RedirectTarget | None:
    match = REDIRECT_TARGET_RE.search(body)
    if match is None:
        return None

    link_target = match.group(1).strip()
    target_text = link_target.split("|", 1)[0].strip()
    target_title, _, target_section = target_text.partition("#")
    normalized_title = normalize_title(target_title)

    if not normalized_title:
        return None

    return RedirectTarget(title=normalized_title, section=target_section.strip())


def iter_articles_from_db(db_path: Path = DEFAULT_INPUT_DB_PATH):
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            SELECT title, body, source_file, article_index, is_redirect
            FROM articles
            ORDER BY source_file, article_index
            """
        )
        for row in cursor:
            yield WikiArticle(*row)


def build_title_lookup(articles: list[WikiArticle]) -> dict[str, WikiArticle]:
    lookup = {}

    for article in articles:
        lookup.setdefault(normalize_title_key(article.title), article)

    return lookup


def resolve_redirect(article: WikiArticle, title_lookup: dict[str, WikiArticle]) -> RedirectMapping:
    target = extract_redirect_target(article.body)

    if target is None:
        return RedirectMapping(
            redirect_title=article.title,
            redirect_source_file=article.source_file,
            redirect_article_index=article.article_index,
            target_title="",
            target_section="",
            resolved_title="",
            resolved_source_file="",
            resolved_article_index=-1,
            resolution_status="invalid_target",
            hops=0,
        )

    current_target = target
    visited_keys = {normalize_title_key(article.title)}
    hops = 0

    while True:
        target_key = normalize_title_key(current_target.title)
        target_article = title_lookup.get(target_key)

        if target_article is None:
            return RedirectMapping(
                redirect_title=article.title,
                redirect_source_file=article.source_file,
                redirect_article_index=article.article_index,
                target_title=current_target.title,
                target_section=current_target.section,
                resolved_title="",
                resolved_source_file="",
                resolved_article_index=-1,
                resolution_status="missing_target",
                hops=hops,
            )

        if target_key in visited_keys:
            return RedirectMapping(
                redirect_title=article.title,
                redirect_source_file=article.source_file,
                redirect_article_index=article.article_index,
                target_title=target.title,
                target_section=target.section,
                resolved_title=target_article.title,
                resolved_source_file=target_article.source_file,
                resolved_article_index=target_article.article_index,
                resolution_status="cycle",
                hops=hops,
            )

        hops += 1
        if hops > MAX_REDIRECT_HOPS:
            return RedirectMapping(
                redirect_title=article.title,
                redirect_source_file=article.source_file,
                redirect_article_index=article.article_index,
                target_title=target.title,
                target_section=target.section,
                resolved_title=target_article.title,
                resolved_source_file=target_article.source_file,
                resolved_article_index=target_article.article_index,
                resolution_status="max_hops_exceeded",
                hops=hops,
            )

        if not target_article.is_redirect:
            return RedirectMapping(
                redirect_title=article.title,
                redirect_source_file=article.source_file,
                redirect_article_index=article.article_index,
                target_title=target.title,
                target_section=target.section,
                resolved_title=target_article.title,
                resolved_source_file=target_article.source_file,
                resolved_article_index=target_article.article_index,
                resolution_status="resolved",
                hops=hops,
            )

        visited_keys.add(target_key)
        next_target = extract_redirect_target(target_article.body)
        if next_target is None:
            return RedirectMapping(
                redirect_title=article.title,
                redirect_source_file=article.source_file,
                redirect_article_index=article.article_index,
                target_title=target.title,
                target_section=target.section,
                resolved_title=target_article.title,
                resolved_source_file=target_article.source_file,
                resolved_article_index=target_article.article_index,
                resolution_status="invalid_chain_target",
                hops=hops,
            )

        current_target = next_target


def initialize_output_database(db_path: Path = DEFAULT_OUTPUT_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS redirects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                redirect_title TEXT NOT NULL,
                redirect_source_file TEXT NOT NULL,
                redirect_article_index INTEGER NOT NULL,
                target_title TEXT NOT NULL,
                target_section TEXT NOT NULL,
                resolved_title TEXT NOT NULL,
                resolved_source_file TEXT NOT NULL,
                resolved_article_index INTEGER NOT NULL,
                resolution_status TEXT NOT NULL,
                hops INTEGER NOT NULL,
                UNIQUE(redirect_source_file, redirect_article_index)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_redirects_title ON redirects(redirect_title)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_redirects_resolved_title ON redirects(resolved_title)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_redirects_status ON redirects(resolution_status)"
        )
        connection.commit()


def redirect_row(mapping: RedirectMapping) -> tuple[str, str, int, str, str, str, str, int, str, int]:
    return (
        mapping.redirect_title,
        mapping.redirect_source_file,
        mapping.redirect_article_index,
        mapping.target_title,
        mapping.target_section,
        mapping.resolved_title,
        mapping.resolved_source_file,
        mapping.resolved_article_index,
        mapping.resolution_status,
        mapping.hops,
    )


def write_batch(
    connection: sqlite3.Connection,
    batch: list[tuple[str, str, int, str, str, str, str, int, str, int]],
) -> None:
    if not batch:
        return

    connection.executemany(
        """
        INSERT OR REPLACE INTO redirects (
            redirect_title,
            redirect_source_file,
            redirect_article_index,
            target_title,
            target_section,
            resolved_title,
            resolved_source_file,
            resolved_article_index,
            resolution_status,
            hops
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )


def materialize_redirect_mappings(
    input_db_path: Path = DEFAULT_INPUT_DB_PATH,
    output_db_path: Path = DEFAULT_OUTPUT_DB_PATH,
    batch_size: int = 1000,
) -> int:
    initialize_output_database(output_db_path)
    articles = list(iter_articles_from_db(input_db_path))
    title_lookup = build_title_lookup(articles)
    total_redirects = 0
    batch = []
    start_time = time.time()

    with sqlite3.connect(output_db_path) as output_connection:
        for article in articles:
            if not article.is_redirect:
                continue

            batch.append(redirect_row(resolve_redirect(article, title_lookup)))
            total_redirects += 1

            if total_redirects % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"[processor_redirect] Redirects: {total_redirects} | Elapsed: {elapsed:.2f}s")

            if len(batch) >= batch_size:
                write_batch(output_connection, batch)
                batch = []

        write_batch(output_connection, batch)
        output_connection.commit()

    elapsed = time.time() - start_time
    print(f"[processor_redirect] Finished | Redirects: {total_redirects} | Elapsed: {elapsed:.2f}s")
    return total_redirects


if __name__ == "__main__":
    total = materialize_redirect_mappings()
    print(f"Stored {total} redirect mappings in {DEFAULT_OUTPUT_DB_PATH}")
