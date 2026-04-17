"""Shared helpers for building Cole-style lead-only indexes."""

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import time

from whoosh import index as whoosh_index
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import STORED, TEXT, Schema
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F

try:
    from src.processor1_parse import DEFAULT_DB_PATH as DEFAULT_INPUT_DB_PATH
    from src.processor2_clean import (
        remove_categories_lines,
        remove_file_markup,
        remove_redirect_lines,
        remove_references,
        remove_section_headers,
        remove_templates,
    )
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_DB_PATH as DEFAULT_INPUT_DB_PATH
    from processor2_clean import (
        remove_categories_lines,
        remove_file_markup,
        remove_redirect_lines,
        remove_references,
        remove_section_headers,
        remove_templates,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
INLINE_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class WikiArticle:
    title: str
    body: str
    source_file: str
    article_index: int
    is_redirect: int


def get_schema() -> Schema:
    analyzer = StandardAnalyzer()
    return Schema(
        title=TEXT(stored=True, analyzer=analyzer),
        body=TEXT(stored=False, analyzer=analyzer),
        source_file=STORED(),
        article_index=STORED(),
        is_redirect=STORED(),
    )


def initialize_index_directory(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)


def create_index(index_dir: Path):
    initialize_index_directory(index_dir)
    return whoosh_index.create_in(index_dir, get_schema())


def open_index(index_dir: Path):
    if not index_dir.exists() or not whoosh_index.exists_in(index_dir):
        raise FileNotFoundError(f"Whoosh index not found at {index_dir}. Build it first.")

    return whoosh_index.open_dir(index_dir)


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


def clean_body_preserving_paragraphs(text: str) -> str:
    cleaned = text
    cleaned = remove_redirect_lines(cleaned)
    cleaned = remove_templates(cleaned)
    cleaned = remove_references(cleaned)
    cleaned = remove_file_markup(cleaned)
    cleaned = remove_categories_lines(cleaned)
    cleaned = remove_section_headers(cleaned)

    cleaned_lines = [line.rstrip() for line in cleaned.splitlines()]
    return "\n".join(cleaned_lines).strip()


def normalize_inline_whitespace(text: str) -> str:
    return INLINE_WHITESPACE_RE.sub(" ", text).strip()


def extract_paragraphs(text: str) -> list[str]:
    cleaned = clean_body_preserving_paragraphs(text)
    if not cleaned:
        return []

    return [
        normalize_inline_whitespace(paragraph)
        for paragraph in PARAGRAPH_SPLIT_RE.split(cleaned)
        if normalize_inline_whitespace(paragraph)
    ]


def split_sentences(text: str) -> list[str]:
    normalized = normalize_inline_whitespace(text)
    if not normalized:
        return []

    return [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(normalized) if sentence.strip()]


def extract_lead_text(text: str, lead_mode: str) -> str:
    paragraphs = extract_paragraphs(text)
    if not paragraphs:
        return ""

    first_paragraph = paragraphs[0]
    if lead_mode == "first_paragraph":
        return first_paragraph

    sentences = split_sentences(first_paragraph)
    if lead_mode == "first_sentence":
        return sentences[0] if sentences else ""

    if lead_mode == "first_two_sentences":
        return " ".join(sentences[:2])

    raise ValueError(f"Unsupported lead mode: {lead_mode}")


def materialize_lead_index(
    lead_mode: str,
    input_db_path: Path,
    index_dir: Path,
    batch_size: int = 1000,
) -> int:
    index = create_index(index_dir)
    total = 0
    skipped_redirects = 0
    start_time = time.time()

    writer = index.writer()
    for article in iter_articles_from_db(input_db_path):
        if article.is_redirect:
            skipped_redirects += 1
            continue

        lead_text = extract_lead_text(article.body, lead_mode)
        writer.add_document(
            title=article.title,
            body=lead_text,
            source_file=article.source_file,
            article_index=article.article_index,
            is_redirect=article.is_redirect,
        )
        total += 1

        if total % batch_size == 0:
            elapsed = time.time() - start_time
            print(f"[processor4_{lead_mode}] Indexed: {total} | Elapsed: {elapsed:.2f}s")

    print(f"[processor4_{lead_mode}] Committing index to disk (this may take a moment)...")
    writer.commit()

    elapsed = time.time() - start_time
    print(
        f"[processor4_{lead_mode}] Done | Indexed: {total} | "
        f"Skipped redirects: {skipped_redirects} | Elapsed: {elapsed:.2f}s"
    )
    return total


def search_index(query: str, index_dir: Path, limit: int = 10) -> list[str]:
    index = open_index(index_dir)
    with index.searcher(weighting=BM25F()) as searcher:
        parser = QueryParser("body", schema=index.schema)
        results = searcher.search(parser.parse(query), limit=limit)
        return [hit["title"] for hit in results]
