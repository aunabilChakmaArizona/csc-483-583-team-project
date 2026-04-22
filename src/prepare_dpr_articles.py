"""Export non-redirect wiki articles with DPR-ready passage variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import gzip
import json
from pathlib import Path
import sqlite3
import time

try:
    from src.processor1_parse import DEFAULT_DB_PATH as DEFAULT_INPUT_DB_PATH
    from src.processor4_lead_index_common import extract_paragraphs, split_sentences
except ModuleNotFoundError:
    from processor1_parse import DEFAULT_DB_PATH as DEFAULT_INPUT_DB_PATH
    from processor4_lead_index_common import extract_paragraphs, split_sentences


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/processed/dpr_articles.jsonl.gz"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data/processed/dpr_articles_summary.json"


@dataclass(frozen=True)
class WikiArticle:
    title: str
    body: str
    source_file: str
    article_index: int
    is_redirect: int


@dataclass(frozen=True)
class DprArticleRecord:
    doc_id: str
    title: str
    source_file: str
    article_index: int
    first_sentence_text: str
    first_two_sentences_text: str
    first_paragraph_text: str
    entire_article_text: str


@dataclass(frozen=True)
class ExportSummary:
    input_db_path: str
    output_path: str
    total_articles_scanned: int
    exported_articles: int
    skipped_redirects: int
    articles_missing_first_sentence: int
    articles_with_fewer_than_two_sentences: int
    articles_missing_first_paragraph: int
    articles_missing_entire_article: int
    elapsed_seconds: float


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


def get_article_sentences(article_body: str) -> list[str]:
    paragraphs = extract_paragraphs(article_body)
    if not paragraphs:
        return []

    return split_sentences(paragraphs[0])


def get_first_sentence(article_body: str) -> str:
    sentences = get_article_sentences(article_body)
    return sentences[0] if sentences else ""


def get_first_two_sentences(article_body: str) -> str:
    sentences = get_article_sentences(article_body)
    return " ".join(sentences[:2])


def get_first_paragraph(article_body: str) -> str:
    paragraphs = extract_paragraphs(article_body)
    return paragraphs[0] if paragraphs else ""


def get_entire_article(article_body: str) -> str:
    return "\n\n".join(extract_paragraphs(article_body))


def make_doc_id(article: WikiArticle) -> str:
    return f"{article.source_file}:{article.article_index}"


def prepare_article_record(article: WikiArticle) -> DprArticleRecord:
    return DprArticleRecord(
        doc_id=make_doc_id(article),
        title=article.title,
        source_file=article.source_file,
        article_index=article.article_index,
        first_sentence_text=get_first_sentence(article.body),
        first_two_sentences_text=get_first_two_sentences(article.body),
        first_paragraph_text=get_first_paragraph(article.body),
        entire_article_text=get_entire_article(article.body),
    )


def open_output_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")

    return path.open("w", encoding="utf-8")


def export_dpr_articles(
    input_db_path: Path = DEFAULT_INPUT_DB_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    summary_path: Path | None = DEFAULT_SUMMARY_PATH,
) -> ExportSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    total_articles_scanned = 0
    exported_articles = 0
    skipped_redirects = 0
    articles_missing_first_sentence = 0
    articles_with_fewer_than_two_sentences = 0
    articles_missing_first_paragraph = 0
    articles_missing_entire_article = 0
    start_time = time.time()

    with open_output_file(output_path) as output_file:
        for article in iter_articles_from_db(input_db_path):
            total_articles_scanned += 1
            if article.is_redirect:
                skipped_redirects += 1
                continue

            record = prepare_article_record(article)
            if not record.first_sentence_text:
                articles_missing_first_sentence += 1
            if len(get_article_sentences(article.body)) < 2:
                articles_with_fewer_than_two_sentences += 1
            if not record.first_paragraph_text:
                articles_missing_first_paragraph += 1
            if not record.entire_article_text:
                articles_missing_entire_article += 1

            output_file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            exported_articles += 1

            if exported_articles % 1000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[prepare_dpr_articles] Exported: {exported_articles} | "
                    f"Skipped redirects: {skipped_redirects} | Elapsed: {elapsed:.2f}s"
                )

    elapsed = time.time() - start_time
    summary = ExportSummary(
        input_db_path=str(input_db_path),
        output_path=str(output_path),
        total_articles_scanned=total_articles_scanned,
        exported_articles=exported_articles,
        skipped_redirects=skipped_redirects,
        articles_missing_first_sentence=articles_missing_first_sentence,
        articles_with_fewer_than_two_sentences=articles_with_fewer_than_two_sentences,
        articles_missing_first_paragraph=articles_missing_first_paragraph,
        articles_missing_entire_article=articles_missing_entire_article,
        elapsed_seconds=round(elapsed, 3),
    )

    if summary_path is not None:
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(asdict(summary), summary_file, indent=2)

    print(
        f"[prepare_dpr_articles] Done | Exported: {exported_articles} | "
        f"Skipped redirects: {skipped_redirects} | Elapsed: {elapsed:.2f}s"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export non-redirect wiki articles with four DPR passage variants."
    )
    parser.add_argument(
        "--input-db",
        type=Path,
        default=DEFAULT_INPUT_DB_PATH,
        help=f"SQLite database containing parsed wiki articles (default: {DEFAULT_INPUT_DB_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination JSONL file for Colab upload (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Destination JSON summary file (default: {DEFAULT_SUMMARY_PATH})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_dpr_articles(
        input_db_path=args.input_db,
        output_path=args.output,
        summary_path=args.summary_output,
    )
