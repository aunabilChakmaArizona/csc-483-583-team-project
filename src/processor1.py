"""Simple utilities for parsing and storing wiki articles."""

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data/raw/wiki-subset-20140602"
DEFAULT_DB_PATH = PROJECT_ROOT / "data/processed/wiki_articles.sqlite3"
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "data/raw/questions.txt"
DEFAULT_QUESTIONS_JSON_PATH = PROJECT_ROOT / "data/processed/questions.json"


@dataclass(frozen=True)
class WikiArticle:
    title: str
    body: str
    source_file: str
    article_index: int

    @property
    def is_redirect(self) -> bool:
        return self.body.startswith("#REDIRECT")


@dataclass(frozen=True)
class JeopardyQuestion:
    category: str
    clue: str
    answer: str


def is_article_header(line: str) -> bool:
    line = line.strip()
    return line.startswith("[[") and line.endswith("]]") and len(line) > 4


def get_article_title(line: str) -> str:
    return line.strip()[2:-2]


def make_article(title: str, lines: list[str], source_file: str, article_index: int) -> WikiArticle:
    return WikiArticle(
        title=title,
        body="\n".join(lines).strip(),
        source_file=source_file,
        article_index=article_index,
    )


def parse_articles_from_text(text: str, source_file: str = "<memory>") -> list[WikiArticle]:
    articles = []
    title = None
    body_lines = []
    article_index = 0
    start_time = time.time()

    print(f"[processor] Start file: {source_file}")

    for line in text.splitlines():
        if is_article_header(line):
            if title is not None:
                articles.append(make_article(title, body_lines, source_file, article_index))
                article_index += 1
                if article_index % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[processor] File: {source_file} | Articles: {article_index} | Elapsed: {elapsed:.2f}s"
                    )
            title = get_article_title(line)
            body_lines = []
        elif title is not None:
            body_lines.append(line)

    if title is not None:
        articles.append(make_article(title, body_lines, source_file, article_index))

    elapsed = time.time() - start_time
    print(
        f"[processor] Finished file: {source_file} | Articles: {len(articles)} | Elapsed: {elapsed:.2f}s"
    )

    return articles


def iter_articles_from_file(path: Path):
    text = path.read_text(encoding="utf-8")
    for article in parse_articles_from_text(text, source_file=path.name):
        yield article


def iter_articles(raw_dir: Path = DEFAULT_RAW_DIR):
    for path in sorted(raw_dir.glob("*.txt")):
        yield from iter_articles_from_file(path)


def initialize_database(db_path: Path = DEFAULT_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                source_file TEXT NOT NULL,
                article_index INTEGER NOT NULL,
                is_redirect INTEGER NOT NULL,
                UNIQUE(source_file, article_index)
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_articles_title ON articles(title)")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_file, article_index)"
        )
        connection.commit()


def article_row(article: WikiArticle) -> tuple[str, str, str, int, int]:
    return (
        article.title,
        article.body,
        article.source_file,
        article.article_index,
        int(article.is_redirect),
    )


def write_batch(connection: sqlite3.Connection, batch: list[tuple[str, str, str, int, int]]) -> None:
    if not batch:
        return

    connection.executemany(
        """
        INSERT OR REPLACE INTO articles
        (title, body, source_file, article_index, is_redirect)
        VALUES (?, ?, ?, ?, ?)
        """,
        batch,
    )


def materialize_articles_to_sqlite(
    raw_dir: Path = DEFAULT_RAW_DIR,
    db_path: Path = DEFAULT_DB_PATH,
    batch_size: int = 1000,
) -> int:
    initialize_database(db_path)
    total = 0
    batch = []

    with sqlite3.connect(db_path) as connection:
        for article in iter_articles(raw_dir):
            batch.append(article_row(article))
            if len(batch) >= batch_size:
                write_batch(connection, batch)
                total += len(batch)
                batch = []

        write_batch(connection, batch)
        total += len(batch)
        connection.commit()

    return total


def parse_questions_file(path: Path = DEFAULT_QUESTIONS_PATH) -> list[JeopardyQuestion]:
    lines = path.read_text(encoding="utf-8").splitlines()
    questions = []

    for i in range(0, len(lines), 4):
        category = lines[i].strip()
        clue = lines[i + 1].strip()
        answer = lines[i + 2].strip()
        questions.append(
            JeopardyQuestion(
                category=category,
                clue=clue,
                answer=answer,
            )
        )

    return questions


def write_questions_to_json(
    questions_path: Path = DEFAULT_QUESTIONS_PATH,
    output_path: Path = DEFAULT_QUESTIONS_JSON_PATH,
) -> int:
    questions = parse_questions_file(questions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump([question.__dict__ for question in questions], file, indent=2)

    return len(questions)


if __name__ == "__main__":
    total_articles = materialize_articles_to_sqlite()
    total_questions = write_questions_to_json()
    print(f"Stored {total_articles} articles in {DEFAULT_DB_PATH}")
    print(f"Stored {total_questions} questions in {DEFAULT_QUESTIONS_JSON_PATH}")
