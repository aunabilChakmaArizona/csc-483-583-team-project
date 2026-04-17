"""Build a Cole-style Whoosh index using the first paragraph of each article."""

from pathlib import Path

try:
    from src.processor4_lead_index_common import (
        DEFAULT_INPUT_DB_PATH,
        materialize_lead_index,
        open_index as open_lead_index,
    )
except ModuleNotFoundError:
    from processor4_lead_index_common import (
        DEFAULT_INPUT_DB_PATH,
        materialize_lead_index,
        open_index as open_lead_index,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index/whoosh_cole_first_paragraph_index"


def materialize_whoosh_cole_first_paragraph_index(
    input_db_path: Path = DEFAULT_INPUT_DB_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    batch_size: int = 1000,
) -> int:
    return materialize_lead_index(
        lead_mode="first_paragraph",
        input_db_path=input_db_path,
        index_dir=index_dir,
        batch_size=batch_size,
    )


def open_index(index_dir: Path = DEFAULT_INDEX_DIR):
    return open_lead_index(index_dir)


if __name__ == "__main__":
    total = materialize_whoosh_cole_first_paragraph_index()
    print(f"Indexed {total} articles in {DEFAULT_INDEX_DIR}")
