"""Index-building entry points."""

from pathlib import Path

try:
    from src.processor4_index import (
        DEFAULT_INDEX_DIR,
        DEFAULT_INPUT_DB_PATH,
        materialize_whoosh_index,
    )
except ModuleNotFoundError:
    from processor4_index import DEFAULT_INDEX_DIR, DEFAULT_INPUT_DB_PATH, materialize_whoosh_index


def build_index(
    input_db_path: Path = DEFAULT_INPUT_DB_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    batch_size: int = 1000,
) -> int:
    """Create or refresh the Whoosh index."""
    return materialize_whoosh_index(
        input_db_path=input_db_path,
        index_dir=index_dir,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    total = build_index()
    print(f"Indexed {total} articles in {DEFAULT_INDEX_DIR}")
