# Whoosh Information Retrieval Project

Minimal project scaffold for building an information retrieval system with Whoosh.

## Structure

```text
.
├── configs/              # Optional config files
├── data/
│   ├── raw/              # Source corpus files
│   ├── processed/        # Derived or cleaned corpus files
│   ├── queries/          # Query or evaluation files
│   ├── questions.txt     # Existing sample questions
│   ├── wiki-data.txt     # Existing sample corpus data
│   └── wiki-example.txt  # Existing sample corpus data
├── index/
│   └── whoosh_index/     # Generated Whoosh index files
├── src/
│   ├── schema.py         # Schema definition
│   ├── build_index.py    # Index creation entry point
│   └── search.py         # Search entry point
└── tests/
    └── test_search.py    # Minimal placeholder test
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Next Steps

1. Define a real Whoosh schema in `src/schema.py`.
2. Implement indexing logic in `src/build_index.py`.
3. Implement query execution in `src/search.py`.
4. Put raw corpus files in `data/raw/` or adapt the code to use the existing files in `data/`.
5. Build the index into `index/whoosh_index/`.
6. Add real tests for indexing and retrieval behavior.

## Running Tests

```bash
pytest -q
```
