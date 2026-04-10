"""Schema definitions for the Whoosh index."""

from whoosh.analysis import RegexTokenizer
from whoosh.fields import ID, NUMERIC, STORED, TEXT, Schema


BODY_ANALYZER = RegexTokenizer(r"[^ ]+")


def get_schema() -> Schema:
    """Return the Whoosh schema for tokenized wiki articles."""
    return Schema(
        title=STORED,
        body=TEXT(stored=True, analyzer=BODY_ANALYZER),
        source_file=ID(stored=True),
        article_index=NUMERIC(stored=True, sortable=True),
        is_redirect=NUMERIC(stored=True),
    )
