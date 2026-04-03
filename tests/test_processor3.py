from src.processor3 import tokenize_body


def test_tokenize_body_applies_requested_pipeline():
    text = """The quick, brown-fox_jumps and river.
_value_ data-driven_test, U.S.
"""

    stop_words = {"the", "and"}

    assert tokenize_body(text, stop_words) == [
        "quick",
        "brown",
        "fox",
        "jumps",
        "river",
        "value",
        "data",
        "driven",
        "test",
        "us",
    ]
