from src.processor1 import parse_articles_from_text


def test_parse_articles_from_text_splits_on_double_bracket_headers():
    raw_text = """[[First Article]]

First body line.
Second body line.

[[Second Article]]

#REDIRECT Something Else
"""

    articles = parse_articles_from_text(raw_text, source_file="sample.txt")

    assert len(articles) == 2
    assert articles[0].title == "First Article"
    assert articles[0].body == "First body line.\nSecond body line."
    assert articles[0].source_file == "sample.txt"
    assert articles[0].article_index == 0
    assert articles[0].is_redirect is False
    assert articles[1].title == "Second Article"
    assert articles[1].body == "#REDIRECT Something Else"
    assert articles[1].article_index == 1
    assert articles[1].is_redirect is True
