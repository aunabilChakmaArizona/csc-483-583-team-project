from src.processor2_clean import clean_body


def test_clean_body_removes_requested_wiki_markup():
    text = """#REDIRECT Some Other Article
CATEGORIES: Sample category, Another category
==History==
Useful intro text.
[tpl]cite web|url=http://example.com|title=Example[/tpl]
[ref]Reference text[/ref]
[[File:example.png|thumb|caption]]
===Legacy===
More useful text.
"""

    assert clean_body(text) == "Useful intro text.\nMore useful text."
