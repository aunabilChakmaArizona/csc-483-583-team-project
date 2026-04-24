from src.colab_build_qwen_summary_dpr_faiss import (
    SUMMARY_MIN_SOURCE_CHARS,
    build_summary_row,
    parse_summary_payload,
    should_summarize_row,
)


def test_parse_summary_payload_reads_json_object():
    payload = (
        '{"summary":"The Washington Post is a major daily newspaper based in Washington, D.C. '
        "It is widely known for its political reporting and national coverage.\"}"
    )

    summary = parse_summary_payload(payload)

    assert summary == (
        "The Washington Post is a major daily newspaper based in Washington, D.C. "
        "It is widely known for its political reporting and national coverage."
    )


def test_parse_summary_payload_falls_back_to_json_substring():
    payload = (
        "Here is the result:\n"
        '{"summary":"A retrieval-focused summary that keeps the main names, places, and dates."}'
    )

    summary = parse_summary_payload(payload)

    assert summary == "A retrieval-focused summary that keeps the main names, places, and dates."


def test_should_summarize_row_uses_source_char_threshold():
    short_row = {"entire_article_text": "a" * SUMMARY_MIN_SOURCE_CHARS}
    long_row = {"entire_article_text": "a" * (SUMMARY_MIN_SOURCE_CHARS + 1)}

    assert should_summarize_row(short_row) is False
    assert should_summarize_row(long_row) is True


def test_build_summary_row_accepts_passthrough_model_name():
    row = {
        "doc_id": "doc-1",
        "title": "Short Article",
        "source_file": "sample.txt",
        "article_index": 1,
        "entire_article_text": "Short body text.",
    }

    summary_row = build_summary_row(
        row,
        "Short body text.",
        "",
        summary_model_name="passthrough_original_body",
    )

    assert summary_row["summary_text"] == "Short body text."
    assert summary_row["summary_model_name"] == "passthrough_original_body"
