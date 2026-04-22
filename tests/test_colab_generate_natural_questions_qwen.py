from src.colab_generate_natural_questions_qwen import parse_questions_payload


def test_parse_questions_payload_reads_json_object():
    payload = (
        '{"questions":['
        '"What newspaper dominates Washington, D.C., circulation?",'
        '"Which major newspaper is the leading paper in the U.S. capital?",'
        '"What is the top newspaper in the nation\\u2019s capital city?",'
        '"Which Washington, D.C., paper ranks among the top 10 in U.S. circulation?",'
        '"What newspaper is the dominant daily paper in Washington, D.C.?"'
        "]}"
    )

    questions = parse_questions_payload(payload)

    assert len(questions) == 5
    assert questions[0] == "What newspaper dominates Washington, D.C., circulation?"


def test_parse_questions_payload_falls_back_to_json_substring():
    payload = (
        "Here is the JSON:\n"
        '{"questions":['
        '"Question one?",'
        '"Question two?",'
        '"Question three?",'
        '"Question four?",'
        '"Question five?"'
        "]}"
    )

    questions = parse_questions_payload(payload)

    assert questions == [
        "Question one?",
        "Question two?",
        "Question three?",
        "Question four?",
        "Question five?",
    ]
