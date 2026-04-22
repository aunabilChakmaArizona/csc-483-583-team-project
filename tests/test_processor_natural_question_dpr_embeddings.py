import json
from pathlib import Path

import numpy as np

from src.processor_natural_question_dpr_embeddings import (
    combined_natural_questions_text,
    materialize_natural_question_dpr_embeddings,
)


def test_combined_natural_questions_text_joins_with_spaces():
    combined = combined_natural_questions_text(
        [
            "Question one?",
            "Question two?",
            "Question three?",
            "Question four?",
            "Question five?",
        ]
    )

    assert combined == "Question one? Question two? Question three? Question four? Question five?"


def test_materialize_natural_question_dpr_embeddings_writes_expected_arrays(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "questions_natural.json"
    output_path = tmp_path / "question_dpr_embeddings.npz"
    input_path.write_text(
        json.dumps(
            [
                {
                    "question_index": 0,
                    "category": "Sample",
                    "clue": "Sample clue",
                    "answer": "Alpha",
                    "natural_questions": [
                        "Question one?",
                        "Question two?",
                        "Question three?",
                        "Question four?",
                        "Question five?",
                    ],
                    "model_name": "Qwen/sample",
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_encode_question_texts(texts, model_name, batch_size, max_length):
        rows = []
        for index, _text in enumerate(texts):
            rows.append([float(index), float(index + 1)])
        return np.array(rows, dtype=np.float32)

    monkeypatch.setattr(
        "src.processor_natural_question_dpr_embeddings.encode_question_texts",
        fake_encode_question_texts,
    )

    total = materialize_natural_question_dpr_embeddings(
        input_path=input_path,
        output_path=output_path,
        model_name="facebook/dpr-question_encoder-single-nq-base",
        batch_size=64,
        max_length=512,
    )

    saved = np.load(output_path, allow_pickle=True)
    assert total == 1
    assert saved["natural_questions_embeddings"].shape == (1, 5, 2)
    assert saved["combined_natural_questions_embeddings"].shape == (1, 2)
    assert saved["combined_natural_questions_texts"].tolist() == [
        "Question one? Question two? Question three? Question four? Question five?"
    ]
    assert saved["source_model_names"].tolist() == ["Qwen/sample"]
