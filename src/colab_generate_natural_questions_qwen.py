"""Generate five natural-language questions per Jeopardy clue with Qwen3-4B."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import time


BASE_DIR = Path.cwd()
DRIVE_TEMP_DIR = Path("/content/drive/MyDrive/temp")
LOCAL_INPUT_JSON_PATH = BASE_DIR / "questions.json"
LOCAL_INPUT_TEXT_PATH = BASE_DIR / "questions.txt"
DRIVE_INPUT_JSON_PATH = DRIVE_TEMP_DIR / "questions.json"
DRIVE_INPUT_TEXT_PATH = DRIVE_TEMP_DIR / "questions.txt"
OUTPUT_PATH = BASE_DIR / "questions_natural_qwen3_4b.json"

MODEL_NAME = "Qwen/Qwen3-4B"
QUESTIONS_TO_PROCESS = None
SAVE_EVERY = 1
PROGRESS_EVERY = 10
MAX_NEW_TOKENS = 5000
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True
ENABLE_THINKING = True
MAX_RETRIES = 3
TORCH_DTYPE_NAME = "bfloat16"

LEADING_ENUMERATION_RE = re.compile(r"^\s*(?:\d+[\.\)]\s*|[-*]\s*)")


def ensure_generation_dependencies() -> None:
    packages = [
        "transformers>=4.51.0",
        "accelerate>=0.30.0",
        "sentencepiece",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def parse_questions_text_file(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    questions = []

    for index in range(0, len(lines), 4):
        if index + 2 >= len(lines):
            break
        category = lines[index].strip()
        clue = lines[index + 1].strip()
        answer = lines[index + 2].strip()
        if not category or not clue:
            continue
        questions.append(
            {
                "category": category,
                "clue": clue,
                "answer": answer,
            }
        )

    return questions


def load_questions() -> list[dict[str, str]]:
    if LOCAL_INPUT_JSON_PATH.exists():
        return json.loads(LOCAL_INPUT_JSON_PATH.read_text(encoding="utf-8"))
    if LOCAL_INPUT_TEXT_PATH.exists():
        return parse_questions_text_file(LOCAL_INPUT_TEXT_PATH)
    if DRIVE_INPUT_JSON_PATH.exists():
        return json.loads(DRIVE_INPUT_JSON_PATH.read_text(encoding="utf-8"))
    if DRIVE_INPUT_TEXT_PATH.exists():
        return parse_questions_text_file(DRIVE_INPUT_TEXT_PATH)

    raise FileNotFoundError(
        "Could not find questions.json or questions.txt in the current directory or "
        "/content/drive/MyDrive/temp."
    )


def clean_generated_question(text: str) -> str:
    text = LEADING_ENUMERATION_RE.sub("", text.strip())
    text = " ".join(text.split())
    return text


def normalize_questions(candidate_questions: list[str]) -> list[str]:
    normalized = []
    seen = set()

    for candidate in candidate_questions:
        cleaned = clean_generated_question(candidate)
        normalized_key = cleaned.casefold()
        if not cleaned or normalized_key in seen:
            continue
        seen.add(normalized_key)
        normalized.append(cleaned)

    return normalized[:5]


def extract_json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped]

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    first_bracket = stripped.find("[")
    last_bracket = stripped.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
        candidates.append(stripped[first_bracket : last_bracket + 1])

    return candidates


def parse_questions_payload(text: str) -> list[str]:
    for candidate in extract_json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            questions = parsed.get("questions")
        else:
            questions = parsed

        if isinstance(questions, list):
            normalized = normalize_questions([str(item) for item in questions])
            if len(normalized) == 5:
                return normalized

    fallback_questions = []
    for line in text.splitlines():
        cleaned = clean_generated_question(line)
        if not cleaned:
            continue
        if cleaned.startswith("{") or cleaned.startswith("[") or cleaned.endswith("]"):
            continue
        fallback_questions.append(cleaned)

    normalized = normalize_questions(fallback_questions)
    if len(normalized) == 5:
        return normalized

    raise ValueError(f"Could not parse exactly 5 questions from model output: {text}")


def build_messages(question: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You rewrite Jeopardy-style clues into natural search and QA questions. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write exactly 5 natural-language questions from the category and clue.\n"
                "Each question must preserve the same intended answer as the original category and clue.\n"
                "The category may provide the topic, answer type, or special instruction.\n"
                "The clue provides the hint, fact, or indirect cue used to identify the same answer.\n"
                "Ask questions with the same answer as the original clue.\n"
                "Use only the information in the category and clue.\n"
                "Do not reveal the answer in the question.\n\n"
                f"Category: {question['category']}\n"
                f"Clue: {question['clue']}\n\n"
                "Return exactly this JSON shape and nothing else:\n"
                '{"questions":["...","...","...","...","..."]}\n\n'
            ),
        },
    ]


def load_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, TORCH_DTYPE_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model


def generate_questions_for_item(tokenizer, model, question: dict[str, str]) -> tuple[list[str], str]:
    prompt_text = tokenizer.apply_chat_template(
        build_messages(question),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return parse_questions_payload(raw_output), raw_output


def load_existing_results() -> list[dict]:
    if not OUTPUT_PATH.exists():
        return []
    return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))


def write_results(rows: list[dict]) -> None:
    OUTPUT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_generation_dependencies()

    questions = load_questions()
    if QUESTIONS_TO_PROCESS is not None:
        questions = questions[:QUESTIONS_TO_PROCESS]

    existing_rows = load_existing_results()
    start_index = len(existing_rows)

    if start_index >= len(questions):
        print(f"Output already contains {start_index} rows: {OUTPUT_PATH}")
        return

    print(f"Loading model: {MODEL_NAME}")
    tokenizer, model = load_model_and_tokenizer()
    print(f"Writing output to: {OUTPUT_PATH}")

    start_time = time.time()
    rows = list(existing_rows)
    total = len(questions)

    for question_number, question in enumerate(questions[start_index:], start=start_index + 1):
        question_start_time = time.time()
        last_error = None
        generated_questions = None
        raw_output = ""

        for _attempt in range(1, MAX_RETRIES + 1):
            try:
                generated_questions, raw_output = generate_questions_for_item(
                    tokenizer,
                    model,
                    question,
                )
                break
            except Exception as error:
                last_error = error
                raw_output = str(error)

        if generated_questions is None:
            raise RuntimeError(
                f"Failed to generate valid questions for item {question_number} after "
                f"{MAX_RETRIES} attempts: {last_error}"
            ) from last_error

        rows.append(
            {
                "question_index": question_number - 1,
                "category": question["category"],
                "clue": question["clue"],
                "answer": question.get("answer", ""),
                "natural_questions": generated_questions,
                "model_name": MODEL_NAME,
                "raw_model_output": raw_output,
            }
        )

        print(f"[question {question_number}] Category: {question['category']}")
        print(f"[question {question_number}] Clue: {question['clue']}")
        for generated_index, generated_question in enumerate(generated_questions, start=1):
            print(f"[question {question_number}] Generated {generated_index}: {generated_question}")

        if question_number % SAVE_EVERY == 0:
            write_results(rows)

        if question_number % PROGRESS_EVERY == 0 or question_number == total:
            elapsed_minutes = (time.time() - start_time) / 60.0
            question_elapsed_seconds = time.time() - question_start_time
            print(
                f"[colab_generate_natural_questions_qwen] Questions: {question_number}/{total} | "
                f"elapsed: {elapsed_minutes:.2f} min | last question: {question_elapsed_seconds:.2f}s"
            )

    write_results(rows)
    total_minutes = (time.time() - start_time) / 60.0
    print(
        f"[colab_generate_natural_questions_qwen] Finished {len(rows)} questions in "
        f"{total_minutes:.2f} min"
    )


main()
