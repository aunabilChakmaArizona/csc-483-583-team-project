"""Generate indirect helper questions for the first 100 Jeopardy clues with Qwen 4B."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time


# Configure everything here.
QUESTIONS_PATH = "questions.json"
OUTPUT_PATH = "qwen_related_questions_100.json"
MODEL_NAME = "Qwen/Qwen3-4B"
QUESTION_LIMIT = 100
HELPER_QUESTION_COUNT = 5
MAX_NEW_TOKENS = 1000

LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)]\s*)")
WHITESPACE_RE = re.compile(r"\s+")
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
UNQUOTED_JSON_KEY_RE = re.compile(r'([{\s,])(\d+)(\s*:)')

PROMPT_TEMPLATE = """Write 5 short, diverse questions that indirectly help solve this clue.

Requirements:
-Do not directly name the answer.
-Avoid simple paraphrases.

CATEGORY: {category}
CLUE: {clue}

Output only one valid JSON object with exactly 5 entries using the format: 
{{"1":"...","2":"...","3":"...","4":"...","5":"..."}}
"""


@dataclass(frozen=True)
class JeopardyQuestion:
    category: str
    clue: str
    answer: str

def load_questions(path: str | Path) -> list[JeopardyQuestion]:
    path = Path(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [JeopardyQuestion(**row) for row in rows]


def build_prompt(question: JeopardyQuestion) -> str:
    return PROMPT_TEMPLATE.format(
        category=question.category.strip(),
        clue=question.clue.strip(),
    )


def normalize_line(text: str) -> str:
    cleaned = LINE_PREFIX_RE.sub("", text.strip())
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def extract_answer_text(text: str) -> str:
    cleaned = text.strip()

    if "</think>" in cleaned.lower():
        lower_text = cleaned.lower()
        closing_index = lower_text.rfind("</think>")
        cleaned = cleaned[closing_index + len("</think>") :].strip()

    cleaned = THINK_BLOCK_RE.sub(" ", cleaned)
    cleaned = THINK_TAG_RE.sub(" ", cleaned)
    return cleaned.strip()


def normalize_json_like_text(text: str) -> str:
    normalized = text.strip()
    normalized = UNQUOTED_JSON_KEY_RE.sub(r'\1"\2"\3', normalized)
    return normalized


def extract_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    answer_text = extract_answer_text(text)

    for start_index, character in enumerate(answer_text):
        if character != "{":
            continue

        candidate = normalize_json_like_text(answer_text[start_index:])
        try:
            parsed_object, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed_object, dict):
            return parsed_object

    return None


def split_questions_from_text(text: str, question_count: int) -> list[str]:
    lines = []
    seen = set()

    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if not line:
            continue

        normalized = line.casefold()
        if normalized in seen:
            continue

        seen.add(normalized)
        lines.append(line)

    if len(lines) >= question_count:
        return lines

    sentence_lines = []
    seen = set()
    for match in re.findall(r"[^?\n]+(?:\?)", text):
        line = normalize_line(match)
        if not line:
            continue
        normalized = line.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        sentence_lines.append(line)

    return sentence_lines or lines


def extract_helper_questions(text: str, question_count: int) -> list[str]:
    cleaned_questions = []
    seen = set()
    json_payload = extract_json_object(text)

    if json_payload is not None:
        candidates = [
            str(json_payload.get(str(index), json_payload.get(index, ""))).strip()
            for index in range(1, question_count + 1)
        ]
    else:
        answer_text = extract_answer_text(text)
        candidates = split_questions_from_text(answer_text, question_count=question_count)

    for candidate in candidates:
        if not candidate:
            continue
        line = candidate if candidate.endswith("?") else f"{candidate}?"
        normalized = line.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned_questions.append(line)
        if len(cleaned_questions) >= question_count:
            break

    return cleaned_questions


def detect_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16

    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16

    return torch.device("cpu"), torch.float32


def load_model(model_name: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "transformers and torch are required to generate Qwen helper questions."
        ) from error

    device, dtype = detect_device()
    print(f"[generate_qwen_related_questions] Loading model: {model_name}")
    print(f"[generate_qwen_related_questions] Device: {device} | dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_text(
    tokenizer,
    model,
    device,
    prompt: str,
    max_new_tokens: int,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def generate_related_questions_for_dataset(
    questions: list[JeopardyQuestion],
    output_path: str | Path,
    model_name: str,
    question_count: int,
    max_new_tokens: int,
) -> list[dict]:
    output_path = Path(output_path)
    tokenizer, model, device = load_model(model_name)
    rows = []
    total = len(questions)
    start_time = time.time()

    for index, question in enumerate(questions, start=1):
        print(f"[generate_qwen_related_questions] Question {index}/{total}")
        prompt = build_prompt(question)
        raw_output = generate_text(
            tokenizer=tokenizer,
            model=model,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        answer_text = extract_answer_text(raw_output)
        json_payload = extract_json_object(raw_output)
        helper_questions = extract_helper_questions(raw_output, question_count=question_count)
        rows.append(
            {
                "question_number": index,
                "category": question.category,
                "clue": question.clue,
                "answer": question.answer,
                "prompt": prompt,
                "raw_output": raw_output,
                "answer_text": answer_text,
                "json_payload": json_payload,
                "helper_questions": helper_questions,
            }
        )

        print("=" * 80)
        print(f"[generate_qwen_related_questions] Original {index}/{total}")
        print(f"Category: {question.category}")
        print(f"Clue: {question.clue}")
        for helper_index, helper_question in enumerate(helper_questions, start=1):
            print(
                f"Generated {helper_index}/{question_count}: {helper_question}"
            )
        print("=" * 80)

        elapsed = time.time() - start_time
        print(
            f"[generate_qwen_related_questions] Stored: {index}/{total} | "
            f"Extracted: {len(helper_questions)} | Elapsed: {elapsed:.2f}s"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def run() -> None:
    questions = load_questions(QUESTIONS_PATH)[:QUESTION_LIMIT]
    rows = generate_related_questions_for_dataset(
        questions=questions,
        output_path=OUTPUT_PATH,
        model_name=MODEL_NAME,
        question_count=HELPER_QUESTION_COUNT,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(f"Stored helper questions for {len(rows)} clues in {OUTPUT_PATH}")


run()
