"""Score exported Qwen verifier inputs on Colab.

Typical Colab usage:

    !pip install -q "transformers>=4.51.0" accelerate torch
    !python colab_score_qwen_verifier_inputs.py \
        --input qwen_verifier_inputs.jsonl \
        --output qwen_verifier_scores.jsonl \
        --chat-model-name Qwen/Qwen3-4B \
        --reranker-model-name Qwen/Qwen3-Reranker-4B \
        --device cuda \
        --batch-size 8
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re


DEFAULT_CHAT_MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
DEFAULT_OUTPUT_PATH = Path("/content/qwen_verifier_scores.jsonl")
MAX_CHAT_INPUT_TOKENS = 420
MAX_RERANKER_INPUT_TOKENS = 4096
RERANKER_INSTRUCTION = (
    "Given a Jeopardy clue rewritten as a natural question, retrieve articles that contain "
    "enough information to identify the answer."
)
TOP_K_VALUES = [1, 5, 10]
NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen yes-token verification over exported candidate rows."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--chat-model-name",
        "--model-name",
        dest="chat_model_name",
        default=DEFAULT_CHAT_MODEL_NAME,
        help=(
            "Qwen chat model used for Yes/No verification. --model-name is kept as an alias. "
            f"Default: {DEFAULT_CHAT_MODEL_NAME}."
        ),
    )
    parser.add_argument(
        "--reranker-model-name",
        default=DEFAULT_RERANKER_MODEL_NAME,
        help=f"Qwen reranker model used for relevance scoring. Default: {DEFAULT_RERANKER_MODEL_NAME}.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reranker-batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def iter_rows(path: Path, limit: int = 0):
    with path.open("r", encoding="utf-8") as input_file:
        for row_number, line in enumerate(input_file, start=1):
            if limit and row_number > limit:
                break
            if line.strip():
                yield json.loads(line)


def normalize_label(text: str) -> str:
    return NORMALIZE_RE.sub(" ", text.casefold()).strip()


def has_required_scores(row: dict, chat_model_name: str, reranker_model_name: str) -> bool:
    return (
        row.get("chat_model_name") == chat_model_name
        and row.get("reranker_model_name") == reranker_model_name
        and "qwen_chat_yes_probability" in row
        and "qwen_reranker_probability" in row
    )


def load_existing_keys(path: Path, chat_model_name: str, reranker_model_name: str) -> set[str]:
    if not path.exists():
        return set()

    keys = set()
    with path.open("r", encoding="utf-8") as output_file:
        for line in output_file:
            if not line.strip():
                continue
            row = json.loads(line)
            if has_required_scores(row, chat_model_name, reranker_model_name):
                keys.add(row["candidate_key"])
    return keys


def load_score_rows(
    path: Path,
    chat_model_name: str | None = None,
    reranker_model_name: str | None = None,
) -> list[dict]:
    if not path.exists():
        return []

    rows_by_key = {}
    with path.open("r", encoding="utf-8") as score_file:
        for line in score_file:
            if line.strip():
                row = json.loads(line)
                if chat_model_name and reranker_model_name and not has_required_scores(
                    row,
                    chat_model_name,
                    reranker_model_name,
                ):
                    continue
                rows_by_key[row["candidate_key"]] = row
    return list(rows_by_key.values())


def resolve_device(torch, requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return requested_device


def load_model(model_name: str, requested_device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(torch, requested_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
    model.eval()
    return tokenizer, model, device


def load_reranker_model(model_name: str, requested_device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(torch, requested_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
    model.eval()
    return tokenizer, model, device


def yes_no_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    yes_ids = {
        ids[-1]
        for text in ("Yes", " yes", "YES")
        for ids in [tokenizer.encode(text, add_special_tokens=False)]
        if ids
    }
    no_ids = {
        ids[-1]
        for text in ("No", " no", "NO")
        for ids in [tokenizer.encode(text, add_special_tokens=False)]
        if ids
    }
    if not yes_ids or not no_ids:
        raise RuntimeError("Could not find tokenizer ids for Yes/No labels.")
    return list(yes_ids), list(no_ids)


def build_prompt(row: dict, natural_question: str) -> str:
    return (
        f"Question: {natural_question}\n"
        f"Article: {row['evidence']}\n\n"
        "=============\n\n"
        "Does this article contain the answer to the question? "
        "Answer with exactly one word: Yes or No."
    )


def chat_text(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt + " /no_think"}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt + "\nAnswer:"


def yes_probabilities(tokenizer, model, prompts: list[str]) -> list[float]:
    import torch

    encoded = tokenizer(
        [chat_text(tokenizer, prompt) for prompt in prompts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_CHAT_INPUT_TOKENS,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits

    last_indexes = encoded["attention_mask"].sum(dim=1) - 1
    batch_indexes = torch.arange(logits.shape[0], device=logits.device)
    next_token_logits = logits[batch_indexes, last_indexes]
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    yes_logits = torch.logsumexp(next_token_logits[:, yes_ids], dim=1)
    no_logits = torch.logsumexp(next_token_logits[:, no_ids], dim=1)
    return torch.softmax(torch.stack([yes_logits, no_logits], dim=1), dim=1)[:, 0].tolist()


def reranker_text(query: str, document: str) -> str:
    return (
        f"<Instruct>: {RERANKER_INSTRUCTION}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


def reranker_prompt_tokens(tokenizer) -> tuple[list[int], list[int]]:
    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct "
        "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
        "<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return (
        tokenizer.encode(prefix, add_special_tokens=False),
        tokenizer.encode(suffix, add_special_tokens=False),
    )


def process_reranker_inputs(tokenizer, pairs: list[str]):
    prefix_tokens, suffix_tokens = reranker_prompt_tokens(tokenizer)
    content_length = MAX_RERANKER_INPUT_TOKENS - len(prefix_tokens) - len(suffix_tokens)
    if content_length <= 0:
        raise RuntimeError("MAX_RERANKER_INPUT_TOKENS is too small for the reranker prompt.")

    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=content_length,
    )
    for index, input_ids in enumerate(inputs["input_ids"]):
        inputs["input_ids"][index] = prefix_tokens + input_ids + suffix_tokens

    encoded = tokenizer.pad(
        inputs,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
        max_length=MAX_RERANKER_INPUT_TOKENS,
    )
    return encoded


def reranker_yes_no_token_ids(tokenizer) -> tuple[int, int]:
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    if yes_id is None or no_id is None or yes_id < 0 or no_id < 0:
        raise RuntimeError("Could not find reranker tokenizer ids for yes/no labels.")
    return yes_id, no_id


def reranker_probabilities(tokenizer, model, pairs: list[str]) -> list[float]:
    import torch

    encoded = process_reranker_inputs(tokenizer, pairs)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        next_token_logits = model(**encoded).logits[:, -1, :]

    yes_id, no_id = reranker_yes_no_token_ids(tokenizer)
    yes_logits = next_token_logits[:, yes_id]
    no_logits = next_token_logits[:, no_id]
    yes_no_logits = torch.stack([no_logits, yes_logits], dim=1)
    return torch.softmax(yes_no_logits, dim=1)[:, 1].tolist()


def score_chat_row(chat_tokenizer, chat_model, row: dict, batch_size: int) -> dict:
    natural_questions = row.get("natural_questions", [])[:5]
    prompts = [build_prompt(row, question) for question in natural_questions]
    chat_scores = []
    for start in range(0, len(prompts), batch_size):
        chat_scores.extend(
            yes_probabilities(chat_tokenizer, chat_model, prompts[start : start + batch_size])
        )

    average_chat_score = sum(chat_scores) / len(chat_scores) if chat_scores else 0.0
    return {
        "qwen_chat_yes_probability": float(average_chat_score),
        "per_question_chat_yes_probability": [float(score) for score in chat_scores],
    }


def score_reranker_row(reranker_tokenizer, reranker_model, row: dict, batch_size: int) -> dict:
    natural_questions = row.get("natural_questions", [])[:5]
    reranker_pairs = [
        reranker_text(natural_question, row["evidence"])
        for natural_question in natural_questions
    ]
    reranker_scores = []
    for start in range(0, len(reranker_pairs), batch_size):
        reranker_scores.extend(
            reranker_probabilities(
                reranker_tokenizer,
                reranker_model,
                reranker_pairs[start : start + batch_size],
            )
        )

    average_reranker_score = (
        sum(reranker_scores) / len(reranker_scores) if reranker_scores else 0.0
    )
    return {
        "qwen_reranker_probability": float(average_reranker_score),
        "per_question_reranker_probability": [float(score) for score in reranker_scores],
    }


def base_score_row(row: dict) -> dict:
    return {
        "candidate_key": row["candidate_key"],
        "question_index": row["question_index"],
        "candidate_rank": row["candidate_rank"],
        "title": row["title"],
        "answer": row["answer"],
        "normalized_title": normalize_label(row["title"]),
        "normalized_answer": normalize_label(row["answer"]),
    }


def compute_topk_metrics(score_rows: list[dict], score_field: str) -> dict[int, float]:
    grouped: dict[int, list[dict]] = {}
    for row in score_rows:
        grouped.setdefault(int(row["question_index"]), []).append(row)

    if not grouped:
        return {k: 0.0 for k in TOP_K_VALUES}

    correct_counts = {k: 0 for k in TOP_K_VALUES}
    for rows in grouped.values():
        ranked = sorted(
            rows,
            key=lambda row: (
                -float(row[score_field]),
                int(row["candidate_rank"]),
                normalize_label(row["title"]),
            ),
        )
        normalized_answer = normalize_label(ranked[0]["answer"])
        for k in TOP_K_VALUES:
            top_titles = {normalize_label(row["title"]) for row in ranked[:k]}
            if normalized_answer in top_titles:
                correct_counts[k] += 1

    total = len(grouped)
    return {k: correct_counts[k] / total for k in TOP_K_VALUES}


def print_topk_metrics(output_path: Path, chat_model_name: str, reranker_model_name: str) -> None:
    score_rows = load_score_rows(output_path, chat_model_name, reranker_model_name)
    print("Qwen chat verifier ranking accuracy:")
    chat_metrics = compute_topk_metrics(score_rows, "qwen_chat_yes_probability")
    for k in TOP_K_VALUES:
        print(f"  Top-{k}: {chat_metrics[k]:.4f}")
    print("Qwen reranker ranking accuracy:")
    reranker_metrics = compute_topk_metrics(score_rows, "qwen_reranker_probability")
    for k in TOP_K_VALUES:
        print(f"  Top-{k}: {reranker_metrics[k]:.4f}")
    print(f"Evaluated questions: {len({row['question_index'] for row in score_rows})}")


def main() -> None:
    args = parse_args()
    existing_keys = load_existing_keys(
        args.output,
        args.chat_model_name,
        args.reranker_model_name,
    )
    pending_rows = [
        row
        for row in iter_rows(args.input, limit=args.limit)
        if row["candidate_key"] not in existing_keys
    ]
    skipped = len(existing_keys)

    if not pending_rows:
        print(f"No pending rows for {args.chat_model_name} + {args.reranker_model_name}")
        print_topk_metrics(args.output, args.chat_model_name, args.reranker_model_name)
        return

    chat_tokenizer, chat_model, chat_device = load_model(args.chat_model_name, args.device)
    print(f"Loaded chat verifier {args.chat_model_name} on {chat_device}")
    chat_scores_by_key = {}
    for row_number, row in enumerate(pending_rows, start=1):
        chat_scores_by_key[row["candidate_key"]] = score_chat_row(
            chat_tokenizer,
            chat_model,
            row,
            args.batch_size,
        )
        print(
            f"[colab-qwen-chat] scored: {row_number}/{len(pending_rows)} | skipped: {skipped}",
            flush=True,
        )

    del chat_model
    del chat_tokenizer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass

    reranker_tokenizer, reranker_model, reranker_device = load_reranker_model(
        args.reranker_model_name,
        args.device,
    )
    print(f"Loaded reranker {args.reranker_model_name} on {reranker_device}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0

    with args.output.open("a", encoding="utf-8") as output_file:
        for row in pending_rows:
            scored = {
                **base_score_row(row),
                **chat_scores_by_key[row["candidate_key"]],
                **score_reranker_row(
                    reranker_tokenizer,
                    reranker_model,
                    row,
                    args.reranker_batch_size,
                ),
            }
            scored["chat_model_name"] = args.chat_model_name
            scored["reranker_model_name"] = args.reranker_model_name
            output_file.write(json.dumps(scored, ensure_ascii=False) + "\n")
            output_file.flush()
            existing_keys.add(row["candidate_key"])
            processed += 1

            print(
                f"[colab-qwen-verifier] processed: {processed} | skipped: {skipped}",
                flush=True,
            )

    print(f"Wrote scores to {args.output}")
    print_topk_metrics(args.output, args.chat_model_name, args.reranker_model_name)


if __name__ == "__main__":
    main()


# Qwen chat verifier ranking accuracy:
#   Top-1: 0.0400
#   Top-5: 0.2900
#   Top-10: 0.5900
# Qwen reranker ranking accuracy:
#   Top-1: 0.4700
#   Top-5: 0.5800
#   Top-10: 0.5900
# Evaluated questions: 100