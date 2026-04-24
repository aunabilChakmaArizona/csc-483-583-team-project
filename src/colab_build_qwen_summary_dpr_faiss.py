"""Summarize wiki passages with Qwen, then build a DPR FAISS index in Colab."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


BASE_DIR = Path.cwd()
DRIVE_TEMP_DIR = Path("/content/drive/MyDrive/temp")
LOCAL_INPUT_PATH = BASE_DIR / "dpr_articles.jsonl.gz"
DRIVE_INPUT_PATH = DRIVE_TEMP_DIR / "dpr_articles.jsonl.gz"
SUMMARY_OUTPUT_PATH = BASE_DIR / "dpr_articles_qwen_summaries.jsonl"
DRIVE_SUMMARY_OUTPUT_PATH = DRIVE_TEMP_DIR / "dpr_articles_qwen_summaries.jsonl"
OUTPUT_DIR = BASE_DIR / "dpr_faiss_qwen_summary"
DRIVE_OUTPUT_DIR = DRIVE_TEMP_DIR / "dpr_faiss_qwen_summary"
DRIVE_OUTPUT_ARCHIVE_PATH = DRIVE_TEMP_DIR / "dpr_faiss_qwen_summary.zip"

QWEN_MODEL_NAME = "Qwen/Qwen3-1.7B"
DPR_MODEL_NAME = "facebook/dpr-ctx_encoder-single-nq-base"
SUMMARY_VARIANT_NAME = "title_qwen_summary"
SUMMARY_SOURCE_FIELD = "entire_article_text"
SUMMARY_MIN_SOURCE_CHARS = 1200

ARTICLES_TO_PROCESS = None
SUMMARY_BATCH_SIZE = 32
SUMMARY_INPUT_MAX_CHARS = 6000
SUMMARY_MAX_NEW_TOKENS = 500
SUMMARY_MAX_RETRIES = 3
SUMMARY_SAVE_EVERY = 25
SUMMARY_PROGRESS_EVERY_BATCHES = 4
SUMMARY_DO_SAMPLE = True
TORCH_DTYPE_NAME = "bfloat16"

DPR_BATCH_SIZE = 64
DPR_MAX_LENGTH = 512
SAVE_EMBEDDINGS = False
SKIP_ZIP = False
COPY_SUMMARIES_TO_DRIVE = True

INLINE_WHITESPACE_RE = re.compile(r"\s+")
JSON_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def ensure_dependencies() -> None:
    packages = [
        "transformers>=4.51.0",
        "accelerate>=0.30.0",
        "sentencepiece",
        "faiss-cpu",
        "tqdm",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def iter_jsonl(path: Path):
    import gzip

    if path.suffix == ".gz":
        input_file = gzip.open(path, "rt", encoding="utf-8")
    else:
        input_file = path.open("r", encoding="utf-8")

    with input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_summary_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def clean_summary_text(text: str) -> str:
    text = JSON_CODE_FENCE_RE.sub("", text.strip())
    text = INLINE_WHITESPACE_RE.sub(" ", text).strip()
    return text


def extract_json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped]

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    return candidates


def parse_summary_payload(text: str) -> str:
    for candidate in extract_json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict) and "summary" in parsed:
            cleaned = clean_summary_text(str(parsed["summary"]))
            if cleaned:
                return cleaned

        if isinstance(parsed, str):
            cleaned = clean_summary_text(parsed)
            if cleaned:
                return cleaned

    cleaned = clean_summary_text(text)
    if cleaned:
        return cleaned

    raise ValueError(f"Could not parse a summary from model output: {text}")


def truncate_source_text(text: str, max_chars: int) -> str:
    cleaned = clean_summary_text(text)
    if len(cleaned) <= max_chars:
        return cleaned

    truncated = cleaned[:max_chars]
    if " " not in truncated:
        return truncated
    return truncated.rsplit(" ", 1)[0]


def build_summary_messages(row: dict[str, object]) -> list[dict[str, str]]:
    article_text = truncate_source_text(str(row.get(SUMMARY_SOURCE_FIELD, "")), SUMMARY_INPUT_MAX_CHARS)
    return [
        {
            "role": "system",
            "content": (
                "You summarize Wikipedia-style articles for retrieval. "
                "Preserve unique facts, names, dates, places, aliases, and distinguishing details. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write one concise factual summary for dense retrieval.\n"
                "Use 3 to 5 sentences.\n"
                "Keep the most identifying facts.\n"
                "Do not invent facts.\n"
                "Do not use bullet points.\n\n"
                f"Title: {row['title']}\n"
                f"Article text: {article_text}\n\n"
                'Return exactly this JSON shape and nothing else:\n{"summary":"..."}'
            ),
        },
    ]


def build_summary_prompt_text(tokenizer, row: dict[str, object]) -> str:
    return tokenizer.apply_chat_template(
        build_summary_messages(row),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def load_qwen_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        torch_dtype = getattr(torch, TORCH_DTYPE_NAME)
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model


def generate_summaries_for_batch(
    tokenizer,
    model,
    rows: list[dict[str, object]],
) -> list[tuple[str, str]]:
    prompt_texts = [build_summary_prompt_text(tokenizer, row) for row in rows]
    model_inputs = tokenizer(prompt_texts, padding=True, return_tensors="pt").to(model.device)
    input_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=SUMMARY_MAX_NEW_TOKENS,
        do_sample=SUMMARY_DO_SAMPLE,
    )

    results = []
    for batch_index, input_length in enumerate(input_lengths):
        output_ids = generated_ids[batch_index][input_length:]
        raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        results.append((parse_summary_payload(raw_output), raw_output))

    return results


def summarize_batch_with_retries(
    tokenizer,
    model,
    rows: list[dict[str, object]],
) -> list[tuple[str, str]]:
    last_error = None

    for _attempt in range(1, SUMMARY_MAX_RETRIES + 1):
        try:
            return generate_summaries_for_batch(tokenizer, model, rows)
        except Exception as error:
            last_error = error

    if len(rows) == 1:
        raise RuntimeError(
            f"Failed to summarize article {rows[0]['doc_id']} after {SUMMARY_MAX_RETRIES} attempts: "
            f"{last_error}"
        ) from last_error

    midpoint = len(rows) // 2
    left_results = summarize_batch_with_retries(tokenizer, model, rows[:midpoint])
    right_results = summarize_batch_with_retries(tokenizer, model, rows[midpoint:])
    return left_results + right_results


def used_source_text_chars(row: dict[str, object]) -> int:
    return len(
        truncate_source_text(
            str(row.get(SUMMARY_SOURCE_FIELD, "")),
            SUMMARY_INPUT_MAX_CHARS,
        )
    )


def source_text_chars(row: dict[str, object]) -> int:
    return len(str(row.get(SUMMARY_SOURCE_FIELD, "")))


def should_summarize_row(row: dict[str, object]) -> bool:
    return source_text_chars(row) > SUMMARY_MIN_SOURCE_CHARS


def build_summary_row(
    row: dict[str, object],
    summary_text: str,
    raw_output: str,
    summary_model_name: str = QWEN_MODEL_NAME,
) -> dict:
    return {
        "doc_id": row["doc_id"],
        "title": row["title"],
        "source_file": row["source_file"],
        "article_index": row["article_index"],
        "summary_text": summary_text,
        "source_field": SUMMARY_SOURCE_FIELD,
        "source_text_chars": source_text_chars(row),
        "used_text_chars": used_source_text_chars(row),
        "summary_model_name": summary_model_name,
        "raw_output_preview": raw_output[:500],
    }


def format_duration(seconds: float) -> str:
    rounded_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def load_remaining_rows_sorted_by_size(
    input_path: Path,
    completed_doc_ids: set[str],
) -> list[dict[str, object]]:
    remaining_rows = []
    limit = ARTICLES_TO_PROCESS

    for row in iter_jsonl(input_path):
        doc_id = row["doc_id"]
        if doc_id in completed_doc_ids:
            continue
        row["_used_text_chars"] = used_source_text_chars(row)
        remaining_rows.append(row)
        if limit is not None and len(completed_doc_ids) + len(remaining_rows) >= limit:
            break

    remaining_rows.sort(
        key=lambda row: (
            -int(row["_used_text_chars"]),
            str(row["source_file"]),
            int(row["article_index"]),
        )
    )
    return remaining_rows


def summarize_articles(input_path: Path, summary_output_path: Path) -> list[dict]:
    existing_rows = load_existing_summary_rows(summary_output_path)
    completed_doc_ids = {row["doc_id"] for row in existing_rows}
    rows = list(existing_rows)

    if ARTICLES_TO_PROCESS is not None and len(rows) >= ARTICLES_TO_PROCESS:
        print(f"Summary output already contains {len(rows)} rows: {summary_output_path}")
        return rows[:ARTICLES_TO_PROCESS]

    remaining_rows = load_remaining_rows_sorted_by_size(input_path, completed_doc_ids)
    rows_for_qwen = [row for row in remaining_rows if should_summarize_row(row)]
    passthrough_rows = len(remaining_rows) - len(rows_for_qwen)
    articles_to_summarize = len(rows_for_qwen)
    total_batches = math.ceil(articles_to_summarize / SUMMARY_BATCH_SIZE) if articles_to_summarize else 0
    print(
        f"[qwen_summary] Remaining articles: {len(remaining_rows)} | "
        f"Qwen batches for >{SUMMARY_MIN_SOURCE_CHARS} chars: {articles_to_summarize} | "
        f"Pass-through: {passthrough_rows} | "
        f"Approx batches: {total_batches}"
    )

    tokenizer = None
    model = None
    if articles_to_summarize:
        print(f"Loading Qwen summarizer: {QWEN_MODEL_NAME}")
        tokenizer, model = load_qwen_model_and_tokenizer()

    started_at = time.time()
    processed_count = len(rows)
    processed_batches = 0
    pending_rows = []

    for row in remaining_rows:
        if not should_summarize_row(row):
            rows.append(
                build_summary_row(
                    row,
                    str(row.get(SUMMARY_SOURCE_FIELD, "")),
                    "",
                    summary_model_name="passthrough_original_body",
                )
            )
            completed_doc_ids.add(row["doc_id"])
            processed_count += 1

            if processed_count % SUMMARY_SAVE_EVERY == 0:
                write_jsonl(summary_output_path, rows)
            continue

        pending_rows.append(row)

        if len(pending_rows) < SUMMARY_BATCH_SIZE:
            continue

        batch_started_at = time.time()
        batch_results = summarize_batch_with_retries(tokenizer, model, pending_rows)

        for pending_row, (summary_text, raw_output) in zip(pending_rows, batch_results):
            rows.append(build_summary_row(pending_row, summary_text, raw_output))
            completed_doc_ids.add(pending_row["doc_id"])
            processed_count += 1
        processed_batches += 1

        if processed_count % SUMMARY_SAVE_EVERY == 0:
            write_jsonl(summary_output_path, rows)

        if processed_batches % SUMMARY_PROGRESS_EVERY_BATCHES == 0:
            elapsed = time.time() - started_at
            batch_elapsed = time.time() - batch_started_at
            average_batch_seconds = elapsed / processed_batches
            remaining_batches = max(0, total_batches - processed_batches)
            eta_seconds = remaining_batches * average_batch_seconds
            print(
                f"[qwen_summary] Batches: {processed_batches}/{total_batches} | "
                f"Summarized {processed_count} articles | "
                f"Last batch size: {len(pending_rows)} | "
                f"Last batch: {batch_elapsed:.2f}s | "
                f"Elapsed: {format_duration(elapsed)} | "
                f"ETA: {format_duration(eta_seconds)}"
            )

        pending_rows = []

    if pending_rows:
        batch_started_at = time.time()
        batch_results = summarize_batch_with_retries(tokenizer, model, pending_rows)

        for pending_row, (summary_text, raw_output) in zip(pending_rows, batch_results):
            rows.append(build_summary_row(pending_row, summary_text, raw_output))
            completed_doc_ids.add(pending_row["doc_id"])
            processed_count += 1
        processed_batches += 1

        elapsed = time.time() - started_at
        batch_elapsed = time.time() - batch_started_at
        average_batch_seconds = elapsed / processed_batches
        remaining_batches = max(0, total_batches - processed_batches)
        eta_seconds = remaining_batches * average_batch_seconds
        print(
            f"[qwen_summary] Batches: {processed_batches}/{total_batches} | "
            f"Summarized {processed_count} articles | "
            f"Last batch size: {len(pending_rows)} | "
            f"Last batch: {batch_elapsed:.2f}s | "
            f"Elapsed: {format_duration(elapsed)} | "
            f"ETA: {format_duration(eta_seconds)}"
        )

    write_jsonl(summary_output_path, rows)
    elapsed = time.time() - started_at
    print(f"[qwen_summary] Done | Summaries: {len(rows)} | Elapsed: {elapsed:.2f}s")
    return rows


def load_dpr_model_and_tokenizer():
    import torch
    from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

    tokenizer = DPRContextEncoderTokenizer.from_pretrained(DPR_MODEL_NAME)
    model = DPRContextEncoder.from_pretrained(DPR_MODEL_NAME, use_safetensors=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def encode_summary_batch(
    titles: list[str],
    summaries: list[str],
    tokenizer,
    model,
    device,
    max_length: int,
):
    encoded_inputs = tokenizer(
        titles,
        summaries,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}

    import torch

    with torch.no_grad():
        outputs = model(**encoded_inputs).pooler_output

    return outputs.detach().cpu().numpy().astype("float32")


def build_summary_faiss_index(
    summary_rows: list[dict],
    output_dir: Path,
    save_embeddings: bool,
) -> Path:
    import numpy as np
    from tqdm.auto import tqdm

    try:
        import faiss
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"])
        import faiss

    tokenizer, model, device = load_dpr_model_and_tokenizer()

    variant_dir = output_dir / SUMMARY_VARIANT_NAME
    variant_dir.mkdir(parents=True, exist_ok=True)

    embedding_batches = []
    title_batch = []
    summary_batch = []

    for row in tqdm(summary_rows, desc="Encoding Qwen summaries with DPR"):
        title_batch.append(row["title"])
        summary_batch.append(row["summary_text"])

        if len(title_batch) >= DPR_BATCH_SIZE:
            embedding_batches.append(
                encode_summary_batch(
                    title_batch,
                    summary_batch,
                    tokenizer,
                    model,
                    device,
                    DPR_MAX_LENGTH,
                )
            )
            title_batch = []
            summary_batch = []

    if title_batch:
        embedding_batches.append(
            encode_summary_batch(
                title_batch,
                summary_batch,
                tokenizer,
                model,
                device,
                DPR_MAX_LENGTH,
            )
        )

    embeddings = np.concatenate(embedding_batches, axis=0)
    if embeddings.shape[0] != len(summary_rows):
        raise ValueError(
            f"Embedding count mismatch: {embeddings.shape[0]} embeddings for "
            f"{len(summary_rows)} summary rows."
        )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(variant_dir / "index.faiss"))

    if save_embeddings:
        np.save(variant_dir / "embeddings.npy", embeddings)

    metadata_rows = [
        {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "source_file": row["source_file"],
            "article_index": row["article_index"],
            "summary_text": row["summary_text"],
        }
        for row in summary_rows
    ]
    write_jsonl(variant_dir / "metadata.jsonl", metadata_rows)

    with (variant_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "summary_model_name": QWEN_MODEL_NAME,
                "dpr_model_name": DPR_MODEL_NAME,
                "variant_name": SUMMARY_VARIANT_NAME,
                "source_field": SUMMARY_SOURCE_FIELD,
                "num_vectors": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "faiss_index_type": "IndexFlatIP",
                "summary_input_max_chars": SUMMARY_INPUT_MAX_CHARS,
                "dpr_max_length": DPR_MAX_LENGTH,
            },
            config_file,
            indent=2,
        )

    return variant_dir


def create_output_archive(output_dir: Path) -> Path:
    archive_base = output_dir.parent / output_dir.name
    archive_path = shutil.make_archive(
        base_name=str(archive_base),
        format="zip",
        root_dir=str(output_dir.parent),
        base_dir=output_dir.name,
    )
    return Path(archive_path)


def ensure_local_input_file() -> Path:
    if LOCAL_INPUT_PATH.exists():
        return LOCAL_INPUT_PATH

    try:
        from google.colab import drive  # type: ignore

        if not DRIVE_TEMP_DIR.exists():
            drive.mount("/content/drive")
    except ModuleNotFoundError:
        pass

    if not DRIVE_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found at {LOCAL_INPUT_PATH} or {DRIVE_INPUT_PATH}."
        )

    print(f"Copying input file from Google Drive: {DRIVE_INPUT_PATH}")
    shutil.copy2(DRIVE_INPUT_PATH, LOCAL_INPUT_PATH)
    return LOCAL_INPUT_PATH


def ensure_drive_temp_dir() -> Path:
    try:
        from google.colab import drive  # type: ignore

        if not DRIVE_TEMP_DIR.exists():
            drive.mount("/content/drive")
    except ModuleNotFoundError:
        raise RuntimeError("Google Drive copy is only supported when running in Colab.")

    DRIVE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return DRIVE_TEMP_DIR


def main() -> None:
    ensure_dependencies()
    local_input_path = ensure_local_input_file()

    summary_rows = summarize_articles(local_input_path, SUMMARY_OUTPUT_PATH)
    if not summary_rows:
        raise RuntimeError("No summaries were generated; cannot build the FAISS index.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variant_dir = build_summary_faiss_index(
        summary_rows=summary_rows,
        output_dir=OUTPUT_DIR,
        save_embeddings=SAVE_EMBEDDINGS,
    )
    print(f"Built DPR FAISS variant: {variant_dir}")

    if COPY_SUMMARIES_TO_DRIVE:
        ensure_drive_temp_dir()
        shutil.copy2(SUMMARY_OUTPUT_PATH, DRIVE_SUMMARY_OUTPUT_PATH)
        print(f"Copied summaries to Google Drive: {DRIVE_SUMMARY_OUTPUT_PATH}")

    ensure_drive_temp_dir()
    shutil.copytree(OUTPUT_DIR, DRIVE_OUTPUT_DIR, dirs_exist_ok=True)
    print(f"Copied index directory to Google Drive: {DRIVE_OUTPUT_DIR}")

    if not SKIP_ZIP:
        archive_path = create_output_archive(OUTPUT_DIR)
        print(f"Created archive: {archive_path}")
        ensure_drive_temp_dir()
        shutil.copy2(archive_path, DRIVE_OUTPUT_ARCHIVE_PATH)
        print(f"Copied archive to Google Drive: {DRIVE_OUTPUT_ARCHIVE_PATH}")


if __name__ == "__main__":
    main()
