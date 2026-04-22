"""Encode DPR passage variants and build four FAISS indexes in Colab."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

try:
    import faiss
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"])
    import faiss


DEFAULT_MODEL_NAME = "facebook/dpr-ctx_encoder-single-nq-base"
BASE_DIR = Path.cwd()
INPUT_PATH = BASE_DIR / "dpr_articles.jsonl.gz"
OUTPUT_DIR = BASE_DIR / "dpr_faiss"
DRIVE_TEMP_DIR = Path("/content/drive/MyDrive/temp")
DRIVE_INPUT_PATH = DRIVE_TEMP_DIR / "dpr_articles.jsonl.gz"
DRIVE_OUTPUT_ARCHIVE_PATH = DRIVE_TEMP_DIR / "dpr_faiss.zip"
MODEL_NAME = DEFAULT_MODEL_NAME
BATCH_SIZE = 64
MAX_LENGTH = 512
SAVE_EMBEDDINGS = False
SKIP_ZIP = False
VARIANT_TO_FIELD = {
    "title_first_sentence": "first_sentence_text",
    "title_first_two_sentences": "first_two_sentences_text",
    "title_first_paragraph": "first_paragraph_text",
    "title_entire_article": "entire_article_text",
}


def iter_jsonl(path: Path):
    if path.suffix == ".gz":
        input_file = gzip.open(path, "rt", encoding="utf-8")
    else:
        input_file = path.open("r", encoding="utf-8")

    with input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def load_metadata(input_path: Path) -> list[dict[str, str | int]]:
    metadata = []
    for row in iter_jsonl(input_path):
        metadata.append(
            {
                "doc_id": row["doc_id"],
                "title": row["title"],
                "source_file": row["source_file"],
                "article_index": row["article_index"],
            }
        )
    return metadata


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode_batch(
    titles: list[str],
    passages: list[str],
    tokenizer: DPRContextEncoderTokenizer,
    model: DPRContextEncoder,
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    encoded_inputs = tokenizer(
        titles,
        passages,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}

    with torch.no_grad():
        outputs = model(**encoded_inputs).pooler_output

    return outputs.detach().cpu().numpy().astype("float32")


def build_variant_index(
    input_path: Path,
    output_dir: Path,
    variant_name: str,
    tokenizer: DPRContextEncoderTokenizer,
    model: DPRContextEncoder,
    device: torch.device,
    batch_size: int,
    max_length: int,
    save_embeddings: bool,
    metadata_rows: list[dict[str, str | int]],
) -> None:
    field_name = VARIANT_TO_FIELD[variant_name]
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    embedding_batches = []
    title_batch = []
    passage_batch = []
    total_rows = 0

    for row in tqdm(iter_jsonl(input_path), desc=f"Encoding {variant_name}"):
        title_batch.append(row["title"])
        passage_batch.append(row[field_name])
        total_rows += 1

        if len(title_batch) >= batch_size:
            embedding_batches.append(
                encode_batch(title_batch, passage_batch, tokenizer, model, device, max_length)
            )
            title_batch = []
            passage_batch = []

    if title_batch:
        embedding_batches.append(
            encode_batch(title_batch, passage_batch, tokenizer, model, device, max_length)
        )

    embeddings = np.concatenate(embedding_batches, axis=0)
    if embeddings.shape[0] != total_rows:
        raise ValueError(
            f"Embedding count mismatch for {variant_name}: "
            f"{embeddings.shape[0]} embeddings for {total_rows} rows."
        )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(variant_dir / "index.faiss"))

    if save_embeddings:
        np.save(variant_dir / "embeddings.npy", embeddings)

    write_jsonl(variant_dir / "metadata.jsonl", metadata_rows)

    with (variant_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "model_name": model.name_or_path,
                "variant_name": variant_name,
                "field_name": field_name,
                "num_vectors": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "faiss_index_type": "IndexFlatIP",
                "max_length": max_length,
            },
            config_file,
            indent=2,
        )


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
    if INPUT_PATH.exists():
        return INPUT_PATH

    try:
        from google.colab import drive  # type: ignore

        if not DRIVE_TEMP_DIR.exists():
            drive.mount("/content/drive")
    except ModuleNotFoundError:
        pass

    if not DRIVE_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found at {INPUT_PATH} or {DRIVE_INPUT_PATH}."
        )

    print(f"Copying input file from Google Drive: {DRIVE_INPUT_PATH}")
    shutil.copy2(DRIVE_INPUT_PATH, INPUT_PATH)
    return INPUT_PATH


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
    local_input_path = ensure_local_input_file()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DPRContextEncoderTokenizer.from_pretrained(MODEL_NAME)
    model = DPRContextEncoder.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    metadata_rows = load_metadata(local_input_path)
    for variant_name in VARIANT_TO_FIELD:
        build_variant_index(
            input_path=local_input_path,
            output_dir=OUTPUT_DIR,
            variant_name=variant_name,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            save_embeddings=SAVE_EMBEDDINGS,
            metadata_rows=metadata_rows,
        )

    if not SKIP_ZIP:
        archive_path = create_output_archive(OUTPUT_DIR)
        print(f"Created archive: {archive_path}")
        ensure_drive_temp_dir()
        shutil.copy2(archive_path, DRIVE_OUTPUT_ARCHIVE_PATH)
        print(f"Copied archive to Google Drive: {DRIVE_OUTPUT_ARCHIVE_PATH}")


main()
