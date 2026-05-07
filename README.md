# Three-Stage Wikipedia Clue Answering

This repository contains a CSC483/583 course project for answering clue-style questions by retrieving Wikipedia page titles. The final system uses three stages: weighted hybrid retrieval, cross-encoder reranking, and Qwen3-Reranker-4B reranking over the top candidates.

## Installation

Create a conda environment and install the Python dependencies:

```bash
conda create -n wiki-clue-ir python=3.11
conda activate wiki-clue-ir
pip install -r requirements.txt
```

The final run expects the local processed data and indexes from the project run to be present under `data/processed/` and `index/`. Large SQLite databases, FAISS indexes, Whoosh indexes, and caches are ignored by Git.

## Code Files

Core evaluation and retrieval:

- `src/run_test_100.py`: runs the 100-question evaluation and Stage 2 cross-encoder reranking.
- `src/search.py`: implements sparse Whoosh retrieval, DPR/FAISS retrieval, weighted hybrid scoring, redirect lookup, and pattern bonuses.
- `src/retrieval_cache.py`: stores reusable retrieval and model-scoring results in SQLite.
- `src/run_test_specific.py`: runs the same evaluation logic for one selected clue for debugging.
- `src/temp_search_query.py`: small manual search helper for checking query results.

Parsing, cleaning, and sparse indexing:

- `src/processor1_parse.py`: parses raw Wikipedia files and the question file into structured data.
- `src/processor2_clean.py`: cleans Wikipedia article text by removing markup, references, categories, and redirect lines.
- `src/processor3_tokenize.py`: creates the custom-tokenized article database used by the older sparse baseline.
- `src/schema.py`: defines Whoosh schemas for tokenized, default-analyzer, title/body, and category-aware indexes.
- `src/build_index.py`: wrapper for building the older tokenized Whoosh index.
- `src/processor4_index.py`: builds the tokenized body-only Whoosh index.
- `src/processor4_whoosh_index.py`: builds the cleaned-body Whoosh index with Whoosh's default analyzer.
- `src/processor4_whoosh_title_body_index.py`: builds the title-plus-body Whoosh index.
- `src/processor4_whoosh_title_body_category_index.py`: builds the title/body/category Whoosh index.
- `src/processor4_cole_index.py`: builds the final sparse body index used by the positive-weight Stage 1 component.
- `src/processor4_lead_index_common.py`: shared helpers for first-sentence, first-two-sentence, and first-paragraph indexes.
- `src/processor4_cole_first_sentence_index.py`: builds the first-sentence sparse index.
- `src/processor4_cole_first_two_sentences_index.py`: builds the first-two-sentences sparse index.
- `src/processor4_cole_first_paragraph_index.py`: builds the opening-paragraph sparse index.
- `src/processor4_cole_redirect_index.py`: builds the redirect-title sparse index.
- `src/processor_redirect.py`: resolves redirect pages to canonical article titles.
- `src/processor_category_semantic.py`: creates the category-semantic support database used in category-aware experiments.

Dense retrieval and generated questions:

- `src/prepare_dpr_articles.py`: exports cleaned non-redirect articles into DPR-ready text variants.
- `src/colab_build_dpr_faiss.py`: builds FAISS indexes from DPR article embeddings in Colab.
- `src/processor_question_dpr_embeddings.py`: precomputes DPR question embeddings for category-plus-clue and clue-only queries.
- `src/processor_natural_question_dpr_embeddings.py`: precomputes DPR embeddings for generated natural questions.
- `src/colab_generate_natural_questions_qwen.py`: generates the natural-question JSON file used by Stage 1 and Stage 2.
- `src/generate_qwen_related_questions.py`: older helper-question generation experiment.
- `src/colab_build_qwen_summary_dpr_faiss.py`: builds the body-summary dense-index experiment.

Cross-encoder and Qwen reranking:

- `src/export_cross_encoder_pairs.py`: exports cross-encoder query/document pairs for offline scoring.
- `src/colab_score_cross_encoder_pairs.py`: scores exported cross-encoder pairs in Colab and writes a cache.
- `src/export_qwen_verifier_inputs.py`: exports the top 10 Stage 2 candidates for Qwen scoring.
- `src/colab_score_qwen_verifier_inputs.py`: scores exported candidates with Qwen3-4B and Qwen3-Reranker-4B.

Inspection helpers:

- `src/temp_compare_query_processing.py`: compares tokenized and Whoosh query processing.
- `src/temp_inspect_paraphrases.py`: inspects old paraphrase-based query variants.
- `src/temp_inspect_parrot_paraphrases.py`: inspects old Parrot paraphrase outputs.
- `src/temp_inspect_whoosh_query.py`: prints Whoosh analyzer and parser output for a query.
- `src/temp_show_top_articles.py`: prints a few stored Wikipedia articles from SQLite.
- `src/__init__.py`: marks `src` as a Python package.

## How to Run

Run Stage 1 and Stage 2 from the project root:

```bash
python -m src.run_test_100 --weighted-cole --include-category \
  --cross-encoder --cross-encoder-mode chunks_100 \
  --cross-encoder-query-mode natural_questions_avg \
  --cross-encoder-natural-questions \
  data/processed/questions_natural_qwen3_8b.json
```

Export the top 10 Stage 2 candidates for Stage 3:

```bash
python -m src.export_qwen_verifier_inputs \
  --output data/processed/qwen_verifier_inputs.jsonl --top-docs 10
```

Score the exported file in Google Colab:

```bash
python colab_score_qwen_verifier_inputs.py \
  --input qwen_verifier_inputs.jsonl \
  --output qwen_verifier_scores.jsonl \
  --chat-model-name Qwen/Qwen3-4B \
  --reranker-model-name Qwen/Qwen3-Reranker-4B \
  --device cuda
```

For a quick check, confirm that the Stage 1/2 command prints Top-k metrics and that the Qwen scoring script prints the Stage 3 Top-k metrics after scoring.
