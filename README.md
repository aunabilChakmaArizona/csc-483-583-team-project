# CSC483/583 Programming Project: Three-Stage Wikipedia Clue Answering

This repository contains a CSC483/583 course project for answering clue-style questions by retrieving Wikipedia page titles. The final system uses three stages: weighted hybrid retrieval, cross-encoder reranking, and Qwen3-Reranker-4B reranking over the top candidates.

Team Members: Cole Mobberly, Suhani Surana, and Aunabil Chakma.

## Installation

Create a conda environment and install the Python dependencies:

```bash
conda create -n wiki-clue-ir python=3.11
conda activate wiki-clue-ir
pip install -r requirements.txt
```

The final run expects the local processed data and indexes from the project run to be present under `data/processed/` and `index/`. Large SQLite databases, FAISS indexes, Whoosh indexes, and caches are ignored by Git.
<!-- 
## Code Files

Core evaluation and retrieval:

- `src/run_test_100.py`: runs the 100-question evaluation and Stage 2 cross-encoder reranking.
  Main methods: `search_questions`, `build_stage2_searches`, `evaluate_questions`, `compute_metrics_from_searches`.
- `src/search.py`: implements sparse Whoosh retrieval, DPR/FAISS retrieval, weighted hybrid scoring, redirect lookup, and pattern bonuses.
  Main methods: `multi_search_whoosh_weighted_cole`, `search_whoosh_weighted_cole`, `search_dpr_faiss`, `build_faiss_precomputed_components`.
- `src/retrieval_cache.py`: stores reusable retrieval and model-scoring results in SQLite.
  Main methods: `build_component_cache_key`, `load_cached_component_results`, `store_component_results`.
- `src/run_test_specific.py`: runs the same evaluation logic for one selected clue for debugging.
  Main methods: `select_question_by_clue`, `average_ranked_results`, `main`.
- `src/temp_search_query.py`: small manual search helper for checking query results.
  Main methods: `run_search`, `main`.

Parsing, cleaning, and sparse indexing:

- `src/processor1_parse.py`: parses raw Wikipedia files and the question file into structured data.
  Main methods: `parse_articles_from_text`, `materialize_articles_to_sqlite`, `parse_questions_file`.
- `src/processor2_clean.py`: cleans Wikipedia article text by removing markup, references, categories, and redirect lines.
  Main methods: `clean_body`, `materialize_cleaned_articles`.
- `src/processor3_tokenize.py`: creates the custom-tokenized article database used by the older sparse baseline.
  Main methods: `tokenize_body`, `materialize_tokenized_articles`.
- `src/schema.py`: defines Whoosh schemas for tokenized, default-analyzer, title/body, and category-aware indexes.
  Main methods: `get_schema`, `get_whoosh_default_schema`, `get_whoosh_title_body_schema`, `get_whoosh_title_body_category_schema`.
- `src/build_index.py`: wrapper for building the older tokenized Whoosh index.
  Main methods: `build_index`.
- `src/processor4_index.py`: builds the tokenized body-only Whoosh index.
  Main methods: `materialize_whoosh_index`, `open_index`.
- `src/processor4_whoosh_index.py`: builds the cleaned-body Whoosh index with Whoosh's default analyzer.
  Main methods: `materialize_whoosh_default_index`, `open_index`.
- `src/processor4_whoosh_title_body_index.py`: builds the title-plus-body Whoosh index.
  Main methods: `materialize_whoosh_title_body_index`, `open_index`.
- `src/processor4_whoosh_title_body_category_index.py`: builds the title/body/category Whoosh index.
  Main methods: `load_semantic_category_lookup`, `materialize_whoosh_title_body_category_index`, `open_index`.
- `src/processor4_cole_index.py`: builds the final sparse body index used by the positive-weight Stage 1 component.
  Main methods: `materialize_whoosh_cole_index`, `open_index`.
- `src/processor4_lead_index_common.py`: shared helpers for first-sentence, first-two-sentence, and first-paragraph indexes.
  Main methods: `extract_lead_text`, `materialize_lead_index`, `search_index`.
- `src/processor4_cole_first_sentence_index.py`: builds the first-sentence sparse index.
  Main methods: `materialize_whoosh_cole_first_sentence_index`, `open_index`.
- `src/processor4_cole_first_two_sentences_index.py`: builds the first-two-sentences sparse index.
  Main methods: `materialize_whoosh_cole_first_two_sentences_index`, `open_index`.
- `src/processor4_cole_first_paragraph_index.py`: builds the opening-paragraph sparse index.
  Main methods: `materialize_whoosh_cole_first_paragraph_index`, `open_index`.
- `src/processor4_cole_redirect_index.py`: builds the redirect-title sparse index.
  Main methods: `materialize_whoosh_cole_redirect_index`, `open_index`.
- `src/processor_redirect.py`: resolves redirect pages to canonical article titles.
  Main methods: `extract_redirect_target`, `resolve_redirect`, `materialize_redirect_mappings`.
- `src/processor_category_semantic.py`: creates the category-semantic support database used in category-aware experiments.
  Main methods: `top_category_matches`, `materialize_category_semantic_matches`.

Dense retrieval and generated questions:

- `src/prepare_dpr_articles.py`: exports cleaned non-redirect articles into DPR-ready text variants.
  Main methods: `prepare_article_record`, `export_dpr_articles`.
- `src/colab_build_dpr_faiss.py`: builds FAISS indexes from DPR article embeddings in Colab.
  Main methods: `build_variant_index`, `create_output_archive`.
- `src/processor_question_dpr_embeddings.py`: precomputes DPR question embeddings for category-plus-clue and clue-only queries.
  Main methods: `category_plus_clue_text`, `encode_question_texts`, `materialize_question_dpr_embeddings`.
- `src/processor_natural_question_dpr_embeddings.py`: precomputes DPR embeddings for generated natural questions.
  Main methods: `combined_natural_questions_text`, `materialize_natural_question_dpr_embeddings`.
- `src/colab_generate_natural_questions_qwen.py`: generates the natural-question JSON file used by Stage 1 and Stage 2.
  Main methods: `build_messages`, `generate_questions_for_item`, `write_results`.
- `src/generate_qwen_related_questions.py`: older helper-question generation experiment.
  Main methods: `build_prompt`, `extract_helper_questions`, `generate_related_questions_for_dataset`.
- `src/colab_build_qwen_summary_dpr_faiss.py`: builds the body-summary dense-index experiment.
  Main methods: `summarize_articles`, `build_summary_faiss_index`, `main`.

Cross-encoder and Qwen reranking:

- `src/export_cross_encoder_pairs.py`: exports cross-encoder query/document pairs for offline scoring.
  Main methods: `document_chunks`, `main`.
- `src/colab_score_cross_encoder_pairs.py`: scores exported cross-encoder pairs in Colab and writes a cache.
  Main methods: `score_rows`, `build_score_records`, `insert_score_records`.
- `src/export_qwen_verifier_inputs.py`: exports the top 10 Stage 2 candidates for Qwen scoring.
  Main methods: `candidate_key`, `first_words`, `main`.
- `src/colab_score_qwen_verifier_inputs.py`: scores exported candidates with Qwen3-4B and Qwen3-Reranker-4B.
  Main methods: `score_chat_row`, `score_reranker_row`, `compute_topk_metrics`, `print_topk_metrics`.

Inspection helpers:

- `src/temp_compare_query_processing.py`: compares tokenized and Whoosh query processing.
  Main methods: `whoosh_query_terms`, `main`.
- `src/temp_inspect_paraphrases.py`: inspects old paraphrase-based query variants.
  Main methods: `build_raw_variants`, `print_question_variants`.
- `src/temp_inspect_parrot_paraphrases.py`: inspects old Parrot paraphrase outputs.
  Main methods: `paraphrase_text`, `inspect_question`.
- `src/temp_inspect_whoosh_query.py`: prints Whoosh analyzer and parser output for a query.
  Main methods: `main`.
- `src/temp_show_top_articles.py`: prints a few stored Wikipedia articles from SQLite.
  Main methods: `fetch_top_articles`, `print_articles`.
- `src/__init__.py`: marks `src` as a Python package. -->

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
