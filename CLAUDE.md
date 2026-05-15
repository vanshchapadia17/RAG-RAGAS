# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`RAG-RAGAS` — a Retrieval-Augmented Generation system with a full evaluation
pipeline (RAGAS + retrieval metrics), MLflow experiment tracking, a CI/CD quality
gate, and a 4-page Flask dashboard. The codebase is implemented and runnable, not
a scaffold.

The LLM runs on **Groq** (`langchain-groq`) and embeddings + reranking run on the
**HuggingFace Inference API** (`HuggingFaceEndpointEmbeddings`, `HuggingFaceCrossEncoder`).
The default pipeline therefore needs no local GPU or large model downloads — the
only exception is `semantic_chunking`, which still downloads a local embedding model.

## Environment & commands

The project is installed as an editable package (`-e .` in `requirements.txt`), so
`src` is importable as a top-level package. The `scripts/` files also do
`sys.path.insert(0, <project root>)`, so they run without the editable install too.

```bash
# create/activate venv (Windows bash)
python -m venv venv
source venv/Scripts/activate

# install deps + editable package
pip install -r requirements.txt

# build the FAISS index from PDFs in data/ (default strategy: recursive)
python scripts/index_documents.py --strategy recursive   # recursive | fixed | semantic

# run the Flask dashboard (entry point)
python run.py                       # serves http://localhost:5000

# run the 4 structured experiments and log them to MLflow
python scripts/run_experiments.py

# inspect experiment runs
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# CI/CD quality gate — exit 0 = deploy, exit 1 = block
python scripts/quality_gate.py
```

Required environment variables (in `.env`, loaded by `python-dotenv` in `src/config.py`):
`GROQ_API_KEY`, `HF_TOKEN`, and optionally `FLASK_SECRET_KEY`. There is no `.env.example`.

No test runner, linter, or formatter is configured. `.github/workflows/` contains
only `.gitkeep` — no CI is wired up despite the quality-gate script existing.

## Architecture

Data flows: **PDFs → chunks → FAISS index → retriever → RAG chain → evaluation → MLflow**.

- **`src/`** — core RAG logic, flat module layout (no sub-packages):
  - `config.py` — single source of truth for models, chunking params, retrieval
    params, all filesystem paths, and `QUALITY_GATES` thresholds. Import settings
    from here rather than hardcoding.
  - `document_processor.py` — PDF loading (`PyMuPDFLoader`) and chunking strategies
    (`fixed_size`, `recursive`, `child`, `semantic`).
  - `vectorstore.py` — FAISS build / load / stats / `add_documents`.
  - `retriever.py` — 8 retrieval strategies behind `get_retriever(strategy, vectorstore, all_chunks=, all_docs=)`.
  - `rag_pipeline.py` — the prompt, `build_rag_chain()` (LCEL chain), and
    `query_with_sources()` (answer + source chunks for the dashboard).
  - `evaluator.py` — RAGAS evaluation (faithfulness, answer relevancy, context
    precision, context recall, answer correctness); the judge LLM is Groq.
  - `metrics.py` — keyword-based retrieval metrics: MRR, NDCG, Precision@K, Recall@K.
  - `experiment_tracker.py` — MLflow logging and `check_quality_gates()`.
  - `logger.py`, `exception.py` — shared logging + error-wrapping helpers.
- **`app/`** — Flask dashboard, application-factory pattern:
  - `__init__.py` — `create_app()` registers the four blueprints.
  - `routes/` — `chat.py`, `documents.py`, `eval.py`, `experiments.py` (each a
    `Blueprint` serving an HTML page + JSON API routes).
  - `templates/`, `static/` — Jinja2 HTML and CSS/JS.
- **`scripts/`** — CLI entry points: `index_documents.py`, `run_experiments.py`,
  `quality_gate.py`.
- **`run.py`** — Flask entry point; calls `create_app()`.

### Conventions

- **Config-driven:** add new tunables to `src/config.py` and import them; do not
  hardcode model names, paths, or thresholds in modules.
- **Logging/exceptions:** `src/logger.py` (import `from src.logger import logging`,
  use the module-level object directly — not `getLogger(__name__)`) and
  `src/exception.py` (`raise CustomException(e, sys)` — `sys` **must** be the second
  argument) exist as project infrastructure. Current `src/` and `app/` code uses
  bare `print()` instead. When editing an existing file, match that file's style;
  when adding substantial new modules, prefer the logger/exception helpers.
- **MLflow store:** tracking URI is `sqlite:///mlruns/mlflow.db` (relative path) —
  run MLflow commands from the project root or a new DB is created elsewhere.
- **Generated artifacts** (`faiss_db/`, `mlruns/`, `eval/ragas_scores.json`,
  `outputs/`) are currently committed to git. Avoid worsening this; ideally they
  would be gitignored.

## Known issues / gotchas

These are real inconsistencies — be aware of them, and fix only when asked:

- **`EMBEDDING_MODEL` is not wired up.** `config.py` sets it to `BAAI/bge-large-en-v1.5`,
  but `vectorstore.py::get_embeddings()` and `evaluator.py::get_judge_embeddings()`
  both hardcode `sentence-transformers/all-MiniLM-L6-v2`. The index runs on MiniLM.
- **Dashboard eval only supports some retrievers.** `app/routes/eval.py` calls
  `get_retriever(strategy, vs)` without `all_chunks`/`all_docs`, so `hybrid`,
  `bm25`, `reranker_hybrid`, and `parent` will fail from the UI.
- **PDF upload replaces the index.** `app/routes/documents.py::upload_document`
  calls `build_vectorstore()`, which overwrites the whole FAISS index instead of
  appending; `vectorstore.py::add_documents()` is the append path.
- **`mrr`/`ndcg` quality gates are skipped.** `QUALITY_GATES` lists them, but
  `run_ragas_evaluation` only writes RAGAS metrics to `ragas_scores.json`, so
  `quality_gate.py` never checks them.
- `semantic_chunking` still uses local `HuggingFaceEmbeddings` (large download),
  unlike the rest of the pipeline which uses the HF Inference endpoint.

## Notes for future work

- `src/logger.py` calls `os.makedirs(logs_path, ...)` on the full log-*file* path
  (not the directory) — a known template quirk. Preserve it unless explicitly fixing.
- `requirements.txt` includes unused/heavy deps (`streamlit`, the `unstructured*`
  stack — the latter only feeds the commented-out `document_aware_chunking`).
