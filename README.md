# RAG Evaluation System with Flask Dashboard

A Retrieval-Augmented Generation system with a full evaluation pipeline:
RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer
Correctness), retrieval metrics (MRR, NDCG, Precision@K, Recall@K), MLflow
experiment tracking, a CI/CD quality gate, and a 4-page Flask dashboard.

The LLM runs on **Groq** and embeddings/reranking run on the **HuggingFace
Inference API** — no local GPU or large model downloads are required to run the
default pipeline.

## Setup

### 1. Clone and install

```bash
git clone <your-repo>
cd RAG

python -m venv venv
source venv/Scripts/activate      # Windows bash; use venv\Scripts\activate on cmd

pip install -r requirements.txt   # installs deps + this project as an editable package
```

### 2. Configure environment

There is no `.env.example`. Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
FLASK_SECRET_KEY=change-me-in-production
```

- `GROQ_API_KEY` — required, for the chat and judge LLMs.
- `HF_TOKEN` — required, for embeddings and the reranker via the HF Inference API.
- `FLASK_SECRET_KEY` — optional; falls back to an insecure dev default.

### 3. Add your documents

```bash
# Copy your PDF files into the data/ folder
cp your_document.pdf data/
```

### 4. Index documents

```bash
python scripts/index_documents.py
# Choose a chunking strategy (default: recursive):
python scripts/index_documents.py --strategy semantic   # recursive | fixed | semantic
```

This builds a FAISS index under `faiss_db/`.

### 5. Start the Flask dashboard

```bash
python run.py
# Open http://localhost:5000
```

## Dashboard Pages

| Page | URL | What it does |
|---|---|---|
| Chat | `/` | Ask questions, see sources, give feedback |
| Documents | `/documents` | Upload PDFs, view index stats |
| Evaluate | `/eval` | Run RAGAS + MRR + NDCG on a question set |
| Experiments | `/experiments` | Compare all MLflow runs |

## Run Experiments

```bash
# Run the 4 structured experiments (basic / hybrid / reranker-vector / reranker-hybrid)
python scripts/run_experiments.py

# View results in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# Open http://localhost:5000 (or the port MLflow prints)
```

## Quality Gate (CI/CD)

```bash
python scripts/quality_gate.py
# exit 0 = deploy, exit 1 = block
```

Thresholds live in `QUALITY_GATES` in `src/config.py`. The gate reads
`eval/ragas_scores.json`, so run an evaluation first.

## Project Structure

```
RAG/
├── app/                        Flask dashboard
│   ├── __init__.py             create_app() application factory
│   ├── routes/                 chat, documents, eval, experiments blueprints
│   ├── templates/              Jinja2 HTML templates
│   └── static/                 CSS + JS
├── src/                        Core RAG logic
│   ├── config.py               All settings, paths, and quality-gate thresholds
│   ├── document_processor.py   PDF loading + chunking strategies
│   ├── vectorstore.py          FAISS build / load / stats
│   ├── retriever.py            8 retrieval strategies
│   ├── rag_pipeline.py         LLM chain, prompt, query_with_sources()
│   ├── evaluator.py            RAGAS evaluation (5 metrics)
│   ├── metrics.py              MRR, NDCG, Precision@K, Recall@K
│   ├── experiment_tracker.py   MLflow logging + quality-gate checks
│   ├── logger.py               Shared logging setup
│   └── exception.py            CustomException error wrapper
├── scripts/
│   ├── index_documents.py      Build the FAISS index
│   ├── run_experiments.py      Run the 4 structured experiments
│   └── quality_gate.py         CI/CD pass/fail check
├── data/                       Your PDF files go here
├── eval/                       ragas_scores.json + feedback_log.csv
├── outputs/                    per_question_scores.csv
├── faiss_db/                   Saved FAISS index
├── mlruns/                     MLflow tracking store (SQLite)
├── run.py                      Flask entry point
└── requirements.txt
```

## Topics Covered

- **Chunking:** fixed-size, recursive, semantic, child
- **Embeddings:** BGE / MiniLM via the HuggingFace Inference API
- **Retrieval (8 strategies):** basic vector, BM25, hybrid (BM25 + vector via RRF),
  BGE reranker on vector, BGE reranker on hybrid, multi-query, MMR, parent-document
- **Evaluation:** RAGAS (5 metrics) + MRR + NDCG + Precision@K + Recall@K
- **Tracking:** MLflow experiment comparison
- **CI/CD:** quality-gate script
- **Dashboard:** Flask with 4 pages
