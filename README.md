# RAG Evaluation System with Flask Dashboard

A complete RAG system with full evaluation pipeline covering:
RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness),
MRR, NDCG, Precision@K, Recall@K, MLflow experiment tracking, and a Flask dashboard.

## Setup

### 1. Clone and install
```bash
git clone <your-repo>
cd rag-project
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and LANGCHAIN_API_KEY
```

### 3. Add your documents
```bash
# Copy your PDF files into the data/ folder
cp your_document.pdf data/
```

### 4. Index documents
```bash
python scripts/index_documents.py
# Or compare chunking strategies:
python scripts/index_documents.py --strategy compare
```

### 5. Start the Flask dashboard
```bash
python run.py
# Open http://localhost:5000
```

## Dashboard Pages

| Page | URL | What it does |
|---|---|---|
| Chat | / | Ask questions, see sources, give feedback |
| Documents | /documents | Upload PDFs, view index stats |
| Evaluate | /eval | Run RAGAS + MRR + NDCG |
| Experiments | /experiments | Compare all MLflow runs |

## Run Experiments

```bash
# Run all 4 structured experiments
python scripts/run_experiments.py

# View in MLflow UI
mlflow ui
# Open http://localhost:5001
```

## Quality Gate (CI/CD)

```bash
python scripts/quality_gate.py
# exit 0 = deploy, exit 1 = block
```

## Project Structure

```
rag-project/
├── app/                    Flask dashboard
│   ├── routes/             chat, eval, experiments, documents
│   ├── templates/          HTML templates
│   └── static/             CSS + JS
├── src/                    Core RAG logic
│   ├── config.py           All settings in one place
│   ├── document_processor.py  Chunking strategies
│   ├── vectorstore.py      Chroma vector store
│   ├── retriever.py        All 7 retrieval strategies
│   ├── rag_pipeline.py     LLM chain + prompts
│   ├── evaluator.py        RAGAS evaluation
│   ├── metrics.py          MRR, NDCG, P@K, R@K
│   └── experiment_tracker.py  MLflow logging
├── scripts/
│   ├── index_documents.py  One-time indexing script
│   ├── run_experiments.py  Run all 4 experiments
│   └── quality_gate.py     CI/CD quality check
├── data/                   Your PDF files go here
├── eval/                   Evaluation datasets + scores
├── outputs/                Reports + per-question scores
└── notebooks/              Jupyter notebooks for exploration
```

## Topics Covered

- Chunking: fixed, recursive, semantic, parent-document
- Embeddings: OpenAI text-embedding-3-small, BGE-large
- Retrieval: basic, hybrid (BM25+vector), BGE reranker, HyDE, multi-query, MMR
- Evaluation: RAGAS (5 metrics) + MRR + NDCG + Precision@K + Recall@K
- Tracking: MLflow experiment comparison
- Tracing: LangSmith per-query debugging
- CI/CD: quality gate script
- Dashboard: Flask with 4 pages