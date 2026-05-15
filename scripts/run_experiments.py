import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectorstore import load_vectorstore
from src.retriever import get_retriever
from src.rag_pipeline import build_rag_chain
from src.document_processor import load_documents, recursive_chunking
from src.evaluator import build_eval_data, run_ragas_evaluation
from src.metrics import run_all_retriever_metrics
from src.experiment_tracker import log_experiment
from src.config import DATA_DIR

# ── Your evaluation questions ──────────────────────────────
QUESTIONS = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is a search algorithm?",
    "What is a neural network?",
    "What is natural language processing?",
]

GROUND_TRUTHS = [
    "AI is the study and design of intelligent agents",
    "Machine learning is a subset of AI that learns from data",
    "A search algorithm finds solutions by exploring possible states",
    "A neural network is a computing system inspired by biological brains",
    "NLP is a branch of AI that deals with human language",
]

RELEVANT_KEYWORDS = [
    ["artificial", "intelligence"],
    ["machine", "learning"],
    ["search", "algorithm"],
    ["neural", "network"],
    ["language", "processing"],
]


def run_single_experiment(name, strategy, vs, all_chunks):
    print(f"\n{'='*50}")
    print(f"Running: {name}")
    print(f"{'='*50}")

    retriever = get_retriever(
        strategy, vs, all_chunks=all_chunks
    )
    rag_chain = build_rag_chain(retriever)

    eval_data = build_eval_data(
        QUESTIONS, retriever, rag_chain, GROUND_TRUTHS
    )
    ragas_scores, _ = run_ragas_evaluation(eval_data, save=False)

    retriever_metrics = run_all_retriever_metrics(
        QUESTIONS, retriever, RELEVANT_KEYWORDS
    )

    config = {
        "retriever_strategy": strategy,
        "chunk_size": 512,
        "embedding_model": "MiniLM",
        "num_questions": len(QUESTIONS),
    }

    log_experiment(name, config, ragas_scores, retriever_metrics)

    return {**ragas_scores, **retriever_metrics}


def main():
    print("Loading vectorstore...")
    vs = load_vectorstore()

    print("Loading chunks for BM25 and hybrid...")
    docs = load_documents(DATA_DIR)
    all_chunks = recursive_chunking(docs)

    experiments = [
        ("exp1-basic-baseline", "basic"),
        ("exp2-hybrid-search", "hybrid"),
        ("exp3-reranker-vector", "reranker_vector"),
        ("exp4-reranker-hybrid", "reranker_hybrid"),
    ]

    results = {}
    for name, strategy in experiments:
        results[name] = run_single_experiment(
            name, strategy, vs, all_chunks
        )

    print("\n\n══ Final Comparison ══")
    metrics = [
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_relevancy",
        "mrr"
    ]
    header = f"{'Experiment':<30}" + "".join(f"{m:<20}" for m in metrics)
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        row = f"{name:<30}" + "".join(
            f"{scores.get(m, 0):<20.4f}" for m in metrics
        )
        print(row)

    print("\nAll experiments logged to MLflow!")
    print("Run: mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db")


if __name__ == "__main__":
    main()