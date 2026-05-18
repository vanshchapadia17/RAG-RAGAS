import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.config import DATASET_FILE
from src.vectorstore import load_vectorstore
from src.retriever import get_retriever
from src.rag_pipeline import build_rag_chain
from src.evaluator import build_eval_data, run_ragas_evaluation
from src.metrics import run_all_retriever_metrics
from src.experiment_tracker import log_experiment
from src.config import DATA_DIR

from src.document_processor import (
    load_documents,
    recursive_chunking,
    fixed_size_chunking,
    semantic_chunking,
    child_chunking
)

from src.metrics import run_all_retriever_metrics, auto_keywords

def get_chunks(docs, strategy="recursive"):
    """Get chunks using specified strategy."""
    if strategy == "fixed":
        return fixed_size_chunking(docs)
    elif strategy == "semantic":
        return semantic_chunking(docs)
    elif strategy == "child":
        return child_chunking(docs)
    else:
        return recursive_chunking(docs)

def load_eval_dataset():
    if not os.path.exists(DATASET_FILE):
        print("No dataset found!")
        sys.exit(1)

    df = pd.read_csv(DATASET_FILE)
    
    # Start with first 10 questions only for testing
    df = df.head(10)
    
    questions = df["question"].tolist()
    ground_truths = df["ground_truth"].tolist()
    print(f"Loaded {len(questions)} questions from dataset")
    return questions, ground_truths


def run_single_experiment(name, strategy, vs, all_chunks, questions, ground_truths):
    print(f"\n{'='*50}")
    print(f"Running: {name}")
    print(f"{'='*50}")

    retriever = get_retriever(
        strategy, vs, all_chunks=all_chunks
    )
    rag_chain = build_rag_chain(retriever)

    eval_data = build_eval_data(
        questions, retriever, rag_chain, ground_truths
    )
    ragas_scores, _ = run_ragas_evaluation(eval_data, save=False)

    relevant_keywords = auto_keywords(questions)
    retriever_metrics = run_all_retriever_metrics(
        questions, retriever, relevant_keywords
    )

    config = {
        "retriever_strategy": strategy,
        "chunk_size": 512,
        "embedding_model": "MiniLM",
        "num_questions": len(questions),
    }

    log_experiment(name, config, ragas_scores, retriever_metrics)

    return {**ragas_scores, **retriever_metrics}


def main():
    # Load auto generated dataset
    QUESTIONS, GROUND_TRUTHS = load_eval_dataset()

    print("Loading vectorstore...")
    vs = load_vectorstore()

    print("Loading chunks for BM25 and hybrid...")
    docs = load_documents(DATA_DIR)
    all_chunks = recursive_chunking(docs)

    experiments = [
    ("exp1-basic-baseline", "basic"),
    ("exp2-reranker-hybrid", "reranker_hybrid"),
    ]

    results = {}
    for name, strategy in experiments:
        results[name] = run_single_experiment(
            name, strategy, vs, all_chunks, QUESTIONS, GROUND_TRUTHS
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