import os
import mlflow
import pandas as pd
from typing import Dict, Any
from src.config import OUTPUTS_DIR, QUALITY_GATES

def start_experiment(experiment_name: str = "rag-evaluation"):
    """Set up MLflow experiment."""
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment(experiment_name)

def check_quality_gates(scores: Dict[str, float]) -> Dict:
    """
    Check all scores against thresholds from config.
    Used in CI/CD to block deployment if quality drops.
    """
    failed = []
    for metric, threshold in QUALITY_GATES.items():
        if metric in scores and scores[metric] < threshold:
            failed.append(
                f"{metric}={scores[metric]:.3f} (need {threshold})"
            )

    return {
        "all_passed": len(failed) == 0,
        "failed": failed,
        "scores": scores
    }

def log_experiment(
    run_name: str,
    config: Dict[str, Any],
    ragas_scores: Dict[str, float],
    retriever_metrics: Dict[str, float] = None,
    tags: Dict[str, str] = None
) -> str:
    """
    Log full experiment run to MLflow.

    Args:
        run_name: descriptive name e.g. exp3-hybrid-bge-reranker
        config: pipeline config (chunk_size, embedding, retriever)
        ragas_scores: RAGAS metric scores
        retriever_metrics: MRR, NDCG, P@K, R@K scores
        tags: optional key-value tags

    Returns:
        run_id string
    """
    start_experiment()

    with mlflow.start_run(run_name=run_name) as run:

        # log pipeline configuration
        mlflow.log_params(config)

        # log RAGAS scores
        for metric, score in ragas_scores.items():
            mlflow.log_metric(f"ragas_{metric}", score)

        # log retriever metrics
        if retriever_metrics:
            for metric, score in retriever_metrics.items():
                # MLflow doesn't allow @ in metric names
                clean_metric = metric.replace("@", "_at_")
                mlflow.log_metric(clean_metric, score)

        # log quality gate results
        all_scores = {**ragas_scores, **(retriever_metrics or {})}
        gates = check_quality_gates(all_scores)
        mlflow.log_param("quality_gates_passed", gates["all_passed"])
        mlflow.log_param("failed_gates", str(gates["failed"]))

        # log tags
        if tags:
            mlflow.set_tags(tags)

        run_id = run.info.run_id

        print(f"\nMLflow run logged: {run_name}")
        print(f"Run ID: {run_id}")
        print(f"Quality gates passed: {gates['all_passed']}")
        if gates["failed"]:
            print(f"Failed gates: {gates['failed']}")

    return run_id

def get_all_experiments() -> pd.DataFrame:
    """
    Fetch all experiment runs as DataFrame.
    Used by Flask dashboard experiments page.
    """
    start_experiment()
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("rag-evaluation")
    if not experiment:
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    if not runs:
        return pd.DataFrame()

    records = []
    for run in runs:
        record = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": pd.to_datetime(
                run.info.start_time, unit="ms"
            ),
            **{f"param_{k}": v for k, v in run.data.params.items()},
            **run.data.metrics
        }
        records.append(record)

    return pd.DataFrame(records)

def get_best_run(metric: str = "ragas_faithfulness") -> Dict:
    """Return the run with highest score for a given metric."""
    df = get_all_experiments()
    if df.empty or metric not in df.columns:
        return {}

    best = df.loc[df[metric].idxmax()]
    return best.to_dict()



"""
from src.vectorstore import load_vectorstore
from src.retriever import get_retriever
from src.rag_pipeline import build_rag_chain
from src.evaluator import build_eval_data, run_ragas_evaluation
from src.metrics import run_all_retriever_metrics
from src.experiment_tracker import log_experiment, get_all_experiments

vs = load_vectorstore()
retriever = get_retriever('basic', vs)
rag_chain = build_rag_chain(retriever)

questions = ['What is artificial intelligence?', 'What is machine learning?']
ground_truths = [
    'AI is the study of intelligent agents',
    'ML is a subset of AI that learns from data'
]

data = build_eval_data(questions, retriever, rag_chain, ground_truths)
ragas_scores, df = run_ragas_evaluation(data, save=True)

relevant_keywords = [['artificial', 'intelligence'], ['machine', 'learning']]
retriever_metrics = run_all_retriever_metrics(questions, retriever, relevant_keywords)

config = {
    'retriever_strategy': 'basic',
    'chunk_size': 512,
    'embedding_model': 'MiniLM',
    'reranker': 'none',
    'num_questions': len(questions)
}

run_id = log_experiment(
    run_name='exp1-basic-baseline',
    config=config,
    ragas_scores=ragas_scores,
    retriever_metrics=retriever_metrics
)

print('Run ID:', run_id)
print()
print('All experiments:')
df = get_all_experiments()
print(df[['run_name', 'ragas_faithfulness', 'mrr']].to_string())
"""