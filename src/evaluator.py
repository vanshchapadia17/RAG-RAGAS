import json
import os
from ragas.executor import RunConfig
from unittest import result
import pandas as pd
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings.huggingface_endpoint import HuggingFaceEndpointEmbeddings # type: ignore
from src.config import (
    LLM_JUDGE_MODEL,
    SCORES_FILE,
    OUTPUTS_DIR,
    HF_TOKEN
)

def get_judge_llm():
    return ChatGroq(model=LLM_JUDGE_MODEL, temperature=0)


def get_judge_embeddings():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN
    )

def extract_score(value) -> float:
    """Handle both float and list return types from RAGAS."""
    if isinstance(value, list):
        return round(float(sum(value) / len(value)), 4)
    return round(float(value), 4)

def save_scores(scores: Dict) -> None:
    """Persist the full metric set (RAGAS + retriever metrics) to SCORES_FILE.

    quality_gate.py reads this file, so every gated metric — including mrr and
    ndcg — must be written here, not just the RAGAS scores.
    """
    os.makedirs(os.path.dirname(SCORES_FILE), exist_ok=True)
    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"All scores saved to {SCORES_FILE}")

def build_eval_data(
    questions: List[str],
    retriever,
    rag_chain,
    ground_truths: List[str] = None
) -> Dict:
    print(f"Running RAG on {len(questions)} questions...")

    all_questions = []
    all_answers = []
    all_contexts = []
    all_ground_truths = []

    for i, question in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {question[:60]}...")

        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        answer = rag_chain.invoke(question)

        all_questions.append(question)
        all_answers.append(answer)
        all_contexts.append(contexts)

        if ground_truths:
            all_ground_truths.append(ground_truths[i])
        else:
            all_ground_truths.append("")

    return {
        "question": all_questions,
        "answer": all_answers,
        "contexts": all_contexts,
        "ground_truth": all_ground_truths
    }

def run_ragas_evaluation(
    data: Dict,
    save: bool = True
) -> tuple:
    print("Running RAGAS evaluation...")

    dataset = Dataset.from_dict(data)
    judge_llm = get_judge_llm()
    judge_embeddings = get_judge_embeddings()

    has_ground_truth = any(gt != "" for gt in data["ground_truth"])
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    if has_ground_truth:
        metrics.append(answer_correctness)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings
    )

    scores = {
    "faithfulness": extract_score(result["faithfulness"]),
    "answer_relevancy": extract_score(result["answer_relevancy"]),
    "context_precision": extract_score(result["context_precision"]),
    "context_recall": extract_score(result["context_recall"]),
    }
    if has_ground_truth:
        scores["answer_correctness"] = extract_score(
            result["answer_correctness"]
        )

    print("\n── RAGAS Scores ──")
    for metric, score in scores.items():
        status = "✓" if score >= 0.75 else "✗"
        print(f"  {status} {metric}: {score:.4f}")

    if save:
        os.makedirs(os.path.dirname(SCORES_FILE), exist_ok=True)
        with open(SCORES_FILE, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"Scores saved to {SCORES_FILE}")

        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        df = result.to_pandas()
        df.to_csv(
            os.path.join(OUTPUTS_DIR, "per_question_scores.csv"),
            index=False
        )

    return scores, result.to_pandas()

def analyze_failures(
    df: pd.DataFrame,
    metric: str = "faithfulness",
    n: int = 5
):
    if metric not in df.columns:
        print(f"Metric '{metric}' not in results.")
        return

    worst = df.nsmallest(n, metric)[
        ["question", metric, "answer"]
    ]

    print(f"\n── Top {n} worst by {metric} ──")
    for _, row in worst.iterrows():
        print(f"\n  Score: {row[metric]:.3f}")
        print(f"  Q: {row['question'][:80]}")
        print(f"  A: {row['answer'][:120]}...")

    return worst


def generate_eval_dataset(
    docs,
    test_size: int = 10,
    save: bool = True
) -> pd.DataFrame:
    """
    Auto generate evaluation dataset from your documents.
    No manual question writing needed!
    """
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers import (
        SingleHopSpecificQuerySynthesizer,
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer
    )
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    from src.config import HF_TOKEN, LLM_MODEL, DATASET_FILE ,EVAL_DIR

    print(f"Generating {test_size} questions from your documents...")

    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN
    )

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings
    )

    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=5,
        query_distribution=[
        (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25)
        ],
        run_config=RunConfig(
        max_retries=10,
        max_wait=120,
        timeout=600
        ),
        raise_exceptions=False
    )

    df = testset.to_pandas()

    if save:
        os.makedirs(EVAL_DIR, exist_ok=True)
        df.to_csv(DATASET_FILE, index=False)
        print(f"Dataset saved to {DATASET_FILE}")

    print(f"Generated {len(df)} Q&A pairs")
    return df

"""
from src.vectorstore import load_vectorstore
from src.retriever import get_retriever
from src.rag_pipeline import build_rag_chain
from src.evaluator import build_eval_data, run_ragas_evaluation

vs = load_vectorstore()
retriever = get_retriever('basic', vs)
rag_chain = build_rag_chain(retriever)

questions = [
    'What is artificial intelligence?',
    'What is machine learning?',
]

ground_truths = [
    'Artificial intelligence is the study of intelligent agents',
    'Machine learning is a subset of AI that learns from data',
]

data = build_eval_data(questions, retriever, rag_chain, ground_truths)
scores, df = run_ragas_evaluation(data, save=True)
print(scores)
"""