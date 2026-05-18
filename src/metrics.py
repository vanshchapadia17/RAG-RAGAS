import re
import numpy as np
from typing import List,Dict,Any,Tuple
from sklearn.metrics import ndcg_score

def compute_mrr(
    queries: List[str],
    retriever,
    relevant_keywords: List[List[str]]
) -> Tuple[float, List[float]]:
    reciprocal_ranks = []

    for query, keywords in zip(queries, relevant_keywords):
        docs = retriever.invoke(query)
        rr = 0.0

        for rank, doc in enumerate(docs, start=1):
            content_lower = doc.page_content.lower()
            if any(kw.lower() in content_lower for kw in keywords):
                rr = 1.0 / rank
                break

        reciprocal_ranks.append(rr)

    mrr = float(np.mean(reciprocal_ranks))
    return mrr, reciprocal_ranks

def compute_precision_recall_at_k(
    queries: List[str],
    retriever,
    relevant_keywords: List[List[str]],
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    results = {f"precision@{k}": [] for k in k_values}
    results.update({f"recall@{k}": [] for k in k_values})

    for query, keywords in zip(queries, relevant_keywords):
        docs = retriever.invoke(query)

        relevance = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            is_relevant = any(kw.lower() in content_lower for kw in keywords)
            relevance.append(1 if is_relevant else 0)

        total_relevant = sum(relevance)

        for k in k_values:
            top_k = relevance[:k]
            relevant_in_k = sum(top_k)
            precision = relevant_in_k / k if k > 0 else 0

            # Fix: handle total_relevant = 0
            if total_relevant == 0:
                recall = 0.0    # no relevant docs found at all
            else:
                recall = relevant_in_k / total_relevant

            results[f"precision@{k}"].append(precision)
            results[f"recall@{k}"].append(recall)

    return {key: float(np.mean(vals)) for key, vals in results.items()}

def compute_ndcg(
    queries: List[str],
    retriever,
    graded_relevance: List[List[Tuple[str, int]]],
    k: int = 5
) -> float:
    ndcg_scores = []

    for query, grade_rules in zip(queries, graded_relevance):
        docs = retriever.invoke(query)[:k]

        true_grades = []
        pred_grades = []

        for rank, doc in enumerate(docs):
            content_lower = doc.page_content.lower()
            grade = 0
            for keyword, g in grade_rules:
                if keyword.lower() in content_lower:
                    grade = max(grade, g)
            true_grades.append(grade)
            pred_grades.append(k - rank)

        if not true_grades or sum(true_grades) == 0:
            ndcg_scores.append(0.0)
            continue

        while len(true_grades) < k:
            true_grades.append(0)
            pred_grades.append(0)

        try:
            score = ndcg_score([true_grades], [pred_grades], k=k)
            ndcg_scores.append(score)
        except Exception:
            ndcg_scores.append(0.0)

    return float(np.mean(ndcg_scores))

def run_all_retriever_metrics(
    queries: List[str],
    retriever,
    relevant_keywords: List[List[str]],
    graded_relevance: List[List[Tuple[str, int]]] = None
) -> Dict[str, float]:

    print("Computing MRR...")
    mrr, rr_list = compute_mrr(queries, retriever, relevant_keywords)

    print("Computing Precision@K and Recall@K...")
    pr_metrics = compute_precision_recall_at_k(
        queries, retriever, relevant_keywords, k_values=[1, 3, 5]
    )

    metrics = {"mrr": mrr, **pr_metrics}

    # NDCG needs graded relevance. When the caller supplies no explicit grades,
    # fall back to binary relevance (each keyword counts as grade 1) so NDCG —
    # and its quality gate — is always evaluated. The key is "ndcg" (no "@5")
    # to match the QUALITY_GATES config.
    if not graded_relevance:
        graded_relevance = [
            [(kw, 1) for kw in keywords] for keywords in relevant_keywords
        ]
    print("Computing NDCG@5...")
    metrics["ndcg"] = compute_ndcg(queries, retriever, graded_relevance, k=5)

    print("\n── Retriever Metrics ──")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics

def auto_keywords(questions):
    """
    Automatically extract keywords from questions.
    No manual keyword definition needed.
    Works for any document and any questions.
    """
    stopwords = {
        'what', 'is', 'are', 'the', 'a', 'an', 'how',
        'why', 'when', 'where', 'who', 'which', 'do',
        'does', 'did', 'was', 'were', 'will', 'can',
        'could', 'would', 'should', 'of', 'in', 'on',
        'at', 'to', 'for', 'with', 'by', 'from', 'it',
        'its', 'this', 'that', 'these', 'those', 'and',
        'or', 'but', 'not', 'be', 'been', 'being', 'has',
        'have', 'had', 'me', 'my', 'we', 'our', 'you'
    }
    keywords = []
    for q in questions:
        words = re.findall(r'\b\w+\b', q.lower())
        kw = [w for w in words if w not in stopwords and len(w) > 2]
        keywords.append(kw if kw else [words[0]])
    return keywords

"""
from src.vectorstore import load_vectorstore
from src.retriever import get_retriever
from src.metrics import run_all_retriever_metrics

print('Loading vectorstore...')
vs = load_vectorstore()
retriever = get_retriever('basic', vs)

queries = [
    'What is artificial intelligence?',
    'What is machine learning?',
    'What is search algorithm?'
]

relevant_keywords = [
    ['artificial', 'intelligence', 'ai'],   # separate words
    ['machine', 'learning'],
    ['search', 'algorithm']
]

graded_relevance = [
    [('artificial intelligence', 2), ('AI', 2), ('intelligent', 1)],
    [('machine learning', 2), ('ML', 2), ('learning', 1)],
    [('search algorithm', 2), ('search', 1), ('algorithm', 1)]
]

metrics = run_all_retriever_metrics(
    queries,
    retriever,
    relevant_keywords,
    graded_relevance
)

print()
print('Final scores:')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')
"""