import json
import os
from flask import Blueprint, render_template, request, jsonify
from src.vectorstore import load_vectorstore
from src.retriever import get_retriever
from src.rag_pipeline import build_rag_chain
from src.evaluator import build_eval_data, run_ragas_evaluation, save_scores
from src.metrics import run_all_retriever_metrics
from src.experiment_tracker import log_experiment, check_quality_gates
from src.config import SCORES_FILE

eval_bp = Blueprint("eval", __name__)


@eval_bp.route("/eval")
def eval_page():
    scores = {}
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            scores = json.load(f)
    return render_template("eval.html", scores=scores)


@eval_bp.route("/api/run-eval", methods=["POST"])
def run_eval():
    data = request.get_json()
    retriever_strategy = data.get("retriever", "basic")
    questions = data.get("questions", [])
    ground_truths = data.get("ground_truths", None)
    run_name = data.get("run_name", f"eval-{retriever_strategy}")

    if not questions:
        return jsonify({"error": "No questions provided"}), 400

    try:
        vs = load_vectorstore()
        retriever = get_retriever(retriever_strategy, vs)
        rag_chain = build_rag_chain(retriever)

        eval_data = build_eval_data(
            questions, retriever, rag_chain, ground_truths
        )
        ragas_scores, df = run_ragas_evaluation(eval_data, save=True)

        relevant_keywords = [[q.split()[0]] for q in questions]
        retriever_metrics = run_all_retriever_metrics(
            questions, retriever, relevant_keywords
        )

        config = {
            "retriever_strategy": retriever_strategy,
            "num_questions": len(questions),
        }
        log_experiment(run_name, config, ragas_scores, retriever_metrics)

        all_scores = {**ragas_scores, **retriever_metrics}
        # Persist the combined set so quality_gate.py can check mrr/ndcg too.
        save_scores(all_scores)
        gates = check_quality_gates(all_scores)

        return jsonify({
            "ragas_scores": ragas_scores,
            "retriever_metrics": retriever_metrics,
            "quality_gates": gates,
            "run_name": run_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@eval_bp.route("/api/scores")
def get_scores():
    if not os.path.exists(SCORES_FILE):
        return jsonify({})
    with open(SCORES_FILE) as f:
        return jsonify(json.load(f))