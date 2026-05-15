import csv
import os
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify
from src.vectorstore import load_vectorstore
from src.retriever import get_retriever
from src.rag_pipeline import query_with_sources
from src.config import FEEDBACK_LOG, EVAL_DIR

chat_bp = Blueprint("chat", __name__)

_retriever = None

def get_pipeline():
    global _retriever
    if _retriever is None:
        try:
            vs = load_vectorstore()
            _retriever = get_retriever("basic", vs)
        except Exception as e:
            print(f"Pipeline error: {e}")
            _retriever = None
    return _retriever


@chat_bp.route("/")
def index():
    return render_template("chat.html")


@chat_bp.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    retriever = get_pipeline()
    if not retriever:
        return jsonify({"error": "Pipeline not ready"}), 503

    try:
        result = query_with_sources(retriever, question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    os.makedirs(EVAL_DIR, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(),
        "question": data.get("question", ""),
        "answer": data.get("answer", ""),
        "rating": data.get("rating", ""),
    }

    write_header = not os.path.exists(FEEDBACK_LOG)
    with open(FEEDBACK_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return jsonify({"status": "saved"})