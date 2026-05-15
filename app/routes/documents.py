import os
from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
from src.document_processor import load_single_pdf, recursive_chunking
from src.vectorstore import (
    build_vectorstore,
    load_vectorstore,
    get_vectorstore_stats,
    add_documents,
)
from src.config import DATA_DIR, FAISS_DIR

documents_bp = Blueprint("documents", __name__)


@documents_bp.route("/documents")
def documents_page():
    pdfs = []
    if os.path.exists(DATA_DIR):
        pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    return render_template("documents.html", pdfs=pdfs)


@documents_bp.route("/api/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file or not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files allowed"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    file.save(save_path)

    try:
        docs = load_single_pdf(save_path)
        chunks = recursive_chunking(docs)

        # Append to the existing index so previously indexed PDFs are kept.
        # Only build a fresh index when none exists yet.
        index_exists = os.path.exists(os.path.join(FAISS_DIR, "index.faiss"))
        if index_exists:
            vs = load_vectorstore()
            vs = add_documents(vs, chunks)
        else:
            vs = build_vectorstore(chunks)

        stats = get_vectorstore_stats(vs)

        return jsonify({
            "message": f"Indexed {filename} successfully",
            "pages": len(docs),
            "chunks": len(chunks),
            "total_chunks": stats["total_chunks"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@documents_bp.route("/api/index-stats")
def index_stats():
    try:
        vs = load_vectorstore()
        return jsonify(get_vectorstore_stats(vs))
    except Exception:
        return jsonify({"total_chunks": 0, "index_type": "FAISS"})