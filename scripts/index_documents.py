import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import (
    load_documents,
    recursive_chunking,
    fixed_size_chunking,
    semantic_chunking,
    child_chunking
)
from src.vectorstore import build_vectorstore
from src.config import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Index documents into FAISS")
    parser.add_argument(
        "--strategy",
        default="recursive",
        choices=["recursive", "fixed", "semantic"],
        help="Chunking strategy to use"
    )
    args = parser.parse_args()

    print(f"Loading documents from {DATA_DIR}...")
    docs = load_documents(DATA_DIR)

    if not docs:
        print("No PDFs found in data/ folder!")
        return

    print(f"Using chunking strategy: {args.strategy}")
    if args.strategy == "fixed":
        chunks = fixed_size_chunking(docs)
    elif args.strategy == "semantic":
        chunks = semantic_chunking(docs)
    else:
        chunks = recursive_chunking(docs)

    print(f"Building FAISS vectorstore...")
    vs = build_vectorstore(chunks)

    print(f"\nDone!")
    print(f"Total chunks indexed: {vs.index.ntotal}")
    print(f"You can now start the Flask app: python run.py")


if __name__ == "__main__":
    main()