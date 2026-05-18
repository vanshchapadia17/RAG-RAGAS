import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import load_documents
from src.evaluator import generate_eval_dataset
from src.config import DATA_DIR, DATASET_FILE


def main():
    print("Loading documents...")
    docs = load_documents(DATA_DIR)

    if not docs:
        print("No PDFs found in data/ folder!")
        return

    print(f"Loaded {len(docs)} pages")

    df = generate_eval_dataset(docs, test_size=10, save=True)

    print("\nGenerated dataset:")
    print(df[["question", "ground_truth"]].to_string())
    print(f"\nSaved to {DATASET_FILE}")
    print("Now run: python scripts/run_experiments.py")


if __name__ == "__main__":
    main()