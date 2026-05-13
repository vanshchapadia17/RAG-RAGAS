import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = "llama-3.1-8b-instant"
LLM_JUDGE_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-large"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
CHILD_CHUNK_SIZE = 128

TOP_K = 20
RERANK_TOP_N = 5
BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVAL_DIR = os.path.join(BASE_DIR, "eval")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_db")
FEEDBACK_LOG = os.path.join(EVAL_DIR, "feedback_log.csv")
SCORES_FILE = os.path.join(EVAL_DIR, "ragas_scores.json")

QUALITY_GATES = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.70,
    "answer_correctness": 0.65,
    "mrr": 0.60,
    "ndcg": 0.65,
}