"""
in this file we have comparing diff retrieval startegies, including:
- Basic Vector Search 
- BM25 Keyword Search
- Hybrid BM25 + Vector
- Reranker on vectorstore (BGE)
- Reranker on hybrid (BM25 + vector + BGE)
- Multi-Query
- MMR (Maximal Marginal Relevance)
- Parent Document Retriever (small child chunks, returns large parent chunks)

so after user write question it convert those question first in vectores 
and return most relevenat chunks with using these 8 strategies 

RRF is not a separate retriever — it is the algorithm used inside EnsembleRetriever 
to merge BM25 + vector results.
"""


import os 
from typing import List
from langchain_core.documents import Document # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever # type: ignore
from langchain.retrievers.document_compressors import CrossEncoderReranker # type: ignore
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.multi_query import MultiQueryRetriever # type: ignore
from langchain_groq import ChatGroq
from langchain.storage import InMemoryStore # type: ignore
from langchain.retrievers import ParentDocumentRetriever # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from src.config import (
    TOP_K, RERANK_TOP_N, BM25_WEIGHT, VECTOR_WEIGHT,
    RERANKER_MODEL, LLM_MODEL, CHUNK_SIZE, CHILD_CHUNK_SIZE
)

def get_basic_retriever(vectorstore: FAISS, k: int = RERANK_TOP_N): #value of top n is 5 
    """
    Simple cosine similarity search.
    Use as Experiment 1 baseline.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

def get_bm25_retriever(
    all_chunks: List[Document],
    k: int = RERANK_TOP_N
):
    """
    Pure keyword search — no vectors needed.
    Good baseline for keyword-heavy documents.
    Uses BM25 ranking algorithm.
    """
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = k
    return bm25_retriever

def get_hybrid_retriever(
    vectorstore: FAISS,
    all_chunks: List[Document],
    k: int = TOP_K
):
    """
    Combines BM25 keyword search + vector semantic search.
    Best single improvement over baseline.
    """
    # BM25 — keyword based
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = k

    # Vector — semantic based
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    # Combine both using Reciprocal Rank Fusion
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[BM25_WEIGHT, VECTOR_WEIGHT]
    )
    return hybrid_retriever

def get_reranker_on_vector(
    vectorstore: FAISS,
    top_n: int = RERANK_TOP_N
):
    """
    Stage 1 — pure vector search gets Top-20
    Stage 2 — BGE reranker picks Top-5
    No BM25 involved.
    """
    # Stage 1 — vector only
    base_retriever = get_basic_retriever(vectorstore, k=TOP_K)

    # Stage 2 — BGE reranker
    model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=model, top_n=top_n)

    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )

def get_reranker_on_hybrid(
    vectorstore: FAISS,
    all_chunks: List[Document],
    top_n: int = RERANK_TOP_N
):
    """
    Stage 1 — hybrid search (BM25 + vector) gets Top-20
    Stage 2 — BGE reranker picks Top-5
    Best overall quality.
    """
    # Stage 1
    base_retriever = get_hybrid_retriever(
        vectorstore, all_chunks, k=TOP_K
    )

    # Stage 2
    model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=model, top_n=top_n)

    reranker_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return reranker_retriever

def get_multiquery_retriever(vectorstore: FAISS, k: int = RERANK_TOP_N):
    """
    Generates 3-5 different phrasings of the query.
    Retrieves for each, merges results.
    Improves Context Recall.
    """
    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

def get_mmr_retriever(vectorstore: FAISS, k: int = RERANK_TOP_N):
    """
    Maximal Marginal Relevance.
    Returns diverse results — no duplicate chunks.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": TOP_K,
            "lambda_mult": 0.5
        }
    )

def get_parent_document_retriever(
    vectorstore: FAISS,
    all_docs: List[Document]
):
    """
    Searches using small child chunks.
    Returns large parent chunks to LLM.
    Best context quality.
    """
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=20
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=50
    )

    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    retriever.add_documents(all_docs)
    return retriever

def get_retriever(
    strategy: str,
    vectorstore: FAISS,
    all_chunks: List[Document] = None,
    all_docs: List[Document] = None
):
    """
    Get any retriever by name.
    strategies: basic, hybrid, reranker, multiquery, mmr, parent
    """
    strategies = {
        "basic": lambda: get_basic_retriever(vectorstore),
        "bm25": lambda: get_bm25_retriever(all_chunks),
        "hybrid": lambda: get_hybrid_retriever(vectorstore, all_chunks),
        "reranker_vector": lambda: get_reranker_on_vector(vectorstore),
        "reranker_hybrid": lambda: get_reranker_on_hybrid(vectorstore, all_chunks),
        "multiquery": lambda: get_multiquery_retriever(vectorstore),
        "mmr": lambda: get_mmr_retriever(vectorstore),
        "parent": lambda: get_parent_document_retriever(vectorstore, all_docs),
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {list(strategies.keys())}"
        )

    print(f"Using retriever: {strategy}")
    return strategies[strategy]()

