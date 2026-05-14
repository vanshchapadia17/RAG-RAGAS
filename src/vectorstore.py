"""
after chunking the data move forward to convert those data to embeddings , embeddings means converting text to number 
and these numbers captures the meaning of text , same context of text equals to same numbers or vector 
"""

import os 
from typing import List 
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_core.documents import Document # type: ignore
from src.config import FAISS_DIR, EMBEDDING_MODEL
from src.document_processor import recursive_chunking, load_documents

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True}
        )

def build_vectorstore(chunks: List[Document]) -> FAISS:
    print(f"Building FAISS index for {len(chunks)} chunks...")
    embeddings=get_embeddings()

    vectorstore=FAISS.from_documents(
        documents=chunks, 
        embedding=embeddings)
    
    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_DIR)
    print(f"FAISS index saved to {FAISS_DIR}")
    print(f"Total vectors:{vectorstore.index.ntotal}")

    return vectorstore

def load_vectorstore() -> FAISS:
    if not os.path.exists(FAISS_DIR):
        raise FileNotFoundError(f"No FAISS index found. Run build_vectorestore() first.")
    
    embeddings=get_embeddings()
    vectorstore=FAISS.load_local(
        FAISS_DIR, 
        embeddings,
        allow_dangerous_deserialization=True)
    
    print(f"FAISS index loaded from {FAISS_DIR}")
    print(f"Total vectors:{vectorstore.index.ntotal}")
    return vectorstore

def get_vectorstore_stats(vectorstore: FAISS) -> dict:
    return {
        "total_chunks": vectorstore.index.ntotal,
        "index_type": "FAISS",
        "location": FAISS_DIR
    }

def add_documents(vectorstore: FAISS, new_chunks: List[Document]) -> FAISS:
    vectorstore.add_documents(new_chunks)
    vectorstore.save_local(FAISS_DIR)
    print(f"Added {len(new_chunks)} chunks. Total: {vectorstore.index.ntotal}")
    return vectorstore


docs = load_documents()
chunks = recursive_chunking(docs)
vs = build_vectorstore(chunks)
stats = get_vectorstore_stats(vs)
print(stats)

"""
# Test a simple search
results = vs.similarity_search('What is artificial intelligence?', k=3)
print('Search results:')
for i, r in enumerate(results):
    print(f'Result {i+1}:', r.page_content[:150])
"""