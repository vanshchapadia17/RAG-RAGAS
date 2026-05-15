#Loads your PDF and splits it into chunks. 
'''3 strategies — fixed (split document in N tokens.
recursive (split doc by para then sentence then word this is the defaullt in langchain ).
semantic (use embedding to know teh context of topic whn topic shifs it devide chunks).'''

#chunk size is pre defined in config.py file 

import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader , DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter # type: ignore
from langchain_community.document_loaders import UnstructuredPDFLoader # type: ignore
from langchain_experimental.text_splitter import SemanticChunker # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_core.documents import Document # type: ignore
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHILD_CHUNK_SIZE, DATA_DIR, EMBEDDING_MODEL




def load_documents(data_dir: str = DATA_DIR) -> List[Document]:

    """
    this is a load documents function which load all the documents 
    data_dir: str = DATA_DIR this re present that this function accets the directory path and 
    if path not exist than it uses the default value of DATA_DIR
    and this fucntion returns the list of documents in the form of langchain document schema 
    """

    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True
        )
    
    docs= loader.load()
    print(f"loaded {len(docs)} pages")
    return docs


def load_single_pdf(pdf_path: str) -> List[Document]:
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    print(f"loaded {len(docs)} pages from {pdf_path}")
    return docs


# load_documents()
# load_single_pdf("C:\\Users\\DELL\\Desktop\\RAG\\data\\AI.pdf") used only for one pdf file loading.

def fixed_size_chunking(docs: List[Document]) -> List[Document]:
    
    splitter=CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n"
    )

    chunks=splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_strategy"]="fixed"
        chunk.metadata["chunk_index"]=i
    print(f"created {len(chunks)} fixed size chunks")
    return chunks

def recursive_chunking(docs: List[Document]) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\n\n", "\n", ". ", " ", ""]
    )

    chunks=splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_strategy"]="recursive"
        chunk.metadata["chunk_index"]=i
    print(f"created {len(chunks)} recursive chunks")
    return chunks

def child_chunking(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=20
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_strategy"] = "child"
        chunk.metadata["chunk_index"] = i
    print(f"Child chunking: {len(chunks)} chunks")
    return chunks
    
def semantic_chunking(docs: List[Document]) -> List[Document]:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_strategy"] = "semantic"
        chunk.metadata["chunk_index"] = i
    print(f"Semantic chunking: {len(chunks)} chunks")
    return chunks


"""
def document_aware_chunking(pdf_path: str) -> List[Document]:
    loader = UnstructuredPDFLoader(
        pdf_path,
        mode="elements",        # splits by detected elements
        strategy="auto"       # detects headers, tables, lists
    )
    docs = loader.load()
    for i, doc in enumerate(docs):
        doc.metadata["chunk_strategy"] = "document_aware"
        doc.metadata["chunk_index"] = i
    print(f"Document-aware chunking: {len(docs)} chunks")
    return docs
"""

"""

docs = load_documents()
small = docs[310:315]

r_chunks = recursive_chunking(small)
s_chunks = semantic_chunking(small)
f_chunks = fixed_size_chunking(small)
c_chunks = child_chunking(small)
#d_chunks = document_aware_chunking("C:\\Users\\DELL\\Desktop\\RAG\\data\\AI.pdf")

print('=== COMPARISON ===')
print(f'Recursive: {len(r_chunks)} chunks')
print(f'Semantic:  {len(s_chunks)} chunks')
print(f'Fixed Size: {len(f_chunks)} chunks')
print(f'Child: {len(c_chunks)} chunks')
#print(f'Document-Aware: {len(d_chunks)} chunks')

print()
print('=== Recursive chunk 1 ===')
print(r_chunks[0].page_content[:])

print()
print('=== Semantic chunk 1 ===')
print(s_chunks[0].page_content[:])

print()
print('=== Fixed Size chunk 1 ===')
print(f_chunks[0].page_content[:])

print()
print('=== Child chunk 1 ===')  
print(c_chunks[0].page_content[:])

print()
print('=== Document-Aware chunk 1 ===')
print(d_chunks[0].page_content[:])


print()
print(f'Recursive avg length: {sum(len(c.page_content) for c in r_chunks) // len(r_chunks)} chars')
print(f'Semantic avg length:  {sum(len(c.page_content) for c in s_chunks) // len(s_chunks)} chars')
print(f'Fixed Size avg length: {sum(len(c.page_content) for c in f_chunks) // len(f_chunks)} chars')
print(f'Child avg length: {sum(len(c.page_content) for c in c_chunks) // len(c_chunks)} chars')
#print(f'Document-Aware avg length: {sum(len(c.page_content) for c in d_chunks) // len(d_chunks)} chars')    

"""