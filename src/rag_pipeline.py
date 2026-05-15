import os 
from typing import List, Dict, Any
from langchain_groq import ChatGroq # type: ignore
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.documents import Document # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain_core.runnables import RunnablePassthrough # type: ignore
from src.config import LLM_MODEL

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. 
        Answer the user question ONLY using the provided context.

        Rules:
        - Answer ONLY from the context below
        - If answer is not in context say: I don't know based on provided documents
        - Be direct and concise
        - Do not make up information

        Context:
        {context} """), ("human", "{question}")
    ])

#this function accepts the retrieved documents and formats them in a way that LLM can understand 
#and use it to answer the question. It also adds some metadata like source, page number and
#chunk index to each chunk for better traceability of answer.
def format_docs(docs: List[Document]) -> str:
    formatted = [] #creates empty list 
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"[Chunk {i+1} | Source: {source} | Page: {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)

#this function creates complete rag pipeline 
def build_rag_chain(retriever):
    llm = ChatGroq(model=LLM_MODEL, temperature=0)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT| llm| StrOutputParser()
    )
    return chain

#this function gives answer , retrived chunks and some meta data to dubuging the rag .
def query_with_sources(retriever, question: str) -> Dict[str, Any]:
    # get retrieved chunks
    docs = retriever.invoke(question)

    # format context
    context = format_docs(docs)

    # get answer from LLM
    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    prompt = RAG_PROMPT.format_messages(
        context=context,
        question=question
    )
    response = llm.invoke(prompt)

    # build sources list
    sources = []
    for doc in docs:
        sources.append({
            "content": doc.page_content[:300],
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "chunk_index": doc.metadata.get("chunk_index", "?"),
            "strategy": doc.metadata.get("chunk_strategy", "?"),
        })

    return {
        "question": question,
        "answer": response.content,
        "sources": sources,
        "num_chunks_retrieved": len(docs)
    }





"""
from src.vectorstore import load_vectorstore
from src.retriever import get_retriever
from src.rag_pipeline import query_with_sources

vs = load_vectorstore()
retriever = get_retriever('basic', vs)

result = query_with_sources(retriever, 'What is artificial intelligence?')
print('Answer:', result['answer'])
print('Chunks used:', result['num_chunks_retrieved'])
for s in result['sources']:
    print('Page:', s['page'])
"""