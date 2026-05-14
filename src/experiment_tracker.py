from src.vectorstore import load_vectorstore
from src.document_processor import load_documents, recursive_chunking
from src.retriever import get_retriever

vs = load_vectorstore()
docs = load_documents()
chunks = recursive_chunking(docs)
question = 'What is artificial intelligence?'

print('Testing reranker_vector...')
r = get_retriever('reranker_vector', vs, all_chunks=chunks)
results = r.invoke(question)
print('Chunks returned:', len(results))
print('First chunk:', results[0].page_content[:150])
print('Page:', results[0].metadata['page'])