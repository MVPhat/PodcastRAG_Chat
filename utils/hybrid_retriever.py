from typing import List, Dict, Any
from langchain.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.schema import Document
from langchain_chroma import Chroma
from utils.init_embedding_model import init_embedding_model
from utils.query_expansion import QueryExpander

class HybridRetriever:
    def __init__(self, llm, k: int = 20):
        self.k = k
        self.embedding_model = init_embedding_model()
        self.query_expander = QueryExpander(llm)
        self._setup_retrievers()
        
    def _setup_retrievers(self):
        # Dense retriever (ChromaDB)
        collection_name = "podcast_transcripts"
        import os
        db_path = os.path.abspath("./custom_data/chroma_db")
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_function=self.embedding_model
        )
        self.dense_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k}
        )
        
        # Sparse retriever (BM25)
        # Get all documents from ChromaDB
        docs = self.vectorstore.get()
        documents = [Document(page_content=doc["text"]) for doc in docs]
        self.sparse_retriever = BM25Retriever.from_documents(documents)
        self.sparse_retriever.k = self.k
        
        # Create ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[0.5, 0.5]
        )
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        # Expand the query
        expanded_queries = self.query_expander.expand_query(query)
        
        # Get documents for each expanded query
        all_docs = []
        for q in expanded_queries:
            docs = self.ensemble_retriever.get_relevant_documents(q)
            all_docs.extend(docs)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        
        # Return top k unique documents
        return unique_docs[:self.k]
    
    def as_retriever(self):
        return self.ensemble_retriever 