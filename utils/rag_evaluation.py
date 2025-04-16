from typing import List, Dict, Any, Tuple
import numpy as np
from langchain.schema import Document
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
import time
from langchain_community.retrievers import SVMRetriever

class RAGEvaluator:
    def __init__(self, embedding_model, llm):
        self.embedding_model = embedding_model
        self.llm = llm
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def evaluate_retrieval(self, query: str, retrieved_docs: List[Document], 
                           relevant_docs: List[Document] = None) -> Dict[str, Any]:
        """
        Evaluate retrieval performance.
        If relevant_docs is not provided, relevance is estimated using similarity to the query.
        """
        results = {}
        
        # Measure retrieval latency
        start_time = time.time()
        # For measurement purposes only - already have retrieved docs
        query_embedding = self.embedding_model.embed_query(query)
        results['latency'] = time.time() - start_time
        
        # If we have ground truth relevant documents
        if relevant_docs:
            # Calculate precision, recall, and F1 score
            relevant_ids = set([doc.metadata.get('id', i) for i, doc in enumerate(relevant_docs)])
            retrieved_ids = set([doc.metadata.get('id', i) for i, doc in enumerate(retrieved_docs)])
            
            tp = len(relevant_ids.intersection(retrieved_ids))
            fp = len(retrieved_ids - relevant_ids)
            fn = len(relevant_ids - retrieved_ids)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results['precision'] = precision
            results['recall'] = recall
            results['f1_score'] = f1
        
        # Semantic similarity between query and retrieved documents
        query_embedding = self.embedding_model.embed_query(query)
        doc_embeddings = [self.embedding_model.embed_document(doc.page_content) for doc in retrieved_docs]
        
        if doc_embeddings:
            similarities = [cosine_similarity([query_embedding], [doc_emb])[0][0] for doc_emb in doc_embeddings]
            results['avg_semantic_similarity'] = np.mean(similarities)
            results['max_semantic_similarity'] = np.max(similarities)
        else:
            results['avg_semantic_similarity'] = 0
            results['max_semantic_similarity'] = 0
            
        # Document diversity (average pairwise cosine distance between documents)
        if len(doc_embeddings) > 1:
            pairwise_distances = []
            for i in range(len(doc_embeddings)):
                for j in range(i+1, len(doc_embeddings)):
                    distance = 1 - cosine_similarity([doc_embeddings[i]], [doc_embeddings[j]])[0][0]
                    pairwise_distances.append(distance)
            results['diversity'] = np.mean(pairwise_distances)
        else:
            results['diversity'] = 0
            
        return results
    
    def evaluate_generation(self, response: str, reference: str = None, 
                           retrieved_docs: List[Document] = None) -> Dict[str, Any]:
        """
        Evaluate generation quality.
        If reference is not provided, factuality is estimated using retrieval context.
        """
        results = {}
        
        # Calculate lexical metrics if reference is available
        if reference:
            rouge_scores = self.rouge_scorer.score(reference, response)
            results['rouge1'] = rouge_scores['rouge1'].fmeasure
            results['rouge2'] = rouge_scores['rouge2'].fmeasure
            results['rougeL'] = rouge_scores['rougeL'].fmeasure
        
        # Estimate factuality/faithfulness to retrieved documents
        if retrieved_docs:
            context = " ".join([doc.page_content for doc in retrieved_docs])
            factuality_prompt = f"""
            Given the following context and response, rate the factual accuracy of the response on a scale of 1-5:
            1: Completely contradicts the context
            2: Mostly contradicts the context with a few accurate points
            3: Partially accurate but contains significant errors
            4: Mostly accurate with minor errors or omissions
            5: Completely accurate and faithful to the context
            
            Context: {context}
            
            Response: {response}
            
            Rate (1-5):
            """
            try:
                factuality_score = int(self.llm.invoke(factuality_prompt).content.strip())
                results['factuality_score'] = min(max(factuality_score, 1), 5)  # Ensure between 1-5
            except:
                results['factuality_score'] = None
                
        # Evaluate response conciseness
        results['response_length'] = len(response.split())
        
        return results
    
    def evaluate_end_to_end(self, query: str, response: str, retrieved_docs: List[Document], 
                           reference: str = None, relevant_docs: List[Document] = None) -> Dict[str, Any]:
        """
        Perform comprehensive end-to-end RAG system evaluation
        """
        results = {}
        
        # Get retrieval metrics
        retrieval_metrics = self.evaluate_retrieval(query, retrieved_docs, relevant_docs)
        results.update({f"retrieval_{k}": v for k, v in retrieval_metrics.items()})
        
        # Get generation metrics
        generation_metrics = self.evaluate_generation(response, reference, retrieved_docs)
        results.update({f"generation_{k}": v for k, v in generation_metrics.items()})
        
        # Calculate RAG Triad metrics
        if reference:
            # Relevance: How well the retrieved docs match the query intent
            results['relevance'] = results.get('retrieval_avg_semantic_similarity', 0)
            
            # Groundedness: How factual/faithful the response is to retrieved docs
            results['groundedness'] = results.get('generation_factuality_score', 0) / 5 if results.get('generation_factuality_score') else None
            
            # Helpfulness: How well the response addresses the query
            response_embedding = self.embedding_model.embed_document(response)
            query_embedding = self.embedding_model.embed_query(query)
            results['helpfulness'] = cosine_similarity([query_embedding], [response_embedding])[0][0]
            
            # RAG Score: Combined metric
            if results['groundedness']:
                results['rag_score'] = (results['relevance'] + results['groundedness'] + results['helpfulness']) / 3
            else:
                results['rag_score'] = (results['relevance'] + results['helpfulness']) / 2
        
        # Context utilization
        if retrieved_docs:
            llm_eval_prompt = f"""
            On a scale of 1-5, rate how well the response utilizes the provided context:
            1: Ignores context completely
            2: Uses minimal context information
            3: Partially uses context
            4: Uses most relevant parts of context
            5: Optimally uses context
            
            Context: {"".join([doc.page_content for doc in retrieved_docs])}
            
            Response: {response}
            
            Rate (1-5):
            """
            try:
                context_utilization = int(self.llm.invoke(llm_eval_prompt).content.strip())
                results['context_utilization'] = min(max(context_utilization, 1), 5)
            except:
                results['context_utilization'] = None
        
        return results
    
    def llm_evaluator(self, query: str, response: str, retrieved_docs: List[Document]) -> Dict[str, float]:
        """
        Use LLM to evaluate RAG system output based on multiple criteria
        """
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        eval_prompt = f"""
        You are an expert evaluator of RAG (Retrieval Augmented Generation) systems. Please evaluate the following response to a query based on the provided retrieved context.
        
        Query: {query}
        
        Retrieved Context: {context}
        
        Response: {response}
        
        Please evaluate the response on the following criteria on a scale of 1-5:
        1. Relevance: How relevant is the response to the query? (1-5)
        2. Factuality: How factually accurate is the response based on the provided context? (1-5)
        3. Completeness: How completely does the response address all aspects of the query? (1-5)
        4. Coherence: How coherent and well-structured is the response? (1-5)
        5. Conciseness: How concise and to-the-point is the response? (1-5)
        
        Return your evaluation as a JSON object with these 5 scores.
        """
        try:
            eval_response = self.llm.invoke(eval_prompt).content.strip()
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', eval_response, re.DOTALL)
            if json_match:
                eval_scores = json.loads(json_match.group(0))
                return eval_scores
            else:
                return {"error": "Could not parse LLM evaluation output"}
        except Exception as e:
            return {"error": str(e)} 