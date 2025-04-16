import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.init_embedding_model import init_embedding_model
from utils.rag_evaluation import RAGEvaluator
from langchain.schema import Document
from typing import List, Dict, Any

class RAGEvaluationRunner:
    def __init__(self, llm=None):
        if llm is None:
            llm = st.session_state.llm
        self.embedding_model = init_embedding_model()
        self.evaluator = RAGEvaluator(self.embedding_model, llm)
        self.evaluation_history = []
        
    def evaluate_qa_pair(self, query: str, response: str, retrieved_docs: List[Document], 
                         reference: str = None, relevant_docs: List[Document] = None):
        """
        Evaluate a single question-answer pair
        """
        # Get comprehensive evaluation metrics
        eval_results = self.evaluator.evaluate_end_to_end(
            query, response, retrieved_docs, reference, relevant_docs
        )
        
        # Get LLM-based evaluation
        llm_eval = self.evaluator.llm_evaluator(query, response, retrieved_docs)
        if not isinstance(llm_eval, dict) or "error" in llm_eval:
            llm_eval = {"relevance": 0, "factuality": 0, "completeness": 0, "coherence": 0, "conciseness": 0}
        
        # Combine all evaluation results
        all_results = {
            "query": query,
            "response": response,
            "timestamp": pd.Timestamp.now(),
            **eval_results,
            **{f"llm_{k}": v for k, v in llm_eval.items() if k != "error"}
        }
        
        # Add to history
        self.evaluation_history.append(all_results)
        return all_results
        
    def batch_evaluate(self, qa_pairs: List[Dict[str, Any]]):
        """
        Evaluate a batch of QA pairs
        Each pair should be a dict with keys:
            - query: str
            - response: str
            - retrieved_docs: List[Document]
            - reference (optional): str
            - relevant_docs (optional): List[Document]
        """
        results = []
        for pair in qa_pairs:
            result = self.evaluate_qa_pair(
                pair['query'], 
                pair['response'], 
                pair['retrieved_docs'],
                pair.get('reference'),
                pair.get('relevant_docs')
            )
            results.append(result)
        return results
    
    def visualize_metrics(self, metrics_to_show=None):
        """
        Visualize the evaluation metrics across the history
        """
        if not self.evaluation_history:
            return "No evaluation data available"
        
        # Convert history to DataFrame
        df = pd.DataFrame(self.evaluation_history)
        
        # Set default metrics to visualize if none provided
        if metrics_to_show is None:
            metrics_to_show = [
                'retrieval_avg_semantic_similarity', 
                'retrieval_diversity',
                'generation_factuality_score', 
                'context_utilization',
                'llm_relevance', 
                'llm_factuality'
            ]
        
        # Filter to only include metrics that exist in the data
        available_metrics = [m for m in metrics_to_show if m in df.columns]
        
        if not available_metrics:
            return "None of the specified metrics are available in the evaluation data"
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        df[available_metrics].mean().plot(kind='bar', ax=ax)
        plt.title('Average RAG Evaluation Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def get_average_metrics(self):
        """
        Return the average metrics across all evaluations
        """
        if not self.evaluation_history:
            return {}
        
        df = pd.DataFrame(self.evaluation_history)
        numeric_cols = df.select_dtypes(include=['number']).columns
        return df[numeric_cols].mean().to_dict()
    
    def export_results(self, format='csv'):
        """
        Export evaluation results to CSV or JSON
        """
        if not self.evaluation_history:
            return "No evaluation data available to export"
        
        df = pd.DataFrame(self.evaluation_history)
        
        if format.lower() == 'csv':
            return df.to_csv(index=False)
        elif format.lower() == 'json':
            return df.to_json(orient='records')
        else:
            return "Unsupported format. Use 'csv' or 'json'."