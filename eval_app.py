import streamlit as st
import pandas as pd
import time
from utils.rag_evaluator_runner import RAGEvaluationRunner
from chatbot.init_chatbot import create_groq_chatbot
from chatbot.inferences.inference import infer_response
from utils.hybrid_retriever import HybridRetriever

st.set_page_config(page_title="RAG Evaluator", layout="wide")
st.title("RAG System Evaluation Dashboard")

# Initialize session states
if 'llm' not in st.session_state:
    with st.spinner("Initializing LLM..."):
        st.session_state.llm = create_groq_chatbot()

if 'evaluator' not in st.session_state:
    st.session_state.evaluator = RAGEvaluationRunner(st.session_state.llm)

if 'retriever' not in st.session_state:
    with st.spinner("Initializing retriever..."):
        st.session_state.retriever = HybridRetriever(st.session_state.llm, k=10)

if 'retrieval_history' not in st.session_state:
    st.session_state.retrieval_history = []

if 'test_queries' not in st.session_state:
    st.session_state.test_queries = [
        "What are the main topics discussed in recent podcasts?",
        "Can you summarize the conversation about technology trends?",
        "What did the speakers say about artificial intelligence?",
        "What were the key points about climate change?",
        "How do the speakers feel about social media?"
    ]

# Sidebar
st.sidebar.header("Evaluation Settings")
k_value = st.sidebar.slider("Number of retrieved documents (k)", 3, 30, 10)

# Tabs
tab1, tab2, tab3 = st.tabs(["Single Query Evaluation", "Batch Evaluation", "Metrics Dashboard"])

# Single Query Evaluation
with tab1:
    st.header("Evaluate a Single Query")
    
    # Query input
    query = st.text_area("Enter your query:", height=100)
    
    # Optional reference answer
    reference = st.text_area("Reference answer (optional):", height=100)
    
    if st.button("Evaluate Query"):
        if query:
            with st.spinner("Processing query..."):
                # Get response and retrieved docs
                st.session_state.retriever = HybridRetriever(st.session_state.llm, k=k_value)
                
                # Get retrieved documents
                retrieved_docs = st.session_state.retriever.get_relevant_documents(query)
                
                # Generate response
                response = infer_response(query, k=k_value)
                
                # Evaluate
                eval_results = st.session_state.evaluator.evaluate_qa_pair(
                    query, response, retrieved_docs, reference
                )
                
                # Display results
                st.subheader("Response")
                st.write(response)
                
                st.subheader("Retrieved Documents")
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"Document {i+1}"):
                        st.write(doc.page_content)
                
                st.subheader("Evaluation Metrics")
                # Filter out non-numeric and metadata fields
                metrics_df = pd.DataFrame({k: [v] for k, v in eval_results.items() 
                                        if isinstance(v, (int, float)) and k not in ['query', 'response', 'timestamp']})
                st.dataframe(metrics_df.T.rename(columns={0: 'Score'}))
        else:
            st.error("Please enter a query")

# Batch Evaluation
with tab2:
    st.header("Batch Evaluation")
    
    # Test queries
    st.subheader("Test Queries")
    queries = st.text_area("Enter test queries (one per line):", 
                           value="\n".join(st.session_state.test_queries), 
                           height=200)
    
    # Update test queries
    if queries != "\n".join(st.session_state.test_queries):
        st.session_state.test_queries = [q.strip() for q in queries.split("\n") if q.strip()]
    
    if st.button("Run Batch Evaluation"):
        if st.session_state.test_queries:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            qa_pairs = []
            
            for i, query in enumerate(st.session_state.test_queries):
                status_text.text(f"Processing query {i+1}/{len(st.session_state.test_queries)}: {query}")
                
                # Get documents and response
                retrieved_docs = st.session_state.retriever.get_relevant_documents(query)
                response = infer_response(query, k=k_value)
                
                qa_pairs.append({
                    'query': query,
                    'response': response,
                    'retrieved_docs': retrieved_docs
                })
                
                # Update progress
                progress_bar.progress((i + 1) / len(st.session_state.test_queries))
                time.sleep(0.5)  # Add small delay to see progress
            
            # Evaluate batch
            results = st.session_state.evaluator.batch_evaluate(qa_pairs)
            
            # Show results
            status_text.text("Evaluation complete!")
            
            # Display summary metrics
            st.subheader("Summary Metrics")
            avg_metrics = st.session_state.evaluator.get_average_metrics()
            metrics_df = pd.DataFrame({k: [v] for k, v in avg_metrics.items()})
            st.dataframe(metrics_df.T.rename(columns={0: 'Average Score'}))
            
            # Visualization
            st.subheader("Metrics Visualization")
            fig = st.session_state.evaluator.visualize_metrics()
            st.pyplot(fig)
            
            # Detailed results
            st.subheader("Detailed Results")
            for i, result in enumerate(results):
                with st.expander(f"Query {i+1}: {result['query'][:50]}..."):
                    st.write("**Response:**", result['response'])
                    st.write("**Metrics:**")
                    metrics = {k: v for k, v in result.items() 
                              if isinstance(v, (int, float)) and k not in ['query', 'response', 'timestamp']}
                    st.dataframe(pd.DataFrame(metrics, index=[0]).T.rename(columns={0: 'Score'}))
        else:
            st.error("Please add at least one test query")

# Metrics Dashboard
with tab3:
    st.header("Metrics Dashboard")
    
    if st.session_state.evaluator.evaluation_history:
        # Summary metrics
        st.subheader("Overall Performance")
        avg_metrics = st.session_state.evaluator.get_average_metrics()
        
        # Create three columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            retrieval_score = avg_metrics.get('retrieval_avg_semantic_similarity', 0)
            st.metric("Retrieval Quality", f"{retrieval_score:.2f}")
        
        with col2:
            generation_score = avg_metrics.get('generation_factuality_score', 0)
            st.metric("Response Factuality", f"{generation_score:.2f}")
        
        with col3:
            context_score = avg_metrics.get('context_utilization', 0)
            st.metric("Context Utilization", f"{context_score:.2f}")
        
        # Visualization
        st.subheader("Metrics Visualization")
        fig = st.session_state.evaluator.visualize_metrics()
        st.pyplot(fig)
        
        # Export options
        st.subheader("Export Results")
        export_format = st.radio("Export Format", ["CSV", "JSON"])
        if st.button("Export"):
            results = st.session_state.evaluator.export_results(format=export_format.lower())
            st.download_button(
                label=f"Download {export_format}",
                data=results,
                file_name=f"rag_evaluation_results.{export_format.lower()}",
                mime="text/plain"
            )
    else:
        st.info("No evaluation data available. Run some evaluations to see metrics.")

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
This dashboard helps evaluate your RAG system performance.
- **Single Query**: Test individual queries and see detailed metrics
- **Batch Evaluation**: Run tests on multiple queries
- **Metrics Dashboard**: View overall performance metrics
""")

# Add dependencies
if st.sidebar.checkbox("Show Required Dependencies"):
    st.sidebar.code("""
    pip install rouge-score
    pip install matplotlib
    pip install sklearn
    pip install pandas
    pip install rank_bm25>=0.2.2
    """) 