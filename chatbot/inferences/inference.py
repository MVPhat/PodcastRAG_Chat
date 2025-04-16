from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from utils.hybrid_retriever import HybridRetriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

def init_hyperparams(k=20):
    # Query rewriting prompt
    query_rewriting_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query rewriter. Your task is to rewrite the user's question to be more specific and clear, 
        taking into account the chat history. Make sure to maintain the original intent while making it more precise.
        If the question is already clear and specific, return it as is."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Contextualization prompt
    system_prompt_contextualization = """
    You are an expert at understanding the context of conversations about podcasts. 
    Your task is to analyze the chat history and current question to determine the most relevant context.
    Consider:
    1. The topic being discussed
    2. Any specific speakers or episodes mentioned
    3. The user's information needs
    4. Previous context that might be relevant
    
    If you cannot find relevant context, acknowledge this and ask for clarification.
    """

    prompt_contextualization = ChatPromptTemplate.from_messages([
        ("system", system_prompt_contextualization),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Initialize hybrid retriever with query expansion
    if 'retriever' not in st.session_state:
        st.session_state.retriever = HybridRetriever(st.session_state.llm, k=k)

    # Create query rewriter chain
    query_rewriter = query_rewriting_prompt | st.session_state.llm

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, 
        st.session_state.retriever.as_retriever(), 
        prompt_contextualization
    )

    # Enhanced QnA prompt
    QnA_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant specialized in podcast content. Your responses should be:
        1. Direct and concise
        2. Well-structured with clear points
        3. Based on the provided context
        4. Natural and conversational
        5. Include relevant details from the context when appropriate
        
        If the context doesn't contain relevant information, acknowledge this and suggest alternative topics or ask for clarification."""),
        ("system", "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    QnA_chain = create_stuff_documents_chain(st.session_state.llm, QnA_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, QnA_chain)

    return rag_chain, query_rewriter

def infer_response(input_text=' ', k=20):
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain, st.session_state.query_rewriter = init_hyperparams(k=k)
    
    # Rewrite the query for better retrieval
    rewritten_query = st.session_state.query_rewriter.invoke({
        "input": input_text,
        "chat_history": st.session_state.retrieval_history
    }).content
    
    # Get response using the rewritten query
    response = st.session_state.rag_chain.invoke({
        "input": rewritten_query,
        "chat_history": st.session_state.retrieval_history
    })['answer']
    
    # Update chat history
    st.session_state.retrieval_history.extend([
        HumanMessage(content=input_text),
        AIMessage(content=response)
    ])
    
    return response