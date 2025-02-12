from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from ChromaDB.create_retriever_chromadb import create_retriever_chromadb
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

def init_hyperparams(k=20):
    system_prompt_contextualization = """
    You are allowed to read the chat history between user and your responses,
    which might relate to reference contexts about podcasts between speakers and audiences from several topics.
    If you cannot retrieve any related contexts, you should response the apology to the user.
    """

    prompt_contextualization = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_contextualization),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # if 'retriever' not in st.session_state:
    #     retriever = create_retriever_chromadb(k)
    #     st.session_state.retriever = retriever

    history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, st.session_state.retriever, prompt_contextualization
    )

    QnA_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are currently an AI assistant who listened to several podcasts with variant topics. You must answer directly without any pharses to show that you are following/according the RAG content, just simply provide the most relevant and well-structured response. The following contexts may help you to answer the user's question."),
        ("system", "Context: {context}"), # must have {context} for create_stuff_documents_chain
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    QnA_chain = create_stuff_documents_chain(st.session_state.llm, QnA_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, QnA_chain)

    return rag_chain

def infer_response(input_text=' ', k=20):
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = init_hyperparams(k=k)
    response = st.session_state.rag_chain.invoke({"input": input_text, "chat_history": st.session_state.retrieval_history})['answer']
    st.session_state.retrieval_history.extend([
        HumanMessage(content=input_text),
        AIMessage(content=response)
    ])
    return response