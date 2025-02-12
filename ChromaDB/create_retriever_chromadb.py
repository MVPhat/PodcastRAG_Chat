from langchain_chroma import Chroma
from utils.init_embedding_model import init_embedding_model
import streamlit as st

def create_retriever_chromadb(k=20):
    # print('prev k:', st.session_state.k, 'new k:', k)
    if st.session_state.k == -1:
        embedding_model = init_embedding_model()
        collection_name = "podcast_transcripts"
        import os
        db_path = os.path.abspath("./custom_data/chroma_db")
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_function=embedding_model
        )
        print("Vector store created and persisted to:", db_path)
        print("Vector store collection name:", vectorstore._collection_name)
        print("Total Documents in Chroma:", vectorstore._collection.count())

        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    else:
        retriever = st.session_state.retriever.vectorstore.as_retriever(search_kwargs={"k": k})

    return retriever

# create_retriever_chromadb(20)