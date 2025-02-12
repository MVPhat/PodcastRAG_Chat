from langchain_chroma import Chroma
from ..utils.init_embedding_model import init_embedding_model

def create_retriever_chromadb(k=20):
    embedding_model = init_embedding_model()

    collection_name = "podcast_transcripts"
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory="../custom_data/chroma_db",
        embedding_function=embedding_model
    )

    print("Total Documents in Chroma:", vectorstore._collection.count())

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return retriever