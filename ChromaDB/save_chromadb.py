# ======================================================
# Chunking and Embedding

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import os

def load_texts():
    files = os.listdir('PATH/TO/DATASET')
    documents = []
    for file in files:
        if file.endswith('.pdf'):
            print("Loading ", file)
            loader = PyPDFLoader(file)
        elif file.endswith('.docx'):
            print("Loading ", file)
            loader = Docx2txtLoader(file)
        else:
            continue
        documents.extend(loader.load())
    return documents

documents = load_texts()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
print("Number of documents:", len(documents))
splits = text_splitter.split_documents(documents)
print("Number of chunks:", len(splits))


# ======================================================


from langchain_chroma import Chroma
from utils.init_embedding_model import init_embedding_model
import os

embedding_model = init_embedding_model()
db_path = os.path.abspath("PATH/TO/VECTOR_DB_STORAGE")
collection_name = "podcast_transcripts"
vectorstore = Chroma.from_documents(
    collection_name=collection_name,
    documents=splits,
    embedding=embedding_model,
    persist_directory=db_path
)
print("Total Documents in Chroma:", vectorstore._collection.count())
print("Vector store created and persisted to '../custom_data/chroma_db'")