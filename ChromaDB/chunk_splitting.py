from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import os

def load_texts():
    files = os.listdir('./')
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