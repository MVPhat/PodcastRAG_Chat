from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
# llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-qwen-32b", groq_api_key=GROQ_API_KEY)

from langchain_huggingface import HuggingFaceEmbeddings
import torch

def init_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hf

from langchain_chroma import Chroma
collection_name = "podcast_transcripts"
db_path = "/kaggle/input/rag-data-vectordb-chroma/chroma_db"
vectorstore = Chroma(
    collection_name=collection_name,
    persist_directory=db_path,
    embedding_function=init_embedding_model()
)
print("Vector store created and persisted to:", db_path)
print("Total Documents in Chroma:", vectorstore._collection.count())

retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import pandas as pd
from tqdm import tqdm
df = pd.read_csv('/kaggle/input/acquired-podcast-transcripts-and-rag-evaluation/acquired-qa-evaluation.csv', encoding='ISO-8859-1')

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Your answer must be concise, short and understandable. Using the following context to answer the user's question."),
    ("system", "Context: {context}"),  # Must have this for document retrieval
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)  # No history-aware retriever

ai_answers = []
i = 0
for question in tqdm(df['question'], total=len(df['question'])):
    response = rag_chain.invoke({"input": question})['answer']
    ai_answers.append(response)

# Save responses to DataFrame
df["ai_answer"] = ai_answers

# Save the updated DataFrame to CSV
output_path = "/kaggle/working/rag_evaluation_results.csv"
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"Results saved to {output_path}")