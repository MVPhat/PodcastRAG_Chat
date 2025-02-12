from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

contextualize_q_system_prompt = """
You are allowed to read the chat history between user and your responses,
which might relate to reference contexts about podcasts between speakers and audiences from several topics.
If you cannot retrieve any relevant contexts, you should response the apology to the user.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)




from langchain_chroma import Chroma

collection_name = "podcast_transcripts"
vectorstore = Chroma(
    collection_name=collection_name,
    persist_directory='/kaggle/input/rag-data-vectordb-chroma/chroma_db',
    embedding_function=hf
)

print("Total Documents in Chroma:", vectorstore._collection.count())




from langchain.chains.retrieval import create_retrieval_chain

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"), # must have for create_stuff_documents_chain
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)