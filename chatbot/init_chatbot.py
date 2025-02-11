from langchain_groq import ChatGroq

def create_groq_chatbot(chatbot_model='llama3-70b-8192'):
    llm = ChatGroq(model_name=chatbot_model)
    return llm