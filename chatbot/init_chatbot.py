from dotenv import load_dotenv
from langchain_groq import ChatGroq
import langchain

load_dotenv()

def create_groq_chatbot(chatbot_model='llama3-70b-8192'):
    langchain.verbose = False
    langchain.debug = False
    langchain.llm_cache = False
    llm = ChatGroq(model_name=chatbot_model)
    return llm