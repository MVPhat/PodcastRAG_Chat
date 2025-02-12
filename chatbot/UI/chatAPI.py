import os
import streamlit as st
from dotenv import load_dotenv
import langchain
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

if "llm" not in st.session_state:
    chatbot_model = 'llama3-70b-8192'

    llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name=chatbot_model)
    st.session_state.llm = llm

def llm_response(prompt):
    output_parser = StrOutputParser()
    response = st.session_state.llm.invoke(prompt)
    return response.content