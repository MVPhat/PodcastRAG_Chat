# import os
# import streamlit as st
# from dotenv import load_dotenv
# import langchain
# from langchain_core.output_parsers import StrOutputParser
# from langchain_groq import ChatGroq
# from ..init_chatbot import create_groq_chatbot

# load_dotenv()
# langchain.verbose = False
# langchain.debug = False
# langchain.llm_cache = False

# if "llm" not in st.session_state:
#     print('init first chatbot')
#     chatbot_model = 'llama3-70b-8192'
#     llm = create_groq_chatbot(chatbot_model)
#     st.session_state.llm = llm

# # def llm_response(prompt):
# #     output_parser = StrOutputParser()
# #     response = st.session_state.llm.invoke(prompt)
# #     return response.content