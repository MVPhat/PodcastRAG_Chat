import streamlit as st
from chatbot.UI.chatbotUI_custom import user_style, bot_style
from chatbot.init_chatbot import create_groq_chatbot
from ChromaDB.create_retriever_chromadb import create_retriever_chromadb
from chatbot.inferences.inference import infer_response, init_hyperparams
from utils.stream_text import stream_text
import time

st.set_page_config(page_title="PodcastChat", page_icon="https://png.pngtree.com/png-vector/20220518/ourmid/pngtree-podcast-icon-color-flat-outline-png-image_4687717.png", layout="centered")
st.markdown(
    """<h1 style="text-align: center; font-family: 'Roboto', sans-serif; color: #343333;">
    Podcast Chat Assistant
    </h1>""",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

if "retrieval_history" not in st.session_state:
    st.session_state.retrieval_history = []

if "k" not in st.session_state:
    st.session_state.k = -1

if 'change_k' not in st.session_state:
    st.session_state.change_k = False

col1, col2 = st.columns([1, 4])
with col1:
    activate_stream_rendering = st.toggle("Stream rendering", True)
with col2:
    k = st.slider("Number of relevant documents for RAG:", 1, 100, 10)

if 'llm' not in st.session_state:
    st.session_state.llm = create_groq_chatbot()

with st.spinner("Initialize LLM and RAG data... This may take a couple of minutes for the first run!"):
    if k != st.session_state.k:
        time.sleep(0.001)
        st.session_state.retriever = create_retriever_chromadb(int(k))
        st.session_state.k = k
        st.session_state.change_k = True

    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = init_hyperparams(k=int(k))

prompt = st.chat_input(placeholder="Type a message...")
if prompt:
    response = infer_response(input_text=prompt, k=int(k))
    st.session_state.history.append((prompt, response))

for i, (old_prompt, old_response) in enumerate(st.session_state.history):
    if i < len(st.session_state.history) - 1:  # Render all previous messages instantly
        stream_text(old_prompt, user_style, 0)
        stream_text(old_response, bot_style, 0)
    else:
        stream_text(old_prompt, user_style, 0)
        if activate_stream_rendering: # Stream only the latest message
            if st.session_state.change_k:
                stream_text(old_response, bot_style, 0)
                st.session_state.change_k = False
            else:
                stream_text(old_response, bot_style, 0.01)
        else:
            stream_text(old_response, bot_style, 0)