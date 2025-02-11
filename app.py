import streamlit as st
from chatbot.chatAPI import llm_response
from chatbot.chatbotUI_custom import user_style, bot_style
from utils.stream_text import stream_text

st.set_page_config(page_title="ChatAPI", page_icon=":shark:", layout="wide")

st.markdown(
    """<h1 style="text-align: center; font-family: 'Roboto', sans-serif; color: #1167bd;">
    AAAAAAAAAA
    </h1>""",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.chat_input(placeholder="Type a message...")
if prompt:
    response = llm_response(prompt)
    st.session_state.history.append((prompt, response))

activate_stream_rendering = st.toggle("Stream rendering", False)

for old_prompt, old_response in st.session_state.history:
    if activate_stream_rendering:
        stream_text(old_prompt, user_style, 0)
        stream_text(old_response, bot_style, 0.001)
    else:
        with st.chat_message(name="user", avatar="🤔"):
            st.write(old_prompt)
        with st.chat_message(name="bot", avatar="🤖"):
            st.write(old_response)