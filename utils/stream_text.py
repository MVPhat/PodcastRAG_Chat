import streamlit as st
import time

def stream_text(text, style, delay=0.05):
    placeholder = st.empty()
    streamed_text = ""
    for char in text:
        streamed_text += char
        formatted_content = style.format(content=streamed_text)  # Apply style
        placeholder.markdown(formatted_content, unsafe_allow_html=True)
        time.sleep(delay)