import streamlit as st
from config import init_page
from asl_engine import FingerspellingEngine
from ui_components import render_reference_sidebar, text_to_speech

init_page()
render_reference_sidebar()

st.title("🔠 ASL Fingerspelling Pro")

# Display the current "Sentence"
if 'history' not in st.session_state: st.session_state.history = []
current_word = "".join(st.session_state.history)

with st.container(border=True):
    st.markdown(f"## {current_word if current_word else 'Start signing...'}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear"): st.session_state.history = []
    with col2:
        if st.button("Backspace"): st.session_state.history.pop() if st.session_state.history else None
    with col3:
        if st.button("🔊 Speak"): text_to_speech(current_word)

