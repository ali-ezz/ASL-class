import streamlit as st

def render_reference_sidebar():
    with st.sidebar:
        st.header("📖 ASL Reference")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/American_Sign_Language_Alphabet.svg/600px-American_Sign_Language_Alphabet.svg.png")
        st.info("Tip: Hold your hand steady for 1 second to capture a letter.")

def text_to_speech(text):
    """Uses browser-native Speech Synthesis via JS"""
    if text:
        js_code = f"window.speechSynthesis.speak(new SpeechSynthesisUtterance('{text}'));"
        st.components.v1.html(f"<script>{js_code}</script>", height=0)