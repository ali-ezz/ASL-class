import streamlit as st

# ASL Configuration
IMG_SIZE = 224
REPO_ID = "ALI-ezz/ASL_CORP"

# SMART CONFUSION LOGIC: Optimized for static alphabet letters
# If confidence is low, the system will suggest these alternatives
CONFUSION_PAIRS = {
    'M': ['N', 'S', 'A'],
    'N': ['M', 'S', 'A'],
    'S': ['M', 'N', 'A', 'E'],
    'A': ['S', 'E', 'M', 'N'],
    'E': ['A', 'S'],
    'K': ['D', 'V', 'P'],
    'U': ['V', 'R'],
    'I': ['J', 'Y'],
}

def init_page():
    st.set_page_config(page_title="ASL Fingerspelling Pro", layout="wide")