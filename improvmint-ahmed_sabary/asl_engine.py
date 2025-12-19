from collections import deque
import streamlit as st

class FingerspellingEngine:
    def __init__(self, stability_frames=10):
        self.buffer = deque(maxlen=stability_frames)
        self.last_finalized_letter = None

    def process_prediction(self, letter):
        """Only 'captures' a letter if it stays the same for N frames."""
        self.buffer.append(letter)
        
        # Stability Check
        if len(self.buffer) == self.buffer.maxlen and len(set(self.buffer)) == 1:
            stable_letter = self.buffer[0]
            if stable_letter != self.last_finalized_letter:
                self.last_finalized_letter = stable_letter
                return stable_letter
        return None

    def commit_to_history(self, letter):
        if 'history' not in st.session_state:
            st.session_state.history = []
        if letter:
            st.session_state.history.append(letter)