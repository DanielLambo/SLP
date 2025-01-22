import streamlit as st
from dotenv import load_dotenv
import os

st.title("Personalized learning full-path recommendation model based on LSTM neural networks")
st.subheader("By: Daniel Lambo")
st.subheader("Mentor: Dr Yujian Fu")
st.divider()



citation_one = '''Jayashri Bagade, Poonam Chaudhari, Poonam Girish Fegade, Ranjit M.
Gawande, Prachi P. Vast, Dipika R. Birari (2024). Adaptive Learning Technologies for Personalized
Research Assistance in Libraries. Library Progress International, 44(1), 242-258.'''
st.divider(
)
st.subheader("Works Cited:")
st.caption(citation_one)