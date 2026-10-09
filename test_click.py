import streamlit as st

st.title("Interaction Test")

name = st.text_input("Enter your name")

if st.button("Test Click"):
    st.success(f"Hello, {name}!")