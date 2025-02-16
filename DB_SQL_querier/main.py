from langchain_helper import DbConvo

import streamlit as st

st.title("Atliq T shirts: Database Q&A 👕")

question = st.text_input("Question: ")

if question:
    db_convo =DbConvo()
    answer = db_convo.get_answer(question)
    st.header("Answer: ")
    st.write(answer)

