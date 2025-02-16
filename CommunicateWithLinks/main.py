import streamlit as st
from langchain_helper import WebConvo

st.title("News Research ")

st.sidebar.title("News articles URL's")



no_of_urls = st.sidebar.text_input("No of urls?")

url_list = []
if no_of_urls:
    query = st.text_input("What do you wish to find out from these URL's?")
    try:
        for i in range(int(no_of_urls)):
            url = st.sidebar.text_input(f"URL {i + 1}")
            url_list.append(url)

    except ValueError as e:
        st.sidebar.write(f"{no_of_urls} is not a number")


get_result_clicked = st.sidebar.button("get result")

if get_result_clicked:
    try:
        web_convo = WebConvo()
        answer = web_convo.get_answer(query, urls=url_list)
        st.header("Answer: ")
        st.write(answer)
    except:
        st.write("There is a problem with your inputs")