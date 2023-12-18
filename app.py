import streamlit as st
from main import create_vector_db, get_qa_chain

st.title("Q & A bot")
btn = st.button("create knowledgebase")
if btn:
    create_vector_db()
question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer: ")
    st.write(response["result"])

