import streamlit as st
from chatbot import send_to_gimini_with_rag  

# Title of the app
st.title("Diabetic Retinopathy Chatbot")

# Description/Instructions
st.write(
    """
    This chatbot answers questions related to **Diabetic Retinopathy**.
    Simply type your question below and the model will generate an answer.
    """
)

# Input for the user question
user_question = st.text_input("Ask a question about Diabetic Retinopathy:")

# Button to submit the question
if st.button("Get Answer"):
    if user_question:
        st.write("Processing your question...")

        # Call the function from your chatbot.py
        answer = send_to_gimini_with_rag(user_question)

        # Display the answer
        st.write("**Answer:**")
        st.write(answer)
    else:
        st.write("Please enter a question first!")

