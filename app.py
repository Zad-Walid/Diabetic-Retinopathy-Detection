# app.py
import streamlit as st
from chatbot import load_system_prompt, send_to_mixtral_with_rag

st.set_page_config(page_title="Diabetic Retinopathy Assistant", page_icon="🧑‍⚕️")
st.title("🧑‍⚕️ Diabetic Retinopathy Assistant")
st.markdown("Ask me anything about Diabetic Retinopathy (DR). I'm here to help!")


if "chat_history" not in st.session_state:
    system_prompt = load_system_prompt()
    st.session_state.chat_history = [
        {"role": "system", "content": system_prompt}
    ]

user_input = st.text_input("Your question:")

if st.button("Ask") and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        response = send_to_mixtral_with_rag(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**Bot:** {response}")

if st.button("🗑️ Clear Chat"):
    system_prompt = load_system_prompt()
    st.session_state.chat_history = [
        {"role": "system", "content": system_prompt}
    ]
    st.success("Chat history cleared.")
