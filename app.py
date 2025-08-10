# app.py
import streamlit as st
from bot_rag import run_rag  # import your RAG function

st.set_page_config(page_title="🧠 Local RAG Chatbot", layout="wide")
st.title("🧠 Local RAG Chatbot (Ollama + FAISS)")

# Initialize chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Input
query = st.text_input("Ask me something about your data:", key="query_input")

if st.button("Send") and query:
    with st.spinner("Getting answer..."):
        answer = run_rag(query)

    # Save conversation
    st.session_state.chat.append(("🧍 You", query))
    st.session_state.chat.append(("🤖 Bot", answer))

# Display chat history
for sender, message in st.session_state.chat[::-1]:  # reverse order
    st.markdown(f"**{sender}:** {message}")
