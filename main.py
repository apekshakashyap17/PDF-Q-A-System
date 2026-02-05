import streamlit as st
from query import query_rag 
from vector_store import create_vector_store

if "db" not in st.session_state:
    with st.spinner("Initializing Database..."):
        st.session_state.db = create_vector_store() 

st.set_page_config(page_title="PDF Q&A System", page_icon="🌍")
st.title("🌍 PDF Q&A System")
st.markdown("Ask questions about the climate impacts and adaptation reports.")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing PDF context..."):
            # Call your RAG function
            # Note: Ensure query_rag returns just the text string
            response = query_rag(prompt) 
            st.markdown(response)
    
    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": response})