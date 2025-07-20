import streamlit as st
import requests
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import json

# Load resources (MedQuAD and FAISS index)
@st.cache_resource
def load_resources():
    data = pd.read_csv("medquad.csv")  # Load MedQuAD dataset
    questions = data['question'].tolist()
    answers = data['answer'].tolist()

    # Load FAISS index
    index = faiss.read_index("medquad_index.faiss")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    return questions, answers, model, index

questions, answers, embed_model, index = load_resources()

# Retrieval function using FAISS
def retrieve_context(query, top_k=3):
    """
    Retrieve the top-k most relevant answers from the MedQuAD dataset.
    """
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    relevant_answers = [answers[i] for i in indices[0]]
    return " ".join(relevant_answers)


def query_llama_stream(query, context):
    """
    Stream and process responses from Llama3.2:3b via Ollama API.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2:3b",
        "prompt": f"Context: {context}\n\nQuestion: {query}"
    }
    headers = {"Content-Type": "application/json"}

    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_lines():
                if chunk:
                    # Parse each JSON chunk
                    data = json.loads(chunk)
                    if "response" in data:
                        yield data["response"]  # Extract and yield the "response" field
    except requests.RequestException as e:
        yield f"Error: {e}"


def main():
    st.title("🩺 DigiDoctor: AI-Powered Medical Assistant")
    st.markdown("Ask your medical queries below and get real-time responses!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Enter your question:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve context and generate response
        with st.spinner("Retrieving relevant context..."):
            context = retrieve_context(prompt)

        response_text = ""
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            for chunk in query_llama_stream(prompt, context):
                response_text += chunk
                response_placeholder.markdown(response_text)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
