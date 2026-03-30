import streamlit as st
from services.session_manager import create_session, get_session
from core.embeddings import retrieve_top_k, model
from core.answer_generation import generate_answer
from services import metrics
import time

def main():
    st.title("ATTIC AI Demo")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        session_id = create_session(uploaded_file)
        st.success(f"Session created: {session_id}")
        session = get_session(session_id)
        st.write(f"Document: {session['file_name']}")
        st.write(f"Chunks extracted: {len(session['chunks'])}")

        query = st.text_input("Enter your question:")
        if query:
            start = time.time()
            query_embedding = model.encode([query])[0]
            topk_idx = retrieve_top_k(session['index'], session['embeddings'], query_embedding, k=3)
            retrieved_chunks = [session['chunks'][i] for i in topk_idx]
            result = generate_answer(query, retrieved_chunks)
            latency = time.time() - start
            metrics.record_query(latency)
            
            st.subheader("Answer")
            st.write(result["answer"])
            st.subheader("Sources")
            st.write(result["sources"])

    st.sidebar.subheader("Monitoring")
    st.sidebar.metric("Total Queries", metrics.metrics["queries"])
    st.sidebar.metric("Avg Latency (s)", f"{metrics.metrics['avg_latency']:.2f}")
    st.sidebar.metric("Total Tokens", metrics.metrics["total_tokens"])