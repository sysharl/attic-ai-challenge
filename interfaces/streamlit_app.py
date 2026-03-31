import streamlit as st
from services.session_manager import create_session, get_session
from core.embeddings import retrieve_top_k, model
from core.answer_generation import generate_answer
from services import metrics
import time
import os
import uuid
from datetime import datetime

UPLOAD_FOLDER = "/data"
ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def main():
    st.title("ATTIC AI Demo")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True 
    )

    if uploaded_files:
        batch_id = datetime.today().strftime('%Y-%m-%d') + "_" + str(uuid.uuid4())[:8]
        batch_dir = os.path.join(UPLOAD_FOLDER, batch_id)
        os.makedirs(batch_dir, exist_ok=True)

        saved_paths = []

        for uploaded_file in uploaded_files:
            if uploaded_file and allowed_file(uploaded_file.name):
                file_path = os.path.join(batch_dir, uploaded_file.name)
                file_bytes = uploaded_file.read() 
                    
                # Save file to disk
                with open(file_path, "wb") as f:
                    f.write(file_bytes)

                saved_paths.append(file_path)
                st.success(f"{len(saved_paths)} files uploaded successfully!")
                st.write(saved_paths)
                
                # Call your existing logic per file
                session_id = create_session(file_path)
                st.success(f"Session created: {session_id}")
                session = get_session(session_id)
                st.write(f"Document: {session['file_name']}")
                st.write(f"Chunks extracted: {len(session['chunks'])}")

        query = st.text_input("Enter your question:")
        if query:
            start = time.time()
            query_embedding = model.encode([query])[0]
            topk_idx = retrieve_top_k(session['index'], query_embedding, k=3)
            retrieved_chunks = [session['chunks'][i] for i in topk_idx]
            result = generate_answer(query, retrieved_chunks)
            latency = time.time() - start
            metrics.record_query(latency)
            
            st.subheader("Answer")
            st.write(result["answer"])
            st.subheader("Sources")
            st.write(result["sources"])