import logging
import os
import uuid
import time
from app.core.embeddings import build_faiss_index, encode_chunks
from app.core.pdf_processing import chunk_fixed, chunk_recursive, extract_text

sessions = {}

def create_session(uploaded_file, strategy="recursive"):
    session_id = str(uuid.uuid4())
    start_time = time.time() 
    text_blocks = extract_text(uploaded_file)

    if not text_blocks:
        logging.error(f"Skipping {uploaded_file}: No text found.")
        raise ValueError("No text found in PDF")

    filename = os.path.basename(uploaded_file)
    if strategy == "fixed":
        chunks = chunk_fixed(text_blocks, filename)
    elif strategy == "recursive":
        chunks = chunk_recursive(text_blocks, filename)
    else:
        raise ValueError("Unknown chunking strategy")

    if not chunks:
        raise ValueError("Could not create text chunks")

    embeddings = encode_chunks(chunks)

    if embeddings is None:
        raise ValueError("Failed to generate embeddings")

    index = build_faiss_index(embeddings)
    indexing_time = time.time() - start_time 
    print(f"DEBUG: Indexing completed in {indexing_time:.2f}s")
    sessions[session_id] = {
        "file_name": uploaded_file,
        "chunks": chunks,
        "embeddings": embeddings,
        "index": index,
        "strategy": strategy,
    }
    return session_id


def get_session(session_id: str):
    return sessions.get(session_id, None)
