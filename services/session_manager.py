import uuid
from core.pdf_processing import extract_text, chunk_fixed, chunk_recursive
from core.embeddings import encode_chunks, build_faiss_index

sessions = {}

def create_session(uploaded_file, strategy="recursive"):
    session_id = str(uuid.uuid4())
    text_blocks = extract_text(uploaded_file)
    
    if strategy == "fixed":
        chunks = chunk_fixed(text_blocks)
    elif strategy == "recursive":
        chunks = chunk_recursive(text_blocks)
    else:
        raise ValueError("Unknown chunking strategy")

    embeddings = encode_chunks(chunks)
    index = build_faiss_index(embeddings)
    sessions[session_id] = {
        "file_name": uploaded_file,
        "chunks": chunks,
        "embeddings": embeddings,
        "index": index,
        "strategy": strategy
    }
    return session_id

def get_session(session_id: str):
    return sessions.get(session_id, None)