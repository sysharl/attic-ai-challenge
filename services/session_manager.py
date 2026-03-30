import uuid
from core.pdf_processing import extract_text, chunk_fixed
from core.embeddings import encode_chunks, build_faiss_index

sessions = {}

def create_session(uploaded_file):
    session_id = str(uuid.uuid4())
    text_blocks = extract_text(uploaded_file)
    chunks = chunk_fixed(text_blocks)
    embeddings = encode_chunks(chunks)
    index = build_faiss_index(embeddings)
    sessions[session_id] = {
        "file_name": uploaded_file.name,
        "chunks": chunks,
        "embeddings": embeddings,
        "index": index
    }
    return session_id

def get_session(session_id: str):
    return sessions.get(session_id, None)