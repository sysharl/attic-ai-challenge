import logging
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import TOP_K

model = SentenceTransformer("intfloat/e5-small-v2", device="cpu")

# -----------------------
# Encode Chunks
# -----------------------
def encode_chunks(chunks: list):
    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    return embeddings


# -----------------------
# Encode Query
# -----------------------
def encode_query(query: str):
    query_instruction = f"query: {query}"
    start_time = time.time()
    embedding = model.encode([query], convert_to_numpy=True)
    end_time = time.time()
    indexing_time = end_time - start_time
    print(f"DEBUG: Indexing Query completed in: {indexing_time:.2f} seconds")
    faiss.normalize_L2(embedding)
    return embedding  # shape: (1, dim)


# -----------------------
# Build Index
# -----------------------
def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) 
    index.add(embeddings)
    return index


# -----------------------
# Retrieve Top-K
# -----------------------
def retrieve_top_k(index, query_embedding, k=TOP_K):
    D, I = index.search(query_embedding, k)
    return I[0]  # indices
