from sentence_transformers import SentenceTransformer
from app.config import TOP_K
import numpy as np
import faiss
import logging

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # safer for API

# -----------------------
# Encode Chunks
# -----------------------
def encode_chunks(chunks: list):
    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    return embeddings


# -----------------------
# Encode Query
# -----------------------
def encode_query(query: str):
    embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(embedding)
    return embedding  # shape: (1, dim)


# -----------------------
# Build Index
# -----------------------
def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product
    index.add(embeddings)
    return index


# -----------------------
# Retrieve Top-K
# -----------------------
def retrieve_top_k(index, query_embedding, k=TOP_K):
    D, I = index.search(query_embedding, k)
    return I[0]  # indices