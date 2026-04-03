import logging
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import TOP_K

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


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
    start_time = time.time()
    embedding = model.encode([query], convert_to_numpy=True)
    end_time = time.time()
    indexing_time = end_time - start_time
    print(f"Total Indexing Time: {indexing_time:.2f} seconds")
    print(f"Time per Page: {indexing_time / 20:.4f} seconds")

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
