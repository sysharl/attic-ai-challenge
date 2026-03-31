from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from config import TOP_K

model = SentenceTransformer("all-MiniLM-L6-v2")

def encode_chunks(chunks: list):
    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    print("embedding sample", embeddings)
    return embeddings

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def retrieve_top_k(index, query_embedding, k=TOP_K):
    D, I = index.search(np.array([query_embedding]), k)
    return I[0]