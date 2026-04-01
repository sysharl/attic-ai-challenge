import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# RAG / chunk settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 3))

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")