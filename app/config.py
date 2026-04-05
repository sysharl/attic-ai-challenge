import logging
import os

from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

API_PREFIX = os.getenv("API_CONTEXT_PATH","")
# --- RAG / Chunking Settings ---
# Method: 'recursive' or 'fixed'
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "recursive").lower()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 3))

# --- Provider & Model Settings ---
# Provider: 'openai' or 'huggingface'
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface").lower()
BASE_MODEL = os.getenv("BASE_MODEL")

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# --- Validation (Optional but recommended for 'Code Quality' criteria) ---
def validate_config():
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        logging.warning(
            "⚠️ WARNING: LLM_PROVIDER is set to OpenAI but OPENAI_API_KEY is missing."
        )
    if LLM_PROVIDER == "huggingface" and not HUGGINGFACE_API_KEY:
        logging.warning(
            "⚠️ WARNING: LLM_PROVIDER is set to Hugging Face but HUGGINGFACE_API_KEY is missing."
        )


validate_config()
