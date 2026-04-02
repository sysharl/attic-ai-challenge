from openai import OpenAI 
from app.config import OPENAI_API_KEY

from huggingface_hub import InferenceClient
from app.config import HUGGINGFACE_API_KEY, LLM_PROVIDER,BASE_MODEL

from app.config import (
    OPENAI_API_KEY, 
    HUGGINGFACE_API_KEY, 
    LLM_PROVIDER, 
    BASE_MODEL,
    TOP_K
)

# Initialize the OpenAI client once
openai_client = OpenAI(api_key=OPENAI_API_KEY)

HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OPENAI_MODEL = "gpt-4o-mini"

client = InferenceClient(
    base_url="https://router.huggingface.co/v1",    
    api_key=HUGGINGFACE_API_KEY
)

def generate_answer(query: str, chunks: list) -> dict:
    """
    Generates an answer based on retrieved chunks and configured LLM provider.
    """
    if not chunks:
        return {"answer": "No relevant content found.", "sources": []}

    # Prepare Context & Prompt
    context = "\n".join([
        f"--- [Page {c['page']}, Chunk {c['chunk_index']}] ---\n{c['text']}" 
        for c in chunks
    ])
    
    system_message = (
        "You are a precise document assistant. Use ONLY the provided context to answer. "
        "If the answer isn't there, say 'I cannot find this in the document.'"
    )
    
    user_prompt = (
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {query}\n\n"
        f"INSTRUCTION: Answer the question using the context above. Citations are handled by the system, "
        f"so focus on a concise, grounded summary."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    sources = chunks
    try:
        # 2. Route to the configured Provider
        if LLM_PROVIDER.lower() == "openai":
            response = openai_client.chat.completions.create(
                model=BASE_MODEL or "gpt-4o-mini",
                messages=messages,
                temperature=0  # Deterministic for RAG
            )
            answer_text = response.choices[0].message.content
            
        else:  # Default to Hugging Face
            response = hf_client.chat.completions.create(
                model=BASE_MODEL or "Qwen/Qwen2.5-7B-Instruct",
                messages=messages,
                max_tokens=512,
                temperature=0.1
            )
            answer_text = response.choices[0].message.content    

    except Exception as e:
        answer_text = f"Error during generation: {str(e)}"
        sources = []

    return {
        "answer": answer_text,
        "sources": sources
    }