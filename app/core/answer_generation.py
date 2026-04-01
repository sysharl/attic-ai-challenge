# -------------------- OpenAI --------------------
from openai import OpenAI  # New 2026 syntax
from app.config import OPENAI_API_KEY

# Initialize the OpenAI client once
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------- Hugging Face --------------------
from huggingface_hub import InferenceClient
from app.config import HUGGINGFACE_API_KEY

HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

client = InferenceClient(
    base_url="https://router.huggingface.co/v1",    
    api_key=HUGGINGFACE_API_KEY
)

OPENAI_MODEL = "gpt-3.5-turbo" # Note: In 2026, gpt-4o-mini is usually faster/cheaper!

def generate_answer(query: str, chunks: list, use_openai=True) -> dict:
    if not chunks:
        return {"answer": "No relevant content found.", "sources": []}

    # Prepare Context & Prompt
    context = "\n".join([
        f"--- [Page {c['page']}, Chunk {c['chunk_index']}] ---\n{c['text']}" 
        for c in chunks
    ])
    
    # Using a clean ChatML-style prompt structure
    system_message = "You are a precise document assistant. Use ONLY the provided context."
    user_prompt = (
        f"STRICT RULES:\n"
        f"1. Use ONLY the provided Context.\n"
        f"2. If the answer isn't there, say 'I cannot find this in the document.'\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    sources = chunks

    try:
        if use_openai:
            # Updated OpenAI v1.0+ Syntax
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0
            )
            answer_text = response.choices[0].message.content
        else:
            # Hugging Face Chat Completion
            response = client.chat.completions.create(
                model=HF_MODEL,
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