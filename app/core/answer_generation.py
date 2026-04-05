from huggingface_hub import InferenceClient
from openai import OpenAI

from app.config import (BASE_MODEL, HUGGINGFACE_API_KEY, LLM_PROVIDER,
                        OPENAI_API_KEY, )

HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
hf_client = InferenceClient(
    base_url="https://router.huggingface.co/v1", api_key=HUGGINGFACE_API_KEY
)

OPENAI_MODEL = "gpt-4o-mini"
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def generate_answer(query: str, chunks: list) -> dict:
    """
    Generates an answer based on retrieved chunks and configured LLM provider.
    """
    if not chunks:
        return {"answer": "No relevant content found.", "sources": []}

    # Prepare Context & Prompt
    context = "\n".join(
        [
            f"--- [Page {c['page']}, Chunk {c['chunk_index']}] ---\n{c['text']}"
            for c in chunks
        ]
    )

    system_message = (
        "You are a strictly grounded Document Assistant. Your task is to provide answers "
        "based ONLY on the provided context. \n\n"
        "RULES:\n"
        "1. Use ONLY the 'CONTEXT' provided below.\n"
        "2. If the answer is not contained within the context, respond exactly with: "
        "'I cannot find this in the document.'\n"
        "3. Do not use outside knowledge or mention your own training data.\n"
        "4. Keep the tone professional and the summary concise."
    )

    user_prompt = (
        f"Below are several snippets retrieved from a 20-page document.\n\n"
        f"<CONTEXT>\n{context}\n</CONTEXT>\n\n"
        f"QUESTION: {query}\n\n"
        f"FINAL INSTRUCTION: Synthesize an answer based on the <CONTEXT> above. "
        f"If the snippets are disconnected, try to find the most relevant one."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    sources = chunks
    provider = LLM_PROVIDER.lower()
    try:
        # 2. Route to the configured Provider
        if provider == "openai":
            response = openai_client.chat.completions.create(
                model=BASE_MODEL or OPENAI_MODEL,
                messages=messages,
                temperature=0,  # Deterministic for RAG
            )
            answer_text = response.choices[0].message.content
        else:  # Default to Hugging Face
            response = hf_client.chat.completions.create(
                model=BASE_MODEL or HF_MODEL,
                messages=messages,
                max_tokens=512,
                temperature=0.1,
            )
            answer_text = response.choices[0].message.content

    except Exception as e:
        answer_text = f"Error during generation: {str(e)}"
        sources = []

    return {"answer": answer_text, "sources": sources}
