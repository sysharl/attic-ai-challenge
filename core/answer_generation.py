import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def generate_answer(query: str, chunks: list) -> dict:
    if not chunks:
        return {"answer": "No relevant content found.", "sources": []}

    context = "\n".join([c["text"] for c in chunks])
    prompt = f"Answer the question based on the context below. Cite page and chunk indices.\n\nContext:\n{context}\n\nQuestion: {query}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer based on context"},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        answer_text = response.choices[0].message.content
        return {
            "answer": answer_text,
            "sources": [{"page": c["page"], "chunk_index": c["chunk_index"]} for c in chunks]
        }
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": []}