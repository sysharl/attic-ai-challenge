import time
from langchain_community.callbacks import get_openai_callback

def generate_answer_with_metrics(query, chunks):
    start_time = time.time()
    
    with get_openai_callback() as cb:
        # Call your existing function
        result = generate_answer(query, chunks)
        
        latency = time.time() - start_time
        
        # Log to console or a CSV/Database
        print(f"--- Session Metrics ---")
        print(f"Latency: {latency:.2f}s")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Cost (USD): ${cb.total_cost}")
        
    return result