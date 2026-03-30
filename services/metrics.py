metrics = {"queries": 0, "total_latency": 0.0, "avg_latency": 0.0, "total_tokens": 0}

def record_query(latency: float, tokens: int = 0):
    metrics["queries"] += 1
    metrics["total_latency"] += latency
    metrics["avg_latency"] = metrics["total_latency"] / metrics["queries"]
    metrics["total_tokens"] += tokens