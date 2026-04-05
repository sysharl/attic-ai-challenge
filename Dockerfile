# --- Stage 1: Build Stage ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies needed for some Python packages (e.g., gcc for numpy/pandas)
RUN apt-get update && apt-get install -y build-essential

# Install dependencies into a local folder
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# --- Stage 2: Final Runtime Stage ---
FROM python:3.11-slim

WORKDIR /app

# Copy only the installed libraries from the builder stage
COPY --from=builder /root/.local /root/.local
COPY . .

# Ensure the scripts in .local/bin are usable
ENV PATH=/root/.local/bin:$PATH
# Keep Python from buffering logs (makes them show up instantly in Docker logs)
ENV PYTHONUNBUFFERED=1

# Change this to whatever starts your API (e.g., uvicorn, flask, etc.)
CMD ["python", "uvicorn", "app.main:app"]