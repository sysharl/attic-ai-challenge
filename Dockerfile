# --- Stage 1: Build Stage ---
FROM python:3.11-slim as builder

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for building some Python packages (like FAISS or hardware-linked libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into a local folder
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# --- Stage 2: Runtime Stage ---
FROM python:3.11-slim

WORKDIR /app

# Copy only the installed packages from the builder stage
COPY --from=builder /root/.local /root/.local
# Copy your application code
COPY . .

# Update PATH to include the local bin where pip installed scripts
ENV PATH=/root/.local/bin:$PATH
# Ensure the app knows it is in production
ENV APP_ENV=production

# Install minimal runtime system libraries (libgomp1 is often required by FAISS/SciPy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Expose the port (Render will override this, but it's good practice)
EXPOSE 10000

# Start command: Use sh -c to expand the $PORT variable provided by Render
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]