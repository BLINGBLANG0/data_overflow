FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY api/ api/
COPY frontend/ frontend/
COPY solution.py .
COPY model.pkl .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run — Render injects $PORT; default to 8000 locally
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
