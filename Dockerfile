# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — build the React frontend
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build          # outputs frontend/dist

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Python backend (serves API + the built frontend)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS backend
WORKDIR /app

# HF_HOME points at an always-writable dir for the model download at runtime
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf_cache

# System libs: tesseract for OCR fallback, libs OpenCV-headless / torch need
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (much smaller than the default CUDA build)
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY core/ ./core/
COPY detectors/ ./detectors/
COPY fusion/ ./fusion/
COPY ingestion/ ./ingestion/
COPY model/ ./model/
# Hugging Face Spaces can't store binaries in plain git, so the trained model is
# kept in the public GitHub repo and downloaded here at build time.
RUN mkdir -p model/checkpoints && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/Suryakarthik-1/docforensics_V2/main/model/checkpoints/best.pt', 'model/checkpoints/best.pt')" && \
    python -c "import os; assert os.path.getsize('model/checkpoints/best.pt') > 1_000_000, 'model download failed'"
COPY synth/ ./synth/
COPY api/ ./api/
COPY --from=frontend /app/frontend/dist ./frontend/dist

EXPOSE 7860
# Render/Railway inject $PORT; Hugging Face Spaces expects 7860 (our default).
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
