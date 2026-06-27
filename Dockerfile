# ──────────────────────────────────────────────────────────────
# FacePipe — Production Docker Image
# ──────────────────────────────────────────────────────────────
# Build:  docker build -t facepipe .
# Run:    docker run -p 8000:8000 facepipe
# GPU:    docker build --build-arg GPU=1 -t facepipe-gpu .
# ──────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install only runtime system dependencies (OpenCV, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos "" facepipe

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY --chown=facepipe:facepipe . .

# Create data directories
RUN mkdir -p data/embeddings data/events data/index models \
    && chown -R facepipe:facepipe data models

# Switch to non-root user
USER facepipe

# Expose API port
EXPOSE 8000

# Production environment defaults
ENV FR_ENV=production \
    FR_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API server
CMD ["uvicorn", "entrypoints.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
