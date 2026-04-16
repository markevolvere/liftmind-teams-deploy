# ── Stage 1: build dependencies ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Build tools: gcc/g++ for psycopg2/numpy/sentence-transformers compilation.
# libpq-dev: PostgreSQL client headers for psycopg2-binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (much smaller than default GPU build).
# Doing this as a separate layer means Docker can cache it independently.
RUN pip install --no-cache-dir --prefix=/install \
    torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Runtime dependencies: libpq for psycopg2, curl for health check.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages from builder stage.
COPY --from=builder /install /usr/local

# ── Pre-download the sentence-transformers model ──────────────────────────────
# This bakes the model into the image so there is no cold-start download.
# The model is ~90 MB (all-MiniLM-L6-v2, 384-dim).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('Model downloaded OK')"

# ── Copy application ───────────────────────────────────────────────────────────
COPY app.py ./
COPY liftmind/ ./liftmind/
COPY prompts/ ./prompts/

# data/ dir is optional (manuals). Mount at runtime or pre-populate.
# COPY data/ ./data/

# ── Non-root user for security ────────────────────────────────────────────────
RUN groupadd -r botuser && useradd -r -g botuser -d /home/botuser -m botuser \
    && chown -R botuser:botuser /app /home/botuser
USER botuser

EXPOSE 3978

ENV PORT=3978
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/home/botuser/.cache/huggingface

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["python", "app.py"]
