# ======================
# Builder stage
# ======================
FROM python:3.10-slim AS builder

WORKDIR /build

# Only build-time system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY lib/ ./lib/

# Install uv
RUN pip install --no-cache-dir uv

# Install Python deps from pyproject.toml
RUN uv pip install --system --no-cache .

# ======================
# Runtime stage
# ======================
FROM python:3.10-slim

WORKDIR /app

# Runtime-only system deps (minimal, no recommends)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    ffmpeg \
    espeak-ng \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python environment
# Copy only what we need from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Clean up Python cache and test files to reduce size
RUN find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type d -name "*.dist-info" -exec rm -rf {}/direct_url.json {} + 2>/dev/null || true \
    && find /usr/local -type f -name "*.pyc" -delete 2>/dev/null || true \
    && find /usr/local -type f -name "*.pyo" -delete 2>/dev/null || true

# Download models from GCS bucket
ARG GCS_BUCKET_URL=https://storage.googleapis.com/fasttts-models
RUN mkdir -p models && \
    curl -L "${GCS_BUCKET_URL}/models.tar.gz" -o /tmp/models.tar.gz && \
    tar -xzf /tmp/models.tar.gz -C models/ && \
    rm /tmp/models.tar.gz

# Copy app + vendored RealtimeTTS
COPY src/ ./src/
COPY lib/ ./lib/
COPY static/ ./static/

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV PYTHONPATH=/app/lib
ENV HF_HOME=models
ENV HF_HUB_CACHE=/app/models/hub
ENV TRANSFORMERS_CACHE=/app/models/hub
ENV HF_TOKEN=hf_bPyeshcmjrOTeQntrCdmnSKAlWmCcoNUimD
# ENV HF_HUB_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8080

CMD ["uvicorn", "fasttts.app:app", "--host", "0.0.0.0", "--port", "8080"]
