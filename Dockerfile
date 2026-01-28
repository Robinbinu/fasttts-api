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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python environment
COPY --from=builder /usr/local /usr/local

# Clean up Python cache and test files to reduce size
RUN find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type d -name "*.dist-info" -exec rm -rf {}/direct_url.json {} + 2>/dev/null || true \
    && find /usr/local -type f -name "*.pyc" -delete 2>/dev/null || true \
    && find /usr/local -type f -name "*.pyo" -delete 2>/dev/null || true

# Copy app + vendored RealtimeTTS
COPY src/ ./src/
COPY lib/ ./lib/
COPY models/ ./models/
COPY static/ ./static/

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV PYTHONPATH=/app/lib
ENV HF_HUB_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8080

CMD ["uvicorn", "fasttts.app:app", "--host", "0.0.0.0", "--port", "8080"]
