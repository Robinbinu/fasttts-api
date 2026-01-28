# ======================
# Builder stage
# ======================
FROM python:3.11-slim AS builder

WORKDIR /build

# Only build-time system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY lib/ ./lib/

# Install uv
RUN pip install --no-cache-dir uv

# Install Python deps from pyproject.toml
# (includes pocket-tts, numpy<2, etc.)
RUN uv pip install --system --no-cache .

# ======================
# Runtime stage
# ======================
FROM python:3.11-slim

WORKDIR /app

# Runtime-only system deps
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment
COPY --from=builder /usr/local /usr/local

# Copy app + vendored RealtimeTTS
COPY src/ ./src/
COPY lib/ ./lib/
COPY static/ ./static/

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "fasttts.app:app", "--host", "0.0.0.0", "--port", "8080"]
