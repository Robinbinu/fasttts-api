# ======================
# Builder
# ======================
FROM python:3.11-slim AS builder
WORKDIR /build

# Build deps only
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# App + vendored lib
COPY pyproject.toml ./
COPY src/ ./src/
COPY lib/ ./lib/

# Tools
RUN pip install --no-cache-dir uv

# CPU-only torch (no CUDA)
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Runtime python deps (minimal)
RUN uv pip install --system --no-cache \
    fastapi \
    uvicorn \
    websockets \
    python-dotenv \
    pydub

# Install your app
RUN uv pip install --system --no-cache .

# ======================
# Runtime
# ======================
FROM python:3.11-slim
WORKDIR /app

# Runtime-only system libs
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy python environment
COPY --from=builder /usr/local /usr/local

# Copy app + vendored lib
COPY src/ ./src/
COPY lib/ ./lib/
COPY static/ ./static/

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080
CMD ["uvicorn", "fasttts.app:app", "--host", "0.0.0.0", "--port", "8080"]
