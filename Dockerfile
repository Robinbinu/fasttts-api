# ======================
# Builder stage
# ======================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    ffmpeg \
    espeak-ng \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir uv

# 🔴 Install ONLY what you need (NO [all])
RUN uv pip install --system --no-cache \
    fastapi \
    uvicorn[standard] \
    websockets \
    python-dotenv \
    pydub \
    realtimetts \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/
COPY lib/ ./lib/

RUN uv pip install --system --no-cache .

# ======================
# Runtime stage
# ======================
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libportaudio2 \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy only what we need
COPY --from=builder /usr/local /usr/local
COPY src/ ./src/
COPY static/ ./static/

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "fasttts.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
