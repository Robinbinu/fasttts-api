# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for audio processing and compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    ffmpeg \
    espeak-ng \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install uv for faster dependency management
RUN pip install --no-cache-dir uv

# Install Python dependencies using uv
RUN uv pip install --system --no-cache \
    fastapi[all] \
    "realtimetts[all]" \
    pydub \
    websockets \
    python-dotenv

# Copy application files
COPY src/ ./src/
COPY static/ ./static/
COPY lib/ ./lib/

# Create models directory for Hugging Face cache
# COPY models/ ./models/
RUN mkdir -p /app/models


# Install the package in editable mode
RUN uv pip install --system --no-cache -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV PORT=8080

# Cloud Run sets PORT environment variable, use it if available
ENV TTS_FASTAPI_PORT=${PORT}

# Expose the port
EXPOSE 8080

# Run the application
CMD exec uvicorn fasttts.app:app --host 0.0.0.0 --port 8080 --workers 1
