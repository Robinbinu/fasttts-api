"""Configuration management for FastTTS."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Server
    port: int = field(default_factory=lambda: int(os.environ.get("TTS_FASTAPI_PORT", 8000)))
    debug_logging: bool = field(default_factory=lambda: os.environ.get("DEBUG_LOGGING", "").lower() == "true")

    # Model paths
    hf_home: Path = field(default_factory=lambda: Path(os.environ.get("HF_HOME", "models")).resolve())

    # API Keys
    azure_speech_key: str | None = field(default_factory=lambda: os.environ.get("AZURE_SPEECH_KEY"))
    azure_speech_region: str | None = field(default_factory=lambda: os.environ.get("AZURE_SPEECH_REGION"))
    elevenlabs_api_key: str | None = field(default_factory=lambda: os.environ.get("ELEVENLABS_API_KEY"))
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))

    # Supported engines
    supported_engines: list[str] = field(default_factory=lambda: [
        "azure",
        "openai",
        "elevenlabs",
        "system",
        "kokoro"
    ])

    # CORS origins
    cors_origins: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Set HF_HOME environment variable
        os.environ["HF_HOME"] = str(self.hf_home)

        # Build CORS origins
        self.cors_origins = [
            "http://localhost",
            f"http://localhost:{self.port}",
            "http://127.0.0.1",
            f"http://127.0.0.1:{self.port}",
            "https://localhost",
            f"https://localhost:{self.port}",
            "https://127.0.0.1",
            f"https://127.0.0.1:{self.port}",
        ]


# Global settings instance
settings = Settings()
