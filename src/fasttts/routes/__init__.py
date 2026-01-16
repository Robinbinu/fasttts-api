"""API route modules for FastTTS."""

from .tts import router as tts_router
from .voices import router as voices_router
from .websocket import router as websocket_router

__all__ = ["tts_router", "voices_router", "websocket_router"]
