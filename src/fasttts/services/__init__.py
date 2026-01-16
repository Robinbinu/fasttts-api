"""Service modules for FastTTS."""

from .audio import (
    detect_audio_format,
    convert_audio_to_pcm,
    create_complete_wav_file,
    create_wave_header_for_engine,
    PYDUB_AVAILABLE,
)
from .synthesis import TTSRequestHandler

__all__ = [
    "detect_audio_format",
    "convert_audio_to_pcm",
    "create_complete_wav_file",
    "create_wave_header_for_engine",
    "TTSRequestHandler",
    "PYDUB_AVAILABLE",
]
