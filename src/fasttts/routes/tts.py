"""TTS synthesis route."""

import threading

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from ..engines import engine_manager
from ..services import TTSRequestHandler

router = APIRouter()

BROWSER_IDENTIFIERS = [
    "mozilla",
    "chrome",
    "safari",
    "firefox",
    "edge",
    "opera",
    "msie",
    "trident",
]


def is_browser_request(request: Request) -> bool:
    """Check if request is from a web browser based on User-Agent."""
    user_agent = request.headers.get("user-agent", "").lower()
    return any(browser_id in user_agent for browser_id in BROWSER_IDENTIFIERS)


@router.get("/tts")
async def tts(request: Request, text: str = Query(...)):
    """Convert text to speech and stream audio response.

    Args:
        request: The HTTP request
        text: Text to convert to speech

    Returns:
        StreamingResponse with WAV audio
    """
    with engine_manager.tts_lock:
        request_handler = TTSRequestHandler(engine_manager.current_engine)
        browser_request = is_browser_request(request)

        if engine_manager.play_semaphore.acquire(blocking=False):
            try:
                threading.Thread(
                    target=request_handler.play_text_to_speech,
                    args=(text,),
                    daemon=True,
                ).start()
            finally:
                engine_manager.play_semaphore.release()

        return StreamingResponse(
            request_handler.audio_chunk_generator(browser_request),
            media_type="audio/wav",
        )
