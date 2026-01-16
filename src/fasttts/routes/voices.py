"""Voice and engine management routes."""

import logging

from fastapi import APIRouter, Query, Request

from ..engines import engine_manager

router = APIRouter()


@router.get("/engines")
def get_engines() -> list[str]:
    """Get list of available TTS engines."""
    return engine_manager.get_engine_names()


@router.get("/set_engine")
def set_engine(request: Request, engine_name: str = Query(...)):
    """Switch to a different TTS engine.

    Args:
        request: The HTTP request
        engine_name: Name of the engine to switch to

    Returns:
        Success or error message
    """
    if engine_name not in engine_manager.engines:
        return {"error": "Engine not supported"}

    try:
        engine_manager.set_engine(engine_name)
        return {"message": f"Switched to {engine_name} engine"}
    except Exception as e:
        logging.error(f"Error switching engine: {str(e)}")
        return {"error": "Failed to switch engine"}


@router.get("/voices")
def get_voices() -> list[str]:
    """Get list of available voices for current engine."""
    return engine_manager.get_voices()


@router.get("/setvoice")
def set_voice(request: Request, voice_name: str = Query(...)):
    """Set the voice for the current TTS engine.

    Args:
        request: The HTTP request
        voice_name: Name of the voice to use

    Returns:
        Success or error message
    """
    print(f"Getting request: {voice_name}")

    if not engine_manager.current_engine:
        print("No engine is currently selected")
        return {"error": "No engine is currently selected"}

    try:
        print(f"Setting voice to {voice_name}")
        if engine_manager.set_voice(voice_name):
            return {"message": f"Voice set to {voice_name} successfully"}
        return {"error": "Failed to set voice"}
    except Exception as e:
        print(f"Error setting voice: {str(e)}")
        logging.error(f"Error setting voice: {str(e)}")
        return {"error": "Failed to set voice"}
