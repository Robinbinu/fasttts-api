"""WebSocket route for real-time TTS."""

import asyncio
import base64
from queue import Empty

from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..engines import engine_manager
from ..services import TTSRequestHandler, detect_audio_format, convert_audio_to_pcm, create_complete_wav_file

router = APIRouter()


@router.websocket("/ws")
@router.websocket("/{voice}/ws")
async def websocket_endpoint(websocket: WebSocket, voice: Optional[str] = None):
    """WebSocket endpoint for TTS - supports multiple concurrent users.

    Accepts text messages and returns audio chunks in WAV format.
    """
    # Check if system engine is active - it doesn't support WebSocket mode
    if engine_manager.current_engine_name == "system":
        await websocket.accept()
        await websocket.send_json({
            "error": {
                "message": "System engine (pyttsx3) does not support WebSocket mode. Please switch to OpenAI, Kokoro, Azure, or ElevenLabs engine.",
                "engineName": "system"
            }
        })
        await websocket.close()
        print("WebSocket connection rejected - system engine not supported")
        return

    await websocket.accept()
    print("WebSocket client connected")

    # Set voice for this session if provided
    if voice:
        engine_manager.set_voice(voice)
        print(f"Voice set to: {voice}")

    request_queue: asyncio.Queue = asyncio.Queue()
    is_active = True

    async def generate_audio(text: str) -> list[bytes]:
        """Generate audio for given text and return chunks."""
        handler = TTSRequestHandler(engine_manager.current_engine)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, handler.play_text_to_speech, text)

        chunks = []
        while True:
            try:
                chunk = handler.audio_queue.get(timeout=0.1)
                if chunk is None:
                    break
                chunks.append(chunk)
            except Empty:
                if handler.generation_complete.is_set():
                    break
                await asyncio.sleep(0.01)

        return chunks

    async def process_requests():
        """Process queued requests one by one."""
        nonlocal is_active
        while is_active:
            try:
                text = await asyncio.wait_for(request_queue.get(), timeout=1.0)
                if text is None:
                    break

                print(f"Processing: '{text}'")

                audio_chunks = await generate_audio(text)

                if not audio_chunks:
                    print("No audio chunks generated")
                    continue

                # Combine all chunks and detect format
                audio_data = b''.join(audio_chunks)
                detected_format = detect_audio_format(audio_chunks[0])
                print(f"WebSocket - Detected audio format: {detected_format}")

                # Convert to PCM if needed
                sample_rate = 24000
                if detected_format != 'pcm':
                    print(f"WebSocket - Converting {detected_format} to PCM...")
                    audio_data, sample_rate, channels, sample_width = convert_audio_to_pcm(audio_data, detected_format)
                    print(f"WebSocket - Conversion complete. Audio size: {len(audio_data)} bytes at {sample_rate}Hz")
                else:
                    try:
                        _, _, engine_rate = engine_manager.current_engine.get_stream_info()
                        if engine_rate and engine_rate > 0:
                            sample_rate = engine_rate
                    except Exception:
                        pass

                # Create complete WAV file for Safari/iOS compatibility
                print(f"WebSocket - Creating complete WAV file: {sample_rate}Hz, size: {len(audio_data)} bytes")
                complete_wav = create_complete_wav_file(audio_data, sample_rate)
                print(f"WebSocket - Complete WAV file size: {len(complete_wav)} bytes")

                # Send complete WAV in chunks
                chunk_size = 8192
                for i in range(0, len(complete_wav), chunk_size):
                    chunk = complete_wav[i:i + chunk_size]
                    await websocket.send_json({
                        "audioOutput": {
                            "audio": base64.b64encode(chunk).decode('utf-8'),
                            "format": "wav",
                            "sampleRate": sample_rate
                        }
                    })

                # Send completion signal
                await websocket.send_json({
                    "finalOutput": {
                        "isFinal": True
                    }
                })
                print(f"Sent {len(audio_chunks)} chunks")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing request: {e}")
                break

    async def receive_messages():
        """Receive and queue text messages."""
        nonlocal is_active
        try:
            while is_active:
                text = await websocket.receive_text()
                if text.strip() and text.strip().lower() != "stop":
                    await request_queue.put(text)
                    print(f"Queued: '{text}'")
                else:
                    is_active = False
                    await request_queue.put(None)
                    break
        except WebSocketDisconnect:
            print("Client disconnected")
            is_active = False
            await request_queue.put(None)
        except Exception as e:
            print(f"Error receiving: {e}")
            is_active = False
            await request_queue.put(None)

    try:
        await asyncio.gather(receive_messages(), process_requests())
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        is_active = False
        try:
            await websocket.close()
        except Exception:
            pass
        print("WebSocket closed")
