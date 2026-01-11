if __name__ == "__main__":
    print("Starting server")
    import logging

    # Enable or disable debug logging
    DEBUG_LOGGING = False

    if DEBUG_LOGGING:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)


from pathlib import Path
import os

# Set Hugging Face cache directories BEFORE importing any models
dir = Path(__file__).parent.resolve()
os.environ["HF_HOME"] = str(dir / "models")

from RealtimeTTS import (
    TextToAudioStream,
    AzureEngine,
    ElevenlabsEngine,
    SystemEngine,
    CoquiEngine,
    OpenAIEngine,
    KokoroEngine
)

from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from queue import Queue, Empty
import threading
import logging
import uvicorn
import wave
import io
import asyncio
import base64
import json

PORT = int(os.environ.get("TTS_FASTAPI_PORT", 8000))

SUPPORTED_ENGINES = [
    # "azure",
    # "openai",
    # "elevenlabs",
    # "system",
    # "coqui",  #multiple queries are not supported on coqui engine right now, comment coqui out for tests where you need server start often,
    "kokoro"
]

# change start engine by moving engine name
# to the first position in SUPPORTED_ENGINES
START_ENGINE = SUPPORTED_ENGINES[0]

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

origins = [
    "http://localhost",
    f"http://localhost:{PORT}",
    "http://127.0.0.1",
    f"http://127.0.0.1:{PORT}",
    "https://localhost",
    f"https://localhost:{PORT}",
    "https://127.0.0.1",
    f"https://127.0.0.1:{PORT}",
]

play_text_to_speech_semaphore = threading.Semaphore(1)
engines = {}
voices = {}
current_engine = None
speaking_lock = threading.Lock()
tts_lock = threading.Lock()
gen_lock = threading.Lock()


class TTSRequestHandler:
    def __init__(self, engine):
        self.engine = engine
        self.audio_queue = Queue()
        self.stream = TextToAudioStream(
            engine, on_audio_stream_stop=self.on_audio_stream_stop, muted=True
        )
        self.speaking = False
        self.generation_complete = threading.Event()

    def on_audio_chunk(self, chunk):
        self.audio_queue.put(chunk)

    def on_audio_stream_stop(self):
        print("on_audio_stream_stop called")
        self.audio_queue.put(None)
        self.speaking = False
        self.generation_complete.set()

    def play_text_to_speech(self, text):
        self.speaking = True
        self.generation_complete.clear()
        self.stream.feed(text)
        logging.debug(f"Playing audio for text: {text}")
        print(f'Synthesizing: "{text}"')
        self.stream.play_async(on_audio_chunk=self.on_audio_chunk, muted=True)
        # Note: play_async returns immediately, chunks will be added to queue via callback

    def audio_chunk_generator(self, send_wave_headers):
        first_chunk = False
        try:
            while True:
                chunk = self.audio_queue.get()
                if chunk is None:
                    print("Terminating stream")
                    break
                if not first_chunk:
                    if send_wave_headers:
                        print("Sending wave header")
                        yield create_wave_header_for_engine(self.engine)
                    first_chunk = True
                yield chunk
        except Exception as e:
            print(f"Error during streaming: {str(e)}")


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a CSP that allows 'self' for script sources for firefox
csp = {
    "default-src": "'self'",
    "script-src": "'self'",
    "style-src": "'self' 'unsafe-inline'",
    "img-src": "'self' data:",
    "font-src": "'self' data:",
    "media-src": "'self' blob:",
}
csp_string = "; ".join(f"{key} {value}" for key, value in csp.items())


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = csp_string
    return response


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")


def _set_engine(engine_name):
    global current_engine, stream
    if current_engine is None:
        current_engine = engines[engine_name]
    else:
        current_engine = engines[engine_name]

    if voices[engine_name]:
        engines[engine_name].set_voice(voices[engine_name][0].name)


@app.get("/set_engine")
def set_engine(request: Request, engine_name: str = Query(...)):
    if engine_name not in engines:
        return {"error": "Engine not supported"}

    try:
        _set_engine(engine_name)
        return {"message": f"Switched to {engine_name} engine"}
    except Exception as e:
        logging.error(f"Error switching engine: {str(e)}")
        return {"error": "Failed to switch engine"}


def is_browser_request(request):
    user_agent = request.headers.get("user-agent", "").lower()
    is_browser = any(browser_id in user_agent for browser_id in BROWSER_IDENTIFIERS)
    return is_browser


def create_wave_header_for_engine(engine):
    _, _, sample_rate = engine.get_stream_info()

    num_channels = 1
    sample_width = 2
    frame_rate = sample_rate

    wav_header = io.BytesIO()
    with wave.open(wav_header, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)

    wav_header.seek(0)
    wave_header_bytes = wav_header.read()
    wav_header.close()

    # Create a new BytesIO with the correct MIME type for Firefox
    final_wave_header = io.BytesIO()
    final_wave_header.write(wave_header_bytes)
    final_wave_header.seek(0)

    return final_wave_header.getvalue()


@app.get("/tts")
async def tts(request: Request, text: str = Query(...)):
    with tts_lock:
        request_handler = TTSRequestHandler(current_engine)
        browser_request = is_browser_request(request)

        if play_text_to_speech_semaphore.acquire(blocking=False):
            try:
                threading.Thread(
                    target=request_handler.play_text_to_speech,
                    args=(text,),
                    daemon=True,
                ).start()
            finally:
                play_text_to_speech_semaphore.release()

        return StreamingResponse(
            request_handler.audio_chunk_generator(browser_request),
            media_type="audio/wav",
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected")
    
    text_queue = asyncio.Queue()
    is_active = True
    
    async def receive_messages():
        """Receive text messages from client and queue them"""
        nonlocal is_active
        try:
            while is_active:
                text = await websocket.receive_text()
                print(f"Received text: '{text}'")
                
                # Check for stop signal
                if text.strip() == "" or text.strip().lower() == "stop":
                    print("Received stop signal")
                    is_active = False
                    await text_queue.put(None)  # Signal to stop processing
                    break
                
                await text_queue.put(text)
        except Exception as e:
            print(f"Error receiving messages: {e}")
            is_active = False
            await text_queue.put(None)
    
    async def process_and_stream():
        """Process queued text and stream audio back"""
        try:
            while is_active or not text_queue.empty():
                # Get next text from queue
                try:
                    text = await asyncio.wait_for(text_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                if text is None:  # Stop signal
                    print("Processing stopped")
                    break
                
                print(f"Processing: '{text}'")
                
                # Create a request handler for this text
                request_handler = TTSRequestHandler(current_engine)
                
                # Start TTS generation
                if play_text_to_speech_semaphore.acquire(blocking=False):
                    try:
                        # Start generation in thread
                        thread = threading.Thread(
                            target=request_handler.play_text_to_speech,
                            args=(text,),
                            daemon=True,
                        )
                        thread.start()
                        
                        # Send WAV header as first chunk
                        wave_header = create_wave_header_for_engine(current_engine)
                        wave_header_b64 = base64.b64encode(wave_header).decode('utf-8')
                        
                        # Get audio format info
                        _, _, sample_rate = current_engine.get_stream_info()
                        
                        header_message = {
                            "audioOutput": {
                                "audio": wave_header_b64,
                                "format": "wav",
                                "sampleRate": sample_rate,
                                "isHeader": True
                            }
                        }
                        await websocket.send_json(header_message)
                        print(f"Sent WAV header (sample rate: {sample_rate})")
                        
                        # Stream audio chunks as they arrive
                        chunks_sent = 0
                        timeout_count = 0
                        max_timeout = 100  # 10 seconds total (100 * 0.1s)
                        
                        while True:
                            try:
                                chunk = request_handler.audio_queue.get(timeout=0.1)
                                if chunk is None:
                                    print(f"Audio complete for this text, sent {chunks_sent} chunks")
                                    # Send final output message
                                    final_message = {
                                        "finalOutput": {
                                            "isFinal": True
                                        }
                                    }
                                    await websocket.send_json(final_message)
                                    print("Sent final output message")
                                    break
                                
                                # Encode audio chunk as base64
                                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                                
                                # Send audio output message
                                audio_message = {
                                    "audioOutput": {
                                        "audio": audio_base64
                                    }
                                }
                                await websocket.send_json(audio_message)
                                chunks_sent += 1
                                timeout_count = 0  # Reset timeout counter when we get data
                            except Empty:
                                timeout_count += 1
                                if timeout_count >= max_timeout:
                                    print(f"Timeout waiting for chunks, sent {chunks_sent} chunks")
                                    # Send final output on timeout
                                    final_message = {
                                        "finalOutput": {
                                            "isFinal": True
                                        }
                                    }
                                    await websocket.send_json(final_message)
                                    break
                                # Check if generation is complete
                                if request_handler.generation_complete.is_set() and chunks_sent > 0:
                                    print(f"Generation marked complete, sent {chunks_sent} chunks")
                                    # Send final output message
                                    final_message = {
                                        "finalOutput": {
                                            "isFinal": True
                                        }
                                    }
                                    await websocket.send_json(final_message)
                                    break
                                await asyncio.sleep(0.01)
                                continue
                    finally:
                        play_text_to_speech_semaphore.release()
                else:
                    print("TTS busy, skipping this request")
        except Exception as e:
            print(f"Error processing: {e}")
            import traceback
            traceback.print_exc()
    
    # Run both tasks concurrently
    try:
        await asyncio.gather(
            receive_messages(),
            process_and_stream()
        )
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        is_active = False
        try:
            await websocket.close()
        except:
            pass


@app.get("/engines")
def get_engines():
    return list(engines.keys())


@app.get("/voices")
def get_voices():
    voices_list = []
    for voice in voices[current_engine.engine_name]:
        voices_list.append(voice.name)
    return voices_list


@app.get("/setvoice")
def set_voice(request: Request, voice_name: str = Query(...)):
    print(f"Getting request: {voice_name}")
    if not current_engine:
        print("No engine is currently selected")
        return {"error": "No engine is currently selected"}

    try:
        print(f"Setting voice to {voice_name}")
        current_engine.set_voice(voice_name)
        return {"message": f"Voice set to {voice_name} successfully"}
    except Exception as e:
        print(f"Error setting voice: {str(e)}")
        logging.error(f"Error setting voice: {str(e)}")
        return {"error": "Failed to set voice"}


@app.get("/")
def root_page():
    engines_options = "".join(
        [
            f'<option value="{engine}">{engine.title()}</option>'
            for engine in engines.keys()
        ]
    )
    content = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Text-To-Speech</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f0f0f0;
                    margin: 0;
                    padding: 0;
                }}
                h2 {{
                    color: #333;
                    text-align: center;
                }}
                #container {{
                    width: 80%;
                    margin: 50px auto;
                    background-color: #fff;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }}
                label {{
                    font-weight: bold;
                }}
                select, textarea {{
                    width: 100%;
                    padding: 10px;
                    margin: 10px 0;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    box-sizing: border-box;
                    font-size: 16px;
                }}
                button {{
                    display: block;
                    width: 100%;
                    padding: 15px;
                    background-color: #007bff;
                    border: none;
                    border-radius: 5px;
                    color: #fff;
                    font-size: 16px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }}
                button:hover {{
                    background-color: #0056b3;
                }}
                audio {{
                    width: 80%;
                    margin: 10px auto;
                    display: block;
                }}
                .mode-selector {{
                    display: flex;
                    justify-content: center;
                    gap: 10px;
                    margin: 20px 0;
                }}
                .mode-selector button {{
                    width: auto;
                    padding: 10px 20px;
                }}
                .mode-selector button.active {{
                    background-color: #28a745;
                }}
                .status {{
                    text-align: center;
                    margin: 10px 0;
                    color: #666;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div id="container">
                <h2>Text to Speech</h2>
                <div class="mode-selector">
                    <button id="httpMode" class="active">HTTP Mode</button>
                    <button id="wsMode">WebSocket Mode</button>
                </div>
                <div class="status" id="status">Mode: HTTP</div>
                <label for="engine">Select Engine:</label>
                <select id="engine">
                    {engines_options}
                </select>
                <label for="voice">Select Voice:</label>
                <select id="voice">
                    <!-- Options will be dynamically populated by JavaScript -->
                </select>
                <textarea id="text" rows="4" cols="50" placeholder="Enter text here..."></textarea>
                <button id="speakButton">Speak</button>
                <audio id="audio" controls></audio>
            </div>
            <script src="/static/tts.js"></script>
        </body>
    </html>
    """
    return HTMLResponse(content=content)


if __name__ == "__main__":
    print("Initializing TTS Engines")

    for engine_name in SUPPORTED_ENGINES:
        if "azure" == engine_name:
            azure_api_key = os.environ.get("AZURE_SPEECH_KEY")
            azure_region = os.environ.get("AZURE_SPEECH_REGION")
            if azure_api_key and azure_region:
                print("Initializing azure engine")
                engines["azure"] = AzureEngine(azure_api_key, azure_region)

        if "elevenlabs" == engine_name:
            elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
            if elevenlabs_api_key:
                print("Initializing elevenlabs engine")
                engines["elevenlabs"] = ElevenlabsEngine(elevenlabs_api_key)

        if "system" == engine_name:
            print("Initializing system engine")
            engines["system"] = SystemEngine()

        if "coqui" == engine_name:
            print("Initializing coqui engine")
            engines["coqui"] = CoquiEngine()

        if "kokoro" == engine_name:
            print("Initializing kokoro engine")
            engines["kokoro"] = KokoroEngine()

        if "openai" == engine_name:
            print("Initializing openai engine")
            engines["openai"] = OpenAIEngine()

    for _engine in engines.keys():
        print(f"Retrieving voices for TTS Engine {_engine}")
        try:
            voices[_engine] = engines[_engine].get_voices()
        except Exception as e:
            voices[_engine] = []
            logging.error(f"Error retrieving voices for {_engine}: {str(e)}")

    _set_engine(START_ENGINE)

    print("Server ready")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
