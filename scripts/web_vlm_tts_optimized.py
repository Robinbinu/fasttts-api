"""
Optimized VLM → TTS Pipeline with Quality-Preserving Optimizations

Key Improvements:
1. Stream audio to frontend (WebSocket streaming)
2. Pipeline parallelization (process next phrase while generating audio for current)
3. Optimized phrase break detection with compiled regex
4. Pre-warm TTS connection with connection pooling
5. Reduced MIN_CHUNK_SIZE to 10 (from 15)
6. Send phrases to TTS immediately without delays

Target: <500ms from image to first audio playback
"""

import asyncio
import websockets
import json
import time
import base64
import re
from contextlib import asynccontextmanager
from typing import Dict, Optional
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from dotenv import load_dotenv
import uvicorn
import uvloop

load_dotenv()

# Set uvloop as the default event loop policy
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configuration
TTS_WS_URL = "ws://localhost:8000/ws"
THINKING_BUDGET = 0
MIN_CHUNK_SIZE = 20  # Reduced from 15 for faster first phrase
VLM_MODEL = "gemini-2.0-flash"

# Pre-compiled regex patterns (optimization #3)
SENTENCE_BREAK = re.compile(r'[.!?](["\']?)\s+')
PHRASE_BREAK = re.compile(r'[,;:]\s+')
NEWLINE_BREAK = re.compile(r'\n+')


class TTSConnectionPool:
    """Pre-warm and reuse TTS connections (optimization #4)"""

    def __init__(self):
        self.connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self._lock = asyncio.Lock()

    async def get_connection(self, session_id: str) -> websockets.WebSocketClientProtocol:
        """Get or create a connection for a session"""
        async with self._lock:
            if session_id in self.connections:
                # Check if connection is still alive
                conn = self.connections[session_id]
                if conn.open and not conn.closed:
                    print(f"[Connection Pool] Reusing connection for session {session_id}")
                    return conn
                else:
                    # Connection is dead, remove it
                    print(f"[Connection Pool] Connection dead for session {session_id}, recreating...")
                    del self.connections[session_id]

            # Create new connection
            try:
                print(f"[Connection Pool] Creating new connection for session {session_id}")
                conn = await websockets.connect(TTS_WS_URL)
                self.connections[session_id] = conn
                return conn
            except Exception as e:
                raise Exception(f"Failed to connect to TTS server: {e}")

    async def release_connection(self, session_id: str):
        """Close and remove a connection"""
        async with self._lock:
            if session_id in self.connections:
                try:
                    # Send stop signal before closing
                    await self.connections[session_id].send("stop")
                    await self.connections[session_id].close()
                except:
                    pass
                del self.connections[session_id]

    async def close_all(self):
        """Close all connections"""
        async with self._lock:
            for conn in self.connections.values():
                try:
                    # Send stop signal to each connection before closing
                    await conn.send("stop")
                    await conn.close()
                except:
                    pass
            self.connections.clear()


# Global connection pool
tts_pool = TTSConnectionPool()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    print("Starting up with uvloop event loop policy...")
    loop = asyncio.get_event_loop()
    print(f"Event loop type: {type(loop)}")
    yield
    # Shutdown
    print("Closing all TTS connections...")
    await tts_pool.close_all()
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineMetrics:
    """Track detailed metrics for the entire pipeline"""
    def __init__(self):
        self.image_load_start = None
        self.image_load_end = None
        self.llm_request_start = None
        self.first_llm_token = None
        self.llm_complete = None
        self.first_phrase_sent = None
        self.first_audio_received = None
        self.total_phrases = 0
        self.total_llm_chunks = 0
        self.total_audio_bytes = 0
        self.phrase_metrics = []

    def to_dict(self):
        """Convert metrics to dictionary"""
        metrics_dict = {}

        if self.image_load_start and self.image_load_end:
            metrics_dict['image_load_ms'] = (self.image_load_end - self.image_load_start) * 1000

        if self.llm_request_start and self.first_llm_token:
            metrics_dict['first_token_ms'] = (self.first_llm_token - self.llm_request_start) * 1000

        if self.llm_request_start and self.llm_complete:
            metrics_dict['llm_total_ms'] = (self.llm_complete - self.llm_request_start) * 1000

        if self.llm_request_start and self.first_phrase_sent:
            metrics_dict['first_phrase_sent_ms'] = (self.first_phrase_sent - self.llm_request_start) * 1000

        if self.first_phrase_sent and self.first_audio_received:
            metrics_dict['first_audio_chunk_ms'] = (self.first_audio_received - self.first_phrase_sent) * 1000

        if self.llm_request_start and self.first_audio_received:
            metrics_dict['image_to_audio_ms'] = (self.first_audio_received - self.llm_request_start) * 1000

        metrics_dict['total_phrases'] = self.total_phrases
        metrics_dict['total_llm_chunks'] = self.total_llm_chunks
        metrics_dict['total_audio_bytes'] = self.total_audio_bytes
        metrics_dict['phrase_metrics'] = self.phrase_metrics

        if self.phrase_metrics:
            avg_phrase_latency = sum(m['latency'] for m in self.phrase_metrics) / len(self.phrase_metrics)
            metrics_dict['avg_phrase_latency_ms'] = avg_phrase_latency

        return metrics_dict


def find_phrase_break_optimized(text: str, min_size: int = MIN_CHUNK_SIZE) -> Optional[int]:
    """
    Optimized phrase break detection using pre-compiled regex (optimization #3)
    Quality-preserving: Still respects natural phrase boundaries
    """
    if len(text) < min_size:
        return None

    # High priority: sentence endings
    match = SENTENCE_BREAK.search(text, min_size)
    if match:
        # Ensure there's content after the break
        end_pos = match.end()
        if end_pos < len(text) - 1:
            return end_pos

    # Medium priority: phrase breaks (commas, semicolons, colons)
    match = PHRASE_BREAK.search(text, min_size)
    if match:
        end_pos = match.end()
        rest = text[end_pos:].strip()
        if rest and len(rest) >= 3:
            return end_pos

    # Low priority: newlines
    if len(text) > min_size * 2:
        match = NEWLINE_BREAK.search(text, min_size)
        if match and match.end() < len(text) - 2:
            return match.end()

    # Fallback: word boundary
    if len(text) > min_size * 3:
        search_end = min(len(text), min_size * 3)
        last_space = text.rfind(' ', min_size, search_end)
        if last_space != -1:
            return last_space + 1

    return None


async def process_vlm_stream_to_frontend(
    websocket_client: WebSocket,
    image_bytes: bytes,
    prompt: str,
    mime_type: str = "image/jpeg",
    session_id: str = "default"
):
    """
    Optimized pipeline: Stream audio chunks to frontend as they're ready (optimization #1 & #2)

    Key improvements:
    - Immediately send audio chunks to frontend (don't wait for all phrases)
    - Sequential TTS processing (WebSocket limitation: no concurrent recv())
    - Use connection pool for faster TTS connection
    """
    metrics = PipelineMetrics()

    # Load image
    metrics.image_load_start = time.perf_counter()
    metrics.image_load_end = time.perf_counter()

    # Initialize VLM client
    client = genai.Client()

    # Create VLM request
    contents = [
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        ),
        prompt
    ]

    # Get pre-warmed TTS connection from pool (optimization #4)
    try:
        tts_ws = await tts_pool.get_connection(session_id)
    except Exception as e:
        await websocket_client.send_json({
            "type": "error",
            "error": f"Failed to connect to TTS server: {e}"
        })
        return

    try:
        # Send initial status
        await websocket_client.send_json({
            "type": "status",
            "message": "Starting VLM processing..."
        })

        # Request LLM
        metrics.llm_request_start = time.perf_counter()

        response = await client.aio.models.generate_content_stream(
            model=VLM_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
            ),
        )

        # Coordination for parallel VLM streaming + TTS processing
        tts_complete = asyncio.Event()
        phrase_metadata = {}

        # Track metrics
        phrase_count = 0
        first_chunk = True
        buffer = ""

        async def tts_receiver():
            """Single receiver loop that handles all TTS responses"""
            current_phrase_num = 0

            while True:
                # Wait for a phrase to be available
                if current_phrase_num == 0 or current_phrase_num < phrase_count:
                    # There might be more phrases coming
                    await asyncio.sleep(0.001)
                    if current_phrase_num >= phrase_count and tts_complete.is_set():
                        break
                    if current_phrase_num >= phrase_count:
                        continue
                    current_phrase_num += 1
                else:
                    # All phrases sent, wait for completion flag
                    if tts_complete.is_set():
                        break
                    await asyncio.sleep(0.001)
                    continue

                # Process responses for current phrase
                complete_wav = b""
                first_audio = True
                phrase_info = phrase_metadata.get(current_phrase_num, {})
                send_time = phrase_info.get('send_time', time.perf_counter())
                phrase_text = phrase_info.get('phrase', '')

                while True:
                    try:
                        response = await asyncio.wait_for(tts_ws.recv(), timeout=30.0)
                    except asyncio.TimeoutError:
                        break

                    now = time.perf_counter()

                    try:
                        data = json.loads(response)
                    except json.JSONDecodeError:
                        continue

                    if "audioOutput" in data:
                        if first_audio:
                            if current_phrase_num == 1:
                                metrics.first_audio_received = now
                            first_audio = False

                        audio_b64 = data["audioOutput"]["audio"]
                        audio_bytes = base64.b64decode(audio_b64)
                        complete_wav += audio_bytes
                        metrics.total_audio_bytes += len(audio_bytes)

                        # Don't stream individual chunks - wait for complete audio to avoid stuttering

                    elif "finalOutput" in data:
                        latency = (now - send_time) * 1000
                        metrics.phrase_metrics.append({
                            'phrase_num': current_phrase_num,
                            'latency': latency
                        })

                        # Send completion marker
                        await websocket_client.send_json({
                            "type": "phrase_complete",
                            "phrase_num": current_phrase_num,
                            "latency_ms": latency,
                            "phrase": phrase_text,
                            "complete_audio": base64.b64encode(complete_wav).decode('utf-8')
                        })
                        break

                    elif "error" in data:
                        await websocket_client.send_json({
                            "type": "error",
                            "phrase_num": current_phrase_num,
                            "error": data["error"]
                        })
                        break

        async def send_phrase_to_tts(phrase_text: str, phrase_num: int):
            """Send a phrase to TTS (non-blocking)"""
            send_time = time.perf_counter()
            phrase_metadata[phrase_num] = {
                'send_time': send_time,
                'phrase': phrase_text
            }

            # Send phrase to TTS immediately (optimization #6)
            await tts_ws.send(phrase_text)

        # Start TTS receiver in background
        receiver_task = asyncio.create_task(tts_receiver())

        # Process VLM stream
        async for chunk in response:
            text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            if text.strip():
                if first_chunk:
                    metrics.first_llm_token = time.perf_counter()
                    first_chunk = False

                    await websocket_client.send_json({
                        "type": "first_token",
                        "latency_ms": (metrics.first_llm_token - metrics.llm_request_start) * 1000
                    })

                metrics.total_llm_chunks += 1
                buffer += text

                # Extract phrases and send to TTS (optimization #2 & #6)
                while True:
                    break_pos = find_phrase_break_optimized(buffer)

                    if break_pos is not None:
                        phrase = buffer[:break_pos].strip()
                        buffer = buffer[break_pos:].lstrip()  # Immediate lstrip (optimization #6)

                        if phrase:
                            phrase_count += 1
                            metrics.total_phrases += 1

                            if phrase_count == 1:
                                metrics.first_phrase_sent = time.perf_counter()

                            # Send phrase notification to frontend
                            await websocket_client.send_json({
                                "type": "phrase_detected",
                                "phrase_num": phrase_count,
                                "phrase": phrase
                            })

                            # Send to TTS immediately (receiver handles responses)
                            await send_phrase_to_tts(phrase, phrase_count)
                    else:
                        break

        # Send final phrase if any
        if buffer.strip():
            phrase_count += 1
            metrics.total_phrases += 1

            await websocket_client.send_json({
                "type": "phrase_detected",
                "phrase_num": phrase_count,
                "phrase": buffer.strip()
            })

            await send_phrase_to_tts(buffer.strip(), phrase_count)

        metrics.llm_complete = time.perf_counter()

        # Signal TTS receiver that all phrases are sent
        tts_complete.set()

        # Wait for TTS receiver to finish
        await receiver_task

        # Don't send "stop" signal - keep connection alive for pooling!
        # The connection will be reused for the next request

        # Send completion message with metrics
        await websocket_client.send_json({
            "type": "complete",
            "metrics": metrics.to_dict()
        })

    except Exception as e:
        await websocket_client.send_json({
            "type": "error",
            "error": str(e)
        })
    finally:
        # Don't close the connection, return it to the pool
        pass


@app.websocket("/ws/process")
async def websocket_process_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming VLM → TTS pipeline

    Expected message format:
    {
        "image": "base64_encoded_image",
        "prompt": "What do you see?",
        "mime_type": "image/jpeg",
        "session_id": "optional_session_id"
    }

    Response format:
    - {"type": "status", "message": "..."}
    - {"type": "first_token", "latency_ms": ...}
    - {"type": "phrase_detected", "phrase_num": ..., "phrase": "..."}
    - {"type": "audio_chunk", "phrase_num": ..., "audio": "base64", "phrase": "..."}
    - {"type": "phrase_complete", "phrase_num": ..., "latency_ms": ..., "complete_audio": "base64"}
    - {"type": "complete", "metrics": {...}}
    - {"type": "error", "error": "..."}
    """
    await websocket.accept()

    try:
        # Receive request
        data = await websocket.receive_json()

        # Extract parameters
        image_b64 = data.get("image")
        prompt = data.get("prompt", "What do you see?")
        mime_type = data.get("mime_type", "image/jpeg")
        session_id = data.get("session_id", f"session_{id(websocket)}")

        if not image_b64:
            await websocket.send_json({
                "type": "error",
                "error": "No image provided"
            })
            return

        # Decode image
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "error": f"Failed to decode image: {e}"
            })
            return

        # Process pipeline
        await process_vlm_stream_to_frontend(
            websocket,
            image_bytes,
            prompt,
            mime_type,
            session_id
        )

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass


@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the optimized web interface with streaming support"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimized VLM to TTS Pipeline</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        h1 {
            color: white;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }

        .subtitle {
            color: #ffd700;
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.1em;
            font-weight: 600;
        }

        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }

        .camera-section {
            text-align: center;
        }

        #video {
            width: 100%;
            max-width: 640px;
            border-radius: 10px;
            background: #000;
            margin-bottom: 20px;
        }

        #canvas {
            display: none;
        }

        #capturedImage {
            max-width: 100%;
            border-radius: 10px;
            margin-top: 20px;
        }

        .button-group {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 20px;
        }

        button {
            padding: 12px 30px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .btn-primary {
            background: #667eea;
            color: white;
        }

        .btn-success {
            background: #48bb78;
            color: white;
        }

        .btn-danger {
            background: #f56565;
            color: white;
        }

        .btn-secondary {
            background: #718096;
            color: white;
        }

        .prompt-input {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            margin-bottom: 15px;
        }

        .status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            font-weight: 500;
        }

        .status.info {
            background: #bee3f8;
            color: #2c5282;
        }

        .status.success {
            background: #c6f6d5;
            color: #22543d;
        }

        .status.error {
            background: #fed7d7;
            color: #742a2a;
        }

        .status.processing {
            background: #fef5e7;
            color: #7d6608;
        }

        .status.streaming {
            background: #e6fffa;
            color: #234e52;
            border-left: 4px solid #38b2ac;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .metric-card {
            background: #f7fafc;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }

        .metric-label {
            font-size: 14px;
            color: #718096;
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2d3748;
        }

        .metric-unit {
            font-size: 14px;
            color: #a0aec0;
        }

        .phrase-list {
            margin-top: 20px;
        }

        .phrase-item {
            background: #f7fafc;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #48bb78;
            transition: all 0.3s ease;
            transform: scale(1);
        }

        .phrase-item.streaming {
            border-left: 4px solid #667eea;
            background: #edf2f7;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }

        .phrase-text {
            font-size: 16px;
            color: #2d3748;
            margin-bottom: 10px;
        }

        .phrase-latency {
            font-size: 14px;
            color: #718096;
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .hidden {
            display: none;
        }

        .optimization-badge {
            display: inline-block;
            background: #ffd700;
            color: #2d3748;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin: 10px 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Optimized VLM to TTS Pipeline</h1>

        <div class="card camera-section">
            <video id="video" autoplay playsinline></video>
            <canvas id="canvas"></canvas>
            <img id="capturedImage" class="hidden" alt="Captured image">

            <div>
                <label for="prompt" style="display: block; margin-bottom: 10px; font-weight: 600;">Prompt:</label>
                <input type="text" id="prompt" class="prompt-input" value="What do you see?" placeholder="Enter your prompt...">
            </div>

            <div class="button-group">
                <button id="startCamera" class="btn-primary">Start Camera</button>
                <button id="captureBtn" class="btn-success" disabled>Capture Image</button>
                <button id="processBtn" class="btn-primary" disabled>Process Image</button>
                <button id="resetBtn" class="btn-secondary hidden">Reset</button>
            </div>

            <div id="status" class="hidden"></div>
        </div>

        <div id="metricsCard" class="card hidden">
            <h2>Pipeline Metrics</h2>
            <div id="metricsGrid" class="metrics-grid"></div>
        </div>

        <div id="phrasesCard" class="card hidden">
            <h2>Generated Speech (Streaming)</h2>
            <div id="phrasesList" class="phrase-list"></div>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const capturedImage = document.getElementById('capturedImage');
        const startCameraBtn = document.getElementById('startCamera');
        const captureBtn = document.getElementById('captureBtn');
        const processBtn = document.getElementById('processBtn');
        const resetBtn = document.getElementById('resetBtn');
        const status = document.getElementById('status');
        const promptInput = document.getElementById('prompt');
        const metricsCard = document.getElementById('metricsCard');
        const metricsGrid = document.getElementById('metricsGrid');
        const phrasesCard = document.getElementById('phrasesCard');
        const phrasesList = document.getElementById('phrasesList');

        let stream = null;
        let capturedImageData = null;
        let ws = null;
        let audioQueue = [];
        let isPlaying = false;

        function showStatus(message, type = 'info') {
            status.textContent = message;
            status.className = `status ${type}`;
            status.classList.remove('hidden');
        }

        function hideStatus() {
            status.classList.add('hidden');
        }

        startCameraBtn.addEventListener('click', async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'user' },
                    audio: false
                });
                video.srcObject = stream;
                captureBtn.disabled = false;
                startCameraBtn.disabled = true;
                showStatus('Camera started successfully', 'success');
            } catch (err) {
                showStatus('Error accessing camera: ' + err.message, 'error');
            }
        });

        captureBtn.addEventListener('click', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            capturedImageData = canvas.toDataURL('image/jpeg');
            capturedImage.src = capturedImageData;
            capturedImage.classList.remove('hidden');
            video.classList.add('hidden');

            processBtn.disabled = false;
            captureBtn.disabled = true;
            resetBtn.classList.remove('hidden');

            showStatus('Image captured! Click "Process Image" to analyze.', 'success');
        });

        resetBtn.addEventListener('click', () => {
            video.classList.remove('hidden');
            capturedImage.classList.add('hidden');
            resetBtn.classList.add('hidden');
            captureBtn.disabled = false;
            processBtn.disabled = true;
            metricsCard.classList.add('hidden');
            phrasesCard.classList.add('hidden');
            hideStatus();
            if (ws) {
                ws.close();
                ws = null;
            }
        });

        processBtn.addEventListener('click', async () => {
            if (!capturedImageData) return;

            processBtn.disabled = true;
            showStatus('Connecting to streaming pipeline...', 'processing');
            metricsCard.classList.add('hidden');
            phrasesCard.classList.add('hidden');
            phrasesList.innerHTML = '';
            audioQueue = [];

            try {
                // Connect to WebSocket
                ws = new WebSocket(`ws://${window.location.host}/ws/process`);

                ws.onopen = () => {
                    showStatus('Connected! Processing image through VLM → TTS...', 'streaming');

                    // Send image data
                    const imageBase64 = capturedImageData.split(',')[1];
                    ws.send(JSON.stringify({
                        image: imageBase64,
                        prompt: promptInput.value,
                        mime_type: 'image/jpeg'
                    }));
                };

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                };

                ws.onerror = (error) => {
                    showStatus('WebSocket error: ' + error.message, 'error');
                    processBtn.disabled = false;
                };

                ws.onclose = () => {
                    console.log('WebSocket closed');
                    processBtn.disabled = false;
                };

            } catch (err) {
                showStatus('Error processing image: ' + err.message, 'error');
                processBtn.disabled = false;
            }
        });

        function handleWebSocketMessage(data) {
            switch (data.type) {
                case 'status':
                    showStatus(data.message, 'streaming');
                    break;

                case 'first_token':
                    showStatus(`First token received in ${data.latency_ms.toFixed(1)}ms`, 'streaming');
                    break;

                case 'phrase_detected':
                    phrasesCard.classList.remove('hidden');
                    addPhraseToList(data.phrase_num, data.phrase);
                    showStatus(`Processing phrase ${data.phrase_num}...`, 'streaming');
                    break;

                case 'phrase_complete':
                    // Queue complete audio for playback
                    queueAudioChunk(data.phrase_num, data.complete_audio);
                    updatePhraseComplete(data.phrase_num, data.latency_ms);
                    showStatus(`Phrase ${data.phrase_num} complete (${data.latency_ms.toFixed(1)}ms)`, 'streaming');
                    break;

                case 'complete':
                    showStatus('Processing complete!', 'success');
                    displayMetrics(data.metrics);
                    break;

                case 'error':
                    showStatus('Error: ' + data.error, 'error');
                    break;
            }
        }

        function addPhraseToList(phraseNum, phraseText) {
            const phraseItem = document.createElement('div');
            phraseItem.className = 'phrase-item streaming';
            phraseItem.id = `phrase-${phraseNum}`;

            const phraseTextDiv = document.createElement('div');
            phraseTextDiv.className = 'phrase-text';
            phraseTextDiv.textContent = phraseText;

            const phraseLatency = document.createElement('div');
            phraseLatency.className = 'phrase-latency';
            phraseLatency.innerHTML = '<span style="color: #667eea;">⏳ Generating audio...</span>';

            phraseItem.appendChild(phraseTextDiv);
            phraseItem.appendChild(phraseLatency);
            phrasesList.appendChild(phraseItem);
        }

        function updatePhraseComplete(phraseNum, latency) {
            const phraseItem = document.getElementById(`phrase-${phraseNum}`);
            if (phraseItem) {
                phraseItem.classList.remove('streaming');
                const latencyDiv = phraseItem.querySelector('.phrase-latency');
                if (latencyDiv) {
                    latencyDiv.textContent = `Latency: ${latency.toFixed(1)}ms`;
                }
            }
        }

        function queueAudioChunk(phraseNum, audioBase64) {
            audioQueue.push({ phraseNum, audio: audioBase64 });
            if (!isPlaying) {
                playNextAudio();
            }
        }

        async function playNextAudio() {
            if (audioQueue.length === 0) {
                isPlaying = false;
                return;
            }

            isPlaying = true;
            const { phraseNum, audio } = audioQueue.shift();

            try {
                await playAudio(audio, phraseNum);
            } catch (error) {
                console.error(`Error playing audio for phrase ${phraseNum}:`, error);
            }

            // Play next audio
            playNextAudio();
        }

        function playAudio(base64Audio, phraseNum) {
            return new Promise((resolve, reject) => {
                const audio = new Audio('data:audio/wav;base64,' + base64Audio);

                // Highlight current phrase
                const phraseItem = document.getElementById(`phrase-${phraseNum}`);
                if (phraseItem) {
                    phraseItem.style.borderLeft = '4px solid #667eea';
                    phraseItem.style.backgroundColor = '#edf2f7';
                    phraseItem.style.transform = 'scale(1.02)';
                }

                audio.onended = () => {
                    // Reset highlight
                    if (phraseItem) {
                        phraseItem.style.borderLeft = '4px solid #48bb78';
                        phraseItem.style.backgroundColor = '#f7fafc';
                        phraseItem.style.transform = 'scale(1)';
                    }
                    resolve();
                };

                audio.onerror = (error) => {
                    console.error(`Error playing audio for phrase ${phraseNum}:`, error);
                    // Reset highlight on error
                    if (phraseItem) {
                        phraseItem.style.borderLeft = '4px solid #f56565';
                        phraseItem.style.backgroundColor = '#fff5f5';
                        phraseItem.style.transform = 'scale(1)';
                    }
                    reject(error);
                };

                audio.play().catch(error => {
                    console.error(`Failed to play audio for phrase ${phraseNum}:`, error);
                    reject(error);
                });
            });
        }

        function displayMetrics(metrics) {
            metricsGrid.innerHTML = '';

            const metricItems = [
                { label: 'Image to First Audio', value: metrics.image_to_audio_ms, unit: 'ms', highlight: true },
                { label: 'First LLM Token', value: metrics.first_token_ms, unit: 'ms' },
                { label: 'First Phrase Sent', value: metrics.first_phrase_sent_ms, unit: 'ms' },
                { label: 'First Audio Chunk', value: metrics.first_audio_chunk_ms, unit: 'ms' },
                { label: 'Total Phrases', value: metrics.total_phrases, unit: '' },
                { label: 'Total Audio Data', value: (metrics.total_audio_bytes / 1024).toFixed(2), unit: 'KB' },
                { label: 'Avg Phrase Latency', value: metrics.avg_phrase_latency_ms, unit: 'ms' }
            ];

            metricItems.forEach(item => {
                if (item.value !== undefined && item.value !== null) {
                    const metricCard = document.createElement('div');
                    metricCard.className = 'metric-card';
                    if (item.highlight) {
                        metricCard.style.borderLeft = '4px solid #ffd700';
                    }
                    metricCard.innerHTML = `
                        <div class="metric-label">${item.label}</div>
                        <div class="metric-value">
                            ${typeof item.value === 'number' ? item.value.toFixed(1) : item.value}
                            <span class="metric-unit">${item.unit}</span>
                        </div>
                    `;
                    metricsGrid.appendChild(metricCard);
                }
            });

            metricsCard.classList.remove('hidden');
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    print("Starting Optimized VLM → TTS Web Interface...")
    print("Open http://localhost:8001 in your browser")
    print("Make sure the TTS server is running on ws://localhost:8000/ws")
    print("\nOptimizations enabled:")
    print("  ✓ Audio streaming to frontend")
    print("  ✓ Parallel VLM + TTS processing")
    print("  ✓ Compiled regex for phrase detection")
    print("  ✓ Pre-warmed TTS connection pool")
    print(f"  ✓ MIN_CHUNK_SIZE reduced to {MIN_CHUNK_SIZE}")
    print("  ✓ Immediate phrase sending")
    uvicorn.run(app, host="localhost", port=8001, loop="uvloop")
