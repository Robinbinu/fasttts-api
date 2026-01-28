import asyncio
import websockets
import json
import time
import base64
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
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
MIN_CHUNK_SIZE = 15
VLM_MODEL = "gemini-2.0-flash"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    print("Starting up with uvloop event loop policy...")
    loop = asyncio.get_event_loop()
    print(f"Event loop type: {type(loop)}")
    yield
    # Shutdown
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


def find_phrase_break(text, min_size=MIN_CHUNK_SIZE):
    """Find the earliest natural break point in text"""
    if len(text) < min_size:
        return None

    # High priority: sentence endings
    for i, char in enumerate(text):
        if i < min_size:
            continue

        if char in '.!?':
            rest = text[i+1:]
            if not rest.strip():
                return None

            j = 0
            while j < len(rest) and rest[j] in ' \t"\'\n':
                j += 1

            if j < len(rest):
                return i + 1

    # Medium priority: phrase breaks
    for i, char in enumerate(text):
        if i < min_size:
            continue

        if char in ',;:':
            rest = text[i+1:]
            if rest.strip() and len(rest.strip()) >= 3:
                return i + 1

    # Low priority: newlines
    if len(text) > min_size * 2:
        newline_pos = text.find('\n', min_size)
        if newline_pos != -1 and newline_pos < len(text) - 2:
            return newline_pos + 1

    # Fallback: word boundary
    if len(text) > min_size * 3:
        search_end = min(len(text), min_size * 3)
        last_space = text.rfind(' ', min_size, search_end)
        if last_space != -1:
            return last_space + 1

    return None


async def process_vlm_to_tts(image_bytes: bytes, prompt: str, mime_type: str = "image/jpeg"):
    """Process image through VLM and stream to TTS"""
    metrics = PipelineMetrics()
    audio_chunks = []

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

    # Connect to TTS websocket
    try:
        websocket = await websockets.connect(TTS_WS_URL)
    except Exception as e:
        return {"error": f"Failed to connect to TTS server: {e}"}, None

    try:
        # Request LLM
        metrics.llm_request_start = time.perf_counter()

        response = await client.aio.models.generate_content_stream(
            model=VLM_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
            ),
        )

        # Process LLM chunks and detect phrase boundaries
        buffer = ""
        phrase_count = 0
        first_chunk = True
        phrase_data = []

        async for chunk in response:
            text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            if text.strip():
                if first_chunk:
                    metrics.first_llm_token = time.perf_counter()
                    first_chunk = False

                metrics.total_llm_chunks += 1
                buffer += text

                # Extract phrases
                while True:
                    break_pos = find_phrase_break(buffer)

                    if break_pos is not None:
                        phrase = buffer[:break_pos].strip()
                        buffer = buffer[break_pos:].lstrip()

                        if phrase:
                            phrase_count += 1
                            metrics.total_phrases += 1

                            if phrase_count == 1:
                                metrics.first_phrase_sent = time.perf_counter()

                            # Send to TTS
                            send_time = time.perf_counter()
                            await websocket.send(phrase)

                            phrase_data.append({
                                'phrase': phrase,
                                'send_time': send_time,
                                'audio_bytes': 0
                            })
                    else:
                        break

        # Send final phrase if any
        if buffer.strip():
            phrase_count += 1
            metrics.total_phrases += 1
            send_time = time.perf_counter()
            await websocket.send(buffer.strip())
            phrase_data.append({
                'phrase': buffer.strip(),
                'send_time': send_time,
                'audio_bytes': 0
            })

        metrics.llm_complete = time.perf_counter()

        # Receive audio chunks
        current_idx = 0
        complete_wav = b""
        first_audio = True

        while current_idx < len(phrase_data):
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            except asyncio.TimeoutError:
                break

            now = time.perf_counter()

            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                continue

            if "audioOutput" in data:
                if first_audio:
                    metrics.first_audio_received = now
                    first_audio = False

                audio_b64 = data["audioOutput"]["audio"]
                audio_bytes = base64.b64decode(audio_b64)
                phrase_data[current_idx]['audio_bytes'] += len(audio_bytes)
                metrics.total_audio_bytes += len(audio_bytes)
                complete_wav += audio_bytes

            elif "finalOutput" in data:
                latency = (now - phrase_data[current_idx]['send_time']) * 1000
                metrics.phrase_metrics.append({
                    'phrase_num': current_idx + 1,
                    'latency': latency
                })

                # Store complete audio chunk
                audio_chunks.append({
                    'phrase': phrase_data[current_idx]['phrase'],
                    'audio': base64.b64encode(complete_wav).decode('utf-8'),
                    'latency_ms': latency
                })

                current_idx += 1
                complete_wav = b""

            elif "error" in data:
                current_idx += 1

        # Send stop signal
        await websocket.send("stop")

    finally:
        await websocket.close()

    return audio_chunks, metrics.to_dict()


@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the web interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VLM to TTS Pipeline</title>
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
            margin-bottom: 30px;
            font-size: 2.5em;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>VLM to TTS Pipeline</h1>

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
            <h2>Generated Speech</h2>
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
        });

        processBtn.addEventListener('click', async () => {
            if (!capturedImageData) return;

            processBtn.disabled = true;
            showStatus('Processing image through VLM → TTS pipeline...', 'processing');
            metricsCard.classList.add('hidden');
            phrasesCard.classList.add('hidden');

            try {
                // Convert base64 to blob
                const response = await fetch(capturedImageData);
                const blob = await response.blob();

                // Create form data
                const formData = new FormData();
                formData.append('image', blob, 'captured.jpg');
                formData.append('prompt', promptInput.value);

                // Send to backend
                const result = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });

                const data = await result.json();

                if (data.error) {
                    showStatus('Error: ' + data.error, 'error');
                    processBtn.disabled = false;
                    return;
                }

                showStatus('Processing complete!', 'success');

                // Display metrics
                displayMetrics(data.metrics);

                // Display phrases and play audio
                displayPhrases(data.audio_chunks);

            } catch (err) {
                showStatus('Error processing image: ' + err.message, 'error');
            } finally {
                processBtn.disabled = false;
            }
        });

        function displayMetrics(metrics) {
            metricsGrid.innerHTML = '';

            const metricItems = [
                { label: 'Image to First Audio', value: metrics.image_to_audio_ms, unit: 'ms' },
                { label: 'First LLM Token', value: metrics.first_token_ms, unit: 'ms' },
                { label: 'Total LLM Time', value: metrics.llm_total_ms, unit: 'ms' },
                { label: 'Total Phrases', value: metrics.total_phrases, unit: '' },
                { label: 'Total Audio Data', value: (metrics.total_audio_bytes / 1024).toFixed(2), unit: 'KB' },
                { label: 'Avg Phrase Latency', value: metrics.avg_phrase_latency_ms, unit: 'ms' }
            ];

            metricItems.forEach(item => {
                if (item.value !== undefined && item.value !== null) {
                    const metricCard = document.createElement('div');
                    metricCard.className = 'metric-card';
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

        function displayPhrases(audioChunks) {
            phrasesList.innerHTML = '';

            audioChunks.forEach((chunk, index) => {
                const phraseItem = document.createElement('div');
                phraseItem.className = 'phrase-item';
                phraseItem.id = `phrase-${index}`;

                const phraseText = document.createElement('div');
                phraseText.className = 'phrase-text';
                phraseText.textContent = chunk.phrase;

                const phraseLatency = document.createElement('div');
                phraseLatency.className = 'phrase-latency';
                phraseLatency.textContent = `Latency: ${chunk.latency_ms.toFixed(1)}ms`;

                const playButton = document.createElement('button');
                playButton.className = 'btn-primary';
                playButton.textContent = 'Play';
                playButton.style.marginTop = '10px';
                playButton.onclick = () => playAudio(chunk.audio, index);

                phraseItem.appendChild(phraseText);
                phraseItem.appendChild(phraseLatency);
                phraseItem.appendChild(playButton);
                phrasesList.appendChild(phraseItem);
            });

            phrasesCard.classList.remove('hidden');

            // Auto-play all audio chunks sequentially
            if (audioChunks.length > 0) {
                playAllAudio(audioChunks);
            }
        }

        function playAudio(base64Audio, phraseIndex) {
            return new Promise((resolve, reject) => {
                const audio = new Audio('data:audio/wav;base64,' + base64Audio);

                // Highlight current phrase
                if (phraseIndex !== undefined) {
                    const phraseItem = document.getElementById(`phrase-${phraseIndex}`);
                    if (phraseItem) {
                        phraseItem.style.borderLeft = '4px solid #667eea';
                        phraseItem.style.backgroundColor = '#edf2f7';
                    }
                }

                audio.onended = () => {
                    // Reset highlight
                    if (phraseIndex !== undefined) {
                        const phraseItem = document.getElementById(`phrase-${phraseIndex}`);
                        if (phraseItem) {
                            phraseItem.style.borderLeft = '4px solid #48bb78';
                            phraseItem.style.backgroundColor = '#f7fafc';
                        }
                    }
                    resolve();
                };

                audio.onerror = (error) => {
                    reject(error);
                };

                audio.play();
            });
        }

        async function playAllAudio(audioChunks) {
            for (let i = 0; i < audioChunks.length; i++) {
                try {
                    await playAudio(audioChunks[i].audio, i);
                } catch (error) {
                    console.error(`Error playing audio chunk ${i}:`, error);
                }
            }
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post("/process")
async def process_image(image: UploadFile = File(...)):
    """Process uploaded image through VLM → TTS pipeline"""
    from fastapi import Form

    try:
        # Read image bytes
        image_bytes = await image.read()

        # Get prompt from form data (default if not provided)
        # Note: In the actual request, we need to handle form data
        prompt = "What do you see?"

        # Process through pipeline
        audio_chunks, metrics = await process_vlm_to_tts(
            image_bytes=image_bytes,
            prompt=prompt,
            mime_type=image.content_type or "image/jpeg"
        )

        if isinstance(audio_chunks, dict) and 'error' in audio_chunks:
            return JSONResponse(content=audio_chunks, status_code=500)

        return JSONResponse(content={
            "audio_chunks": audio_chunks,
            "metrics": metrics
        })

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


# Updated process endpoint to handle prompt
from fastapi import Form as FastAPIForm

@app.post("/process")
async def process_image_with_prompt(
    image: UploadFile = File(...),
    prompt: str = FastAPIForm(default="What do you see?")
):
    """Process uploaded image through VLM → TTS pipeline with custom prompt"""
    try:
        # Read image bytes
        image_bytes = await image.read()

        # Process through pipeline
        audio_chunks, metrics = await process_vlm_to_tts(
            image_bytes=image_bytes,
            prompt=prompt,
            mime_type=image.content_type or "image/jpeg"
        )

        if isinstance(audio_chunks, dict) and 'error' in audio_chunks:
            return JSONResponse(content=audio_chunks, status_code=500)

        return JSONResponse(content={
            "audio_chunks": audio_chunks,
            "metrics": metrics
        })

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


if __name__ == "__main__":
    print("Starting VLM → TTS Web Interface...")
    print("Open http://localhost:8001 in your browser")
    print("Make sure the TTS server is running on ws://localhost:8000/ws")
    uvicorn.run(app, host="localhost", port=8001, loop="uvloop")
