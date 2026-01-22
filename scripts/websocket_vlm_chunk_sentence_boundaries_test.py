import asyncio
import websockets
import json
import time
import base64
import io
import wave
import numpy as np
import sounddevice as sd
import uvloop
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


WS_URL = "ws://localhost:8000/ws"  # Adjust port if needed
CHUNK_DELAY = 0  # Zero delay for minimum latency
THINKING_BUDGET = 0
LLM_PROMPT = "Caption this image in detail and tell 3 facts"
IMAGE_PATH = "/Users/robin/Downloads/images/Kodak Charmera Image.jpg"  # Set your image path here
IMAGE_MIME_TYPE = "image/jpeg"
MIN_CHUNK_SIZE = 15  # Minimum characters before sending (very aggressive)

def play_wav_bytes(wav_bytes):
    with io.BytesIO(wav_bytes) as wav_io:
        with wave.open(wav_io, 'rb') as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
            dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sampwidth]
            audio = np.frombuffer(frames, dtype=dtype)
            if channels > 1:
                audio = audio.reshape(-1, channels)
            sd.play(audio, samplerate=sample_rate)
            sd.wait()

class PhraseMetrics:
    def __init__(self, phrase):
        self.phrase = phrase
        self.send_time = None
        self.first_chunk_time = None
        self.end_time = None
        self.chunks = 0
        self.chunk_times = []
        self.total_bytes = 0

    def report(self):
        print(f"\n--- Metrics for: '{self.phrase}' ---")
        if self.first_chunk_time and self.send_time:
            print(f"  First chunk latency (send→recv): {self.first_chunk_time - self.send_time:.3f} s")
        if self.send_time and self.end_time:
            print(f"  Total time (send→final): {self.end_time - self.send_time:.3f} s")
        print(f"  Chunks received: {self.chunks}")
        print(f"  Total bytes: {self.total_bytes}")
        if self.chunk_times:
            print(f"  Per-chunk times (s since send): {[round(t,3) for t in self.chunk_times]}")


# Producer: streams LLM output chunks and enqueues them, with image support
async def llm_chunk_producer(chunk_queue, prompt, model="gemini-2.0-flash"):
    client = genai.Client()
    print(f"\n[Loading image: {IMAGE_PATH}]", flush=True)
    # Load image bytes
    with open(IMAGE_PATH, 'rb') as f:
        image_bytes = f.read()

    print(f"[Image loaded: {len(image_bytes)} bytes]", flush=True)
    print(f"[Sending request to {model}...]", flush=True)

    contents = [
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=IMAGE_MIME_TYPE,
        ),
        prompt
    ]

    request_start = time.perf_counter()
    response = await client.aio.models.generate_content_stream(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
        ),
    )

    first_chunk = True
    async for chunk in response:
        text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        if text.strip():
            if first_chunk:
                first_token_latency = time.perf_counter() - request_start
                print(f"\n🚀 FIRST LLM TOKEN LATENCY: {first_token_latency:.3f}s\n", flush=True)
                first_chunk = False
            await chunk_queue.put(text)

    print(f"\n[LLM streaming complete]", flush=True)

# Aggressive phrase/sentence detector: optimized for minimal latency
async def sentence_boundary_detector(llm_chunk_queue, sentence_queue):
    """
    Ultra-low latency chunking: sends phrases/sentences ASAP while maintaining natural breaks.
    Detects: periods, exclamation, question marks, commas, semicolons, colons, and paragraph breaks.
    """
    buffer = ""

    def find_phrase_break(text, min_size=MIN_CHUNK_SIZE):
        """
        Find the earliest natural break point in text.
        Priority: .!? > , ; : > newlines > word boundaries
        """
        if len(text) < min_size:
            return None

        # High priority: sentence endings
        for i, char in enumerate(text):
            if i < min_size:
                continue

            if char in '.!?':
                # Found sentence end - check if there's more content after
                rest = text[i+1:]
                if not rest.strip():  # Nothing after, wait for more
                    return None

                # Skip whitespace after punctuation
                j = 0
                while j < len(rest) and rest[j] in ' \t"\'\n':
                    j += 1

                # If we have content after punctuation, this is a good break
                if j < len(rest):
                    return i + 1

        # Medium priority: phrase breaks (comma, semicolon, colon)
        for i, char in enumerate(text):
            if i < min_size:
                continue

            if char in ',;:':
                rest = text[i+1:]
                # Make sure there's content after the punctuation
                if rest.strip() and len(rest.strip()) >= 3:
                    return i + 1

        # Low priority: newlines (but only if buffer is getting large)
        if len(text) > min_size * 2:
            newline_pos = text.find('\n', min_size)
            if newline_pos != -1 and newline_pos < len(text) - 2:
                return newline_pos + 1

        # Fallback: if buffer is very large, break at word boundary
        if len(text) > min_size * 3:
            # Find last space in reasonable range
            search_end = min(len(text), min_size * 3)
            last_space = text.rfind(' ', min_size, search_end)
            if last_space != -1:
                return last_space + 1

        return None

    while True:
        chunk = await llm_chunk_queue.get()

        if chunk is None:
            # End of stream - send any remaining text
            if buffer.strip():
                final_text = buffer.strip()
                print(f"\n[FINAL CHUNK]: {final_text}\n", flush=True)
                await sentence_queue.put(final_text)
            await sentence_queue.put(None)
            break

        print(f"\n[LLM CHUNK]: {repr(chunk[:60])}..." if len(chunk) > 60 else f"\n[LLM CHUNK]: {repr(chunk)}", flush=True)
        buffer += chunk

        # Aggressively extract phrases from buffer
        while True:
            break_pos = find_phrase_break(buffer)

            if break_pos is not None:
                # Extract the phrase/sentence
                phrase = buffer[:break_pos].strip()
                buffer = buffer[break_pos:].lstrip()

                if phrase:
                    print(f"[SENDING TO TTS]: {phrase[:80]}..." if len(phrase) > 80 else f"[SENDING TO TTS]: {phrase}", flush=True)
                    await sentence_queue.put(phrase)
            else:
                # No break point yet
                print(f"[BUFFERING ({len(buffer)} chars)]", flush=True)
                break

# Consumer: dequeues chunks and sends to websocket (optimized for minimal latency)
async def sender_llm_chunks(chunk_queue, websocket, phrase_queues, metrics_list, send_times):
    i = 0
    while True:
        chunk = await chunk_queue.get()
        if chunk is None:
            break

        # Apply delay only if configured (0 = no delay for max speed)
        if i > 0 and CHUNK_DELAY > 0:
            await asyncio.sleep(CHUNK_DELAY)

        metrics = PhraseMetrics(chunk)
        metrics_list.append(metrics)
        phrase_queues[chunk] = asyncio.Queue()
        send_time = time.perf_counter()
        metrics.send_time = send_time
        send_times[chunk] = send_time

        # Send immediately
        await websocket.send(chunk)
        print(f"[SENT #{i+1} at {send_time:.3f}]", flush=True)
        i += 1

async def receiver(websocket, phrase_queues, metrics_list, send_times, audio_queue=None, play_audio=True):
    current_phrase_idx = 0
    complete_wav = b""
    sample_rate = 24000
    first_chunk = True
    # Wait for metrics to be available for the first phrase
    while len(metrics_list) <= current_phrase_idx:
        await asyncio.sleep(0.001)
    metrics = metrics_list[current_phrase_idx]
    while True:
        try:
            response = await websocket.recv()
        except websockets.ConnectionClosed:
            break
        now = time.perf_counter()
        data = json.loads(response)
        phrase = metrics.phrase
        if "audioOutput" in data:
            if first_chunk:
                metrics.first_chunk_time = now
                latency = metrics.first_chunk_time - metrics.send_time
                print(f"Phrase: '{phrase}' - First chunk latency (send→recv): {latency:.3f} s")
                first_chunk = False
            metrics.chunks += 1
            metrics.chunk_times.append(now - metrics.send_time)
            audio_b64 = data["audioOutput"]["audio"]
            audio_bytes = base64.b64decode(audio_b64)
            metrics.total_bytes += len(audio_bytes)
            complete_wav += audio_bytes
            sample_rate = data["audioOutput"].get("sampleRate", 24000)
        elif "finalOutput" in data:
            metrics.end_time = now
            if play_audio and audio_queue is not None:
                await audio_queue.put(complete_wav)
            # Move to next phrase
            current_phrase_idx += 1
            if current_phrase_idx >= len(metrics_list):
                break
            # Wait for metrics to be available for the next phrase
            while len(metrics_list) <= current_phrase_idx:
                await asyncio.sleep(0.001)
            metrics = metrics_list[current_phrase_idx]
            complete_wav = b""
            first_chunk = True
        elif "error" in data:
            print(f"Error from server for '{phrase}':", data["error"])
            metrics.end_time = now
            # Move to next phrase
            current_phrase_idx += 1
            if current_phrase_idx >= len(metrics_list):
                break
            # Wait for metrics to be available for the next phrase
            while len(metrics_list) <= current_phrase_idx:
                await asyncio.sleep(0.001)
            metrics = metrics_list[current_phrase_idx]
            complete_wav = b""
            first_chunk = True

async def audio_player(audio_queue):
    while True:
        wav_bytes = await audio_queue.get()
        if wav_bytes is None:
            break
        await asyncio.to_thread(play_wav_bytes, wav_bytes)

async def main():
    print("=" * 80)
    print("ULTRA-LOW LATENCY MODE - Optimized for speed")
    print(f"Configuration: CHUNK_DELAY={CHUNK_DELAY}s, MIN_CHUNK_SIZE={MIN_CHUNK_SIZE} chars")
    print("=" * 80)
    print("\nUsing uvloop for improved performance.")
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    start_time = time.perf_counter()
    metrics_list = []
    phrase_queues = {}
    send_times = {}
    audio_queue = asyncio.Queue()
    chunk_queue = asyncio.Queue()
    sentence_queue = asyncio.Queue()

    async with websockets.connect(WS_URL) as websocket:
        # Pipeline: LLM → Phrase Detector (aggressive) → TTS
        # Optimized for minimal latency while maintaining natural speech breaks
        producer_task = asyncio.create_task(llm_chunk_producer(chunk_queue, LLM_PROMPT))
        detector_task = asyncio.create_task(sentence_boundary_detector(chunk_queue, sentence_queue))
        send_task = asyncio.create_task(sender_llm_chunks(sentence_queue, websocket, phrase_queues, metrics_list, send_times))
        recv_task = asyncio.create_task(receiver(websocket, phrase_queues, metrics_list, send_times, audio_queue=audio_queue))
        audio_task = asyncio.create_task(audio_player(audio_queue))

        await producer_task
        await chunk_queue.put(None)
        await detector_task
        await send_task
        await recv_task
        await websocket.send("stop")
        await audio_queue.put(None)
        await audio_task

    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    print(f"TOTAL PIPELINE TIME: {total_time:.3f}s")
    print("=" * 80)
    print("\n==== Per-Phrase Latency Metrics ====")
    for i, metrics in enumerate(metrics_list, 1):
        print(f"\n--- Phrase {i} ---")
        metrics.report()

if __name__ == "__main__":
    asyncio.run(main())