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
CHUNK_DELAY = 0.1
THINKING_BUDGET = 0
LLM_PROMPT = "Tell me a story in 300 words. send 'Start' as the first chunk, then continue sending the story in short chunks until complete."

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

# Producer: streams LLM output chunks and enqueues them
async def llm_chunk_producer(chunk_queue, prompt, model="gemini-2.0-flash"):
    client = genai.Client()
    response = await client.aio.models.generate_content_stream(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
            # Turn off thinking:
            # thinking_config=types.ThinkingConfig(thinking_budget=0)
            # Turn on dynamic thinking:
            # thinking_config=types.ThinkingConfig(thinking_budget=-1)
    ),
    )
    async for chunk in response:
        text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        if text.strip():
            await chunk_queue.put(text)

# Consumer: dequeues chunks and sends to websocket
async def sender_llm_chunks(chunk_queue, websocket, phrase_queues, metrics_list, send_times):
    i = 0
    while True:
        chunk = await chunk_queue.get()
        if chunk is None:
            break
        if i > 0:
            await asyncio.sleep(CHUNK_DELAY)
        metrics = PhraseMetrics(chunk)
        metrics_list.append(metrics)
        phrase_queues[chunk] = asyncio.Queue()
        send_time = time.perf_counter()
        metrics.send_time = send_time
        send_times[chunk] = send_time
        await websocket.send(chunk)
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
    print("Using uvloop for improved performance.")
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    metrics_list = []
    phrase_queues = {}
    send_times = {}
    audio_queue = asyncio.Queue()
    chunk_queue = asyncio.Queue()
    async with websockets.connect(WS_URL) as websocket:
        producer_task = asyncio.create_task(llm_chunk_producer(chunk_queue, LLM_PROMPT))
        send_task = asyncio.create_task(sender_llm_chunks(chunk_queue, websocket, phrase_queues, metrics_list, send_times))
        recv_task = asyncio.create_task(receiver(websocket, phrase_queues, metrics_list, send_times, audio_queue=audio_queue))
        audio_task = asyncio.create_task(audio_player(audio_queue))
        await producer_task
        await chunk_queue.put(None)
        await send_task
        await recv_task
        await websocket.send("stop")
        await audio_queue.put(None)
        await audio_task

    print("\n==== Latency Metrics Summary ====")
    for metrics in metrics_list:
        metrics.report()

if __name__ == "__main__":
    asyncio.run(main())