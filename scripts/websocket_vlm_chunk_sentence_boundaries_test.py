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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

load_dotenv()

# Configuration
WS_URL = "ws://localhost:8000/ws"
CHUNK_DELAY = 0  # Zero delay for minimum latency
THINKING_BUDGET = 0
LLM_PROMPT = "What do see?"
IMAGE_PATH = "/Users/robin/Downloads/images/Kodak Charmera Image.jpg"
IMAGE_MIME_TYPE = "image/jpeg"
MIN_CHUNK_SIZE = 15  # Minimum characters before sending (very aggressive)

# Rich console for colored output
console = Console()


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
        self.first_audio_played = None
        self.last_audio_played = None
        self.total_phrases = 0
        self.total_llm_chunks = 0
        self.total_audio_bytes = 0
        self.phrase_metrics = []

    def report(self):
        """Generate a comprehensive metrics report"""
        table = Table(title="Pipeline Performance Metrics", box=box.ROUNDED, show_header=True)
        table.add_column("Stage", style="cyan", no_wrap=True)
        table.add_column("Latency (ms)", style="magenta", justify="right")
        table.add_column("Details", style="green")

        if self.image_load_start and self.image_load_end:
            latency = (self.image_load_end - self.image_load_start) * 1000
            table.add_row("Image Load", f"{latency:.1f}", "File read + encoding")

        if self.llm_request_start and self.first_llm_token:
            latency = (self.first_llm_token - self.llm_request_start) * 1000
            table.add_row("First LLM Token", f"{latency:.1f}", "TTFT (Time To First Token)")

        if self.llm_request_start and self.llm_complete:
            latency = (self.llm_complete - self.llm_request_start) * 1000
            table.add_row("LLM Total", f"{latency:.1f}", f"{self.total_llm_chunks} chunks")

        if self.llm_request_start and self.first_phrase_sent:
            latency = (self.first_phrase_sent - self.llm_request_start) * 1000
            table.add_row("First Phrase Sent", f"{latency:.1f}", "LLM → Phrase detector → WS")

        if self.first_phrase_sent and self.first_audio_received:
            latency = (self.first_audio_received - self.first_phrase_sent) * 1000
            table.add_row("First Audio Chunk", f"{latency:.1f}", "TTS processing")

        if self.llm_request_start and self.first_audio_received:
            latency = (self.first_audio_received - self.llm_request_start) * 1000
            table.add_row("Image Upload → Audio", f"{latency:.1f}", "VLM request to first audio", style="bold yellow")

        if self.llm_request_start and self.first_audio_played:
            latency = (self.first_audio_played - self.llm_request_start) * 1000
            table.add_row("E2E to Audio Playback", f"{latency:.1f}", "Total latency to first sound", style="bold green")

        if self.llm_request_start and self.last_audio_played:
            latency = (self.last_audio_played - self.llm_request_start) * 1000
            table.add_row("Complete Pipeline", f"{latency:.1f}", "End-to-end total")

        console.print(table)

        # Summary stats
        summary = Table(title="Summary Statistics", box=box.SIMPLE)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="yellow", justify="right")

        summary.add_row("Total Phrases", str(self.total_phrases))
        summary.add_row("Total LLM Chunks", str(self.total_llm_chunks))
        summary.add_row("Total Audio Bytes", f"{self.total_audio_bytes:,}")

        if self.phrase_metrics:
            avg_phrase_latency = sum(m['latency'] for m in self.phrase_metrics) / len(self.phrase_metrics)
            summary.add_row("Avg Phrase Latency", f"{avg_phrase_latency:.1f} ms")

        console.print(summary)


def play_wav_bytes(wav_bytes):
    """Play WAV audio bytes"""
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


async def llm_chunk_producer(chunk_queue, prompt, model="gemini-2.0-flash", metrics=None):
    """Produce LLM chunks with detailed logging"""
    client = genai.Client()

    console.log("[cyan]Loading image...[/cyan]", style="bold")
    metrics.image_load_start = time.perf_counter()

    with open(IMAGE_PATH, 'rb') as f:
        image_bytes = f.read()

    metrics.image_load_end = time.perf_counter()
    console.log(f"[green]Image loaded: {len(image_bytes):,} bytes[/green]")

    contents = [
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=IMAGE_MIME_TYPE,
        ),
        prompt
    ]

    console.log(f"[yellow]Requesting LLM ({model})...[/yellow]", style="bold")
    metrics.llm_request_start = time.perf_counter()

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
                metrics.first_llm_token = time.perf_counter()
                ttft = (metrics.first_llm_token - metrics.llm_request_start) * 1000
                console.log(f"[bold green]FIRST TOKEN: {ttft:.1f}ms[/bold green]")
                first_chunk = False

            metrics.total_llm_chunks += 1
            console.log(f"[dim]LLM chunk #{metrics.total_llm_chunks}: {repr(text[:40])}...[/dim]")
            await chunk_queue.put(text)

    metrics.llm_complete = time.perf_counter()
    console.log(f"[green]LLM streaming complete ({metrics.total_llm_chunks} chunks)[/green]")


async def sentence_boundary_detector(llm_chunk_queue, sentence_queue, metrics=None):
    """Detect sentence boundaries with detailed logging"""
    buffer = ""
    phrase_count = 0

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

    while True:
        chunk = await llm_chunk_queue.get()

        if chunk is None:
            if buffer.strip():
                final_text = buffer.strip()
                phrase_count += 1
                console.log(f"[yellow]FINAL phrase #{phrase_count}: {final_text[:60]}...[/yellow]")
                await sentence_queue.put(final_text)
            await sentence_queue.put(None)
            break

        buffer += chunk
        console.log(f"[dim]Buffer: {len(buffer)} chars[/dim]")

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

                    console.log(f"[bold cyan]Phrase #{phrase_count}: {phrase[:60]}...[/bold cyan]")
                    await sentence_queue.put(phrase)
            else:
                break


async def sender_llm_chunks(chunk_queue, websocket, phrase_data):
    """Send phrases to websocket"""
    phrase_num = 0

    while True:
        chunk = await chunk_queue.get()
        if chunk is None:
            console.log("[dim]Sender received None, stopping[/dim]")
            break

        if not chunk or not chunk.strip():
            console.log("[yellow]Skipping empty chunk[/yellow]")
            continue

        if phrase_num > 0 and CHUNK_DELAY > 0:
            await asyncio.sleep(CHUNK_DELAY)

        phrase_num += 1
        send_time = time.perf_counter()

        phrase_data.append({
            'phrase': chunk,
            'send_time': send_time,
            'first_audio': None,
            'complete_time': None,
            'audio_bytes': 0
        })

        try:
            await websocket.send(chunk)
            console.log(f"[magenta]Sent phrase #{phrase_num} to TTS ({len(chunk)} chars)[/magenta]")
        except Exception as e:
            console.log(f"[red]Failed to send phrase #{phrase_num}: {e}[/red]")
            break


async def receiver(websocket, phrase_data, metrics, audio_queue=None):
    """Receive audio chunks from websocket"""
    current_idx = 0
    complete_wav = b""
    first_chunk = True

    # Wait for first phrase (with timeout)
    wait_start = time.perf_counter()
    while len(phrase_data) <= current_idx:
        await asyncio.sleep(0.001)
        if time.perf_counter() - wait_start > 60:  # 60 second timeout
            console.log("[yellow]Timeout waiting for phrases[/yellow]")
            return

    while True:
        try:
            response = await websocket.recv()
        except websockets.ConnectionClosed as e:
            console.log(f"[yellow]WebSocket connection closed: {e}[/yellow]")
            break
        except Exception as e:
            console.log(f"[red]Error receiving from websocket: {e}[/red]")
            break

        now = time.perf_counter()

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            console.log(f"[red]Failed to parse JSON: {e}[/red]")
            console.log(f"[dim]Response: {response[:100]}...[/dim]")
            continue

        if "audioOutput" in data:
            if first_chunk:
                phrase_data[current_idx]['first_audio'] = now
                latency = (now - phrase_data[current_idx]['send_time']) * 1000

                if current_idx == 0:
                    metrics.first_audio_received = now

                console.log(f"[green]Audio chunk for phrase #{current_idx + 1}: {latency:.1f}ms latency[/green]")
                first_chunk = False

            audio_b64 = data["audioOutput"]["audio"]
            audio_bytes = base64.b64decode(audio_b64)
            phrase_data[current_idx]['audio_bytes'] += len(audio_bytes)
            metrics.total_audio_bytes += len(audio_bytes)
            complete_wav += audio_bytes

        elif "finalOutput" in data:
            phrase_data[current_idx]['complete_time'] = now

            if audio_queue is not None:
                await audio_queue.put((complete_wav, current_idx))

            latency = (now - phrase_data[current_idx]['send_time']) * 1000
            metrics.phrase_metrics.append({
                'phrase_num': current_idx + 1,
                'latency': latency
            })

            console.log(f"[bold green]Phrase #{current_idx + 1} complete: {latency:.1f}ms total[/bold green]")

            current_idx += 1
            if current_idx >= len(phrase_data):
                break

            # Wait for next phrase (with timeout)
            wait_start = time.perf_counter()
            while len(phrase_data) <= current_idx:
                await asyncio.sleep(0.001)
                if time.perf_counter() - wait_start > 30:  # 30 second timeout for subsequent phrases
                    console.log("[yellow]Timeout waiting for next phrase[/yellow]")
                    return

            complete_wav = b""
            first_chunk = True

        elif "error" in data:
            console.log(f"[red]Error for phrase #{current_idx + 1}: {data['error']}[/red]")
            current_idx += 1
            if current_idx >= len(phrase_data):
                break

            # Wait for next phrase (with timeout)
            wait_start = time.perf_counter()
            while len(phrase_data) <= current_idx:
                await asyncio.sleep(0.001)
                if time.perf_counter() - wait_start > 30:
                    console.log("[yellow]Timeout waiting for next phrase after error[/yellow]")
                    return

            complete_wav = b""
            first_chunk = True


async def audio_player(audio_queue, metrics):
    """Play audio chunks"""
    phrase_num = 0

    while True:
        result = await audio_queue.get()
        if result is None:
            break

        wav_bytes, idx = result
        phrase_num = idx + 1

        console.log(f"[bold blue]Playing phrase #{phrase_num} ({len(wav_bytes):,} bytes)...[/bold blue]")

        if phrase_num == 1:
            metrics.first_audio_played = time.perf_counter()

        await asyncio.to_thread(play_wav_bytes, wav_bytes)

        metrics.last_audio_played = time.perf_counter()
        console.log(f"[blue]Finished playing phrase #{phrase_num}[/blue]")


async def main():
    # Display header
    console.print(Panel.fit(
        "[bold cyan]Ultra-Low Latency VLM → TTS Pipeline[/bold cyan]\n" +
        f"[yellow]Config:[/yellow] MIN_CHUNK_SIZE={MIN_CHUNK_SIZE}, CHUNK_DELAY={CHUNK_DELAY}s\n" +
        f"[dim]Image: {IMAGE_PATH}[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))

    # Set uvloop
    console.log("[cyan]Setting up uvloop for performance...[/cyan]")
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    # Initialize metrics
    metrics = PipelineMetrics()
    phrase_data = []
    chunk_queue = asyncio.Queue()
    sentence_queue = asyncio.Queue()
    audio_queue = asyncio.Queue()

    try:
        # Connect to websocket and keep it open (prewarmed and ready)
        console.log("[yellow]Connecting to websocket server...[/yellow]", style="bold")

        try:
            connection_start = time.perf_counter()
            websocket = await websockets.connect(WS_URL)
            connection_time = (time.perf_counter() - connection_start) * 1000
            console.log(f"[bold green]Websocket connected in {connection_time:.1f}ms - Socket is prewarmed and ready![/bold green]")
        except Exception as e:
            console.log(f"[red]Failed to connect to websocket: {e}[/red]")
            console.print("\n[yellow]Is the TTS server running? Start it with:[/yellow]")
            console.print("[dim]uv run fasttts[/dim]\n")
            return

        async with websocket:

            # Ask user if ready to send image
            console.print("\n[bold yellow]Ready to send image and start pipeline?[/bold yellow]")
            console.print("[dim]Press Enter to start, or Ctrl+C to cancel[/dim]")

            wait_start = time.perf_counter()
            try:
                await asyncio.to_thread(input)
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled by user[/yellow]")
                await websocket.send("stop")
                return

            wait_time = time.perf_counter() - wait_start
            console.log(f"[dim]Socket kept warm for {wait_time:.2f}s while waiting[/dim]")

            console.rule("[bold green]Starting Pipeline[/bold green]")
            pipeline_start = time.perf_counter()

            # Create tasks
            producer_task = asyncio.create_task(
                llm_chunk_producer(chunk_queue, LLM_PROMPT, metrics=metrics)
            )
            detector_task = asyncio.create_task(
                sentence_boundary_detector(chunk_queue, sentence_queue, metrics=metrics)
            )
            sender_task = asyncio.create_task(
                sender_llm_chunks(sentence_queue, websocket, phrase_data)
            )
            receiver_task = asyncio.create_task(
                receiver(websocket, phrase_data, metrics, audio_queue=audio_queue)
            )
            audio_task = asyncio.create_task(
                audio_player(audio_queue, metrics)
            )

            # Wait for completion
            try:
                await producer_task
                console.log("[dim]Producer task complete[/dim]")
                await chunk_queue.put(None)

                await detector_task
                console.log("[dim]Detector task complete[/dim]")

                await sender_task
                console.log("[dim]Sender task complete[/dim]")

                await receiver_task
                console.log("[dim]Receiver task complete[/dim]")

                await websocket.send("stop")
                console.log("[dim]Sent stop signal[/dim]")

                await audio_queue.put(None)
                await audio_task
                console.log("[dim]Audio task complete[/dim]")

                pipeline_end = time.perf_counter()
                total_time = pipeline_end - pipeline_start

            except Exception as e:
                console.log(f"[red]Error during pipeline execution: {e}[/red]")
                import traceback
                console.print(traceback.format_exc())
                raise

    except websockets.exceptions.WebSocketException as e:
        console.log(f"[red]Websocket error: {e}[/red]")
        return
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return
    except Exception as e:
        console.log(f"[red]Unexpected error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return

    # Display results
    console.rule("[bold cyan]Pipeline Complete[/bold cyan]")
    console.print(f"\n[bold green]Total Pipeline Time: {total_time:.3f}s ({total_time * 1000:.1f}ms)[/bold green]\n")

    metrics.report()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
