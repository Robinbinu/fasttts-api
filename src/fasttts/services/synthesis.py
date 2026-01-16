"""TTS synthesis request handling."""

import logging
import threading
from queue import Queue

from RealtimeTTS import TextToAudioStream

from .audio import (
    detect_audio_format,
    convert_audio_to_pcm,
    create_complete_wav_file,
)


class TTSRequestHandler:
    """Handles a single TTS synthesis request."""

    def __init__(self, engine):
        self.engine = engine
        self.audio_queue: Queue = Queue()
        self.stream = TextToAudioStream(
            engine, on_audio_stream_stop=self.on_audio_stream_stop, muted=True
        )
        self.speaking = False
        self.generation_complete = threading.Event()

    def on_audio_chunk(self, chunk: bytes) -> None:
        """Callback when an audio chunk is received."""
        self.audio_queue.put(chunk)

    def on_audio_stream_stop(self) -> None:
        """Callback when audio stream stops."""
        self.audio_queue.put(None)
        self.speaking = False
        self.generation_complete.set()

    def play_text_to_speech(self, text: str) -> None:
        """Start TTS synthesis for given text."""
        self.speaking = True
        self.stream.feed(text)
        logging.debug(f"Playing audio for text: {text}")
        print(f'Synthesizing: "{text}"')
        self.stream.play_async(on_audio_chunk=self.on_audio_chunk, muted=True)

    def audio_chunk_generator(self, send_wave_headers: bool):
        """Generate audio chunks, converting format if necessary.

        Always outputs consistent PCM + WAV header for browser requests.

        Args:
            send_wave_headers: If True, wraps audio in complete WAV file

        Yields:
            Audio data chunks
        """
        try:
            # Collect all chunks first
            chunks = []
            while True:
                chunk = self.audio_queue.get()
                if chunk is None:
                    print("Terminating stream")
                    break
                chunks.append(chunk)

            if not chunks:
                print("No audio chunks received")
                return

            # Combine all chunks into single audio data
            audio_data = b''.join(chunks)

            # Detect format from first chunk
            detected_format = detect_audio_format(chunks[0])
            print(f"Detected audio format: {detected_format}")

            # Convert to PCM if needed
            sample_rate = 24000
            if detected_format != 'pcm':
                print(f"Converting {detected_format} to PCM...")
                audio_data, sample_rate, channels, sample_width = convert_audio_to_pcm(audio_data, detected_format)
                print(f"Conversion complete. Audio size: {len(audio_data)} bytes at {sample_rate}Hz")
            else:
                # For PCM, try to get sample rate from engine
                try:
                    _, _, engine_rate = self.engine.get_stream_info()
                    if engine_rate and engine_rate > 0:
                        sample_rate = engine_rate
                except Exception:
                    pass

            # For browser requests, create complete WAV file for Safari/iOS compatibility
            if send_wave_headers:
                print(f"Creating complete WAV file for browser: {sample_rate}Hz, size: {len(audio_data)} bytes")
                complete_wav = create_complete_wav_file(audio_data, sample_rate)
                print(f"Complete WAV file size: {len(complete_wav)} bytes")

                # Stream the complete WAV file in chunks
                chunk_size = 4096
                for i in range(0, len(complete_wav), chunk_size):
                    yield complete_wav[i:i + chunk_size]
            else:
                # For non-browser requests, just send raw PCM
                chunk_size = 4096
                for i in range(0, len(audio_data), chunk_size):
                    yield audio_data[i:i + chunk_size]

        except Exception as e:
            print(f"Error during streaming: {str(e)}")
            import traceback
            traceback.print_exc()
