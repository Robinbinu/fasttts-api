"""Audio processing utilities for format detection and conversion."""

import io
import wave

# Try to import pydub for audio format conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None
    print("Warning: pydub not available. MP3 conversion will not work. Install with: pip install pydub")


def detect_audio_format(chunk: bytes) -> str:
    """Detect audio format from the first few bytes of a chunk.

    Args:
        chunk: Raw audio bytes

    Returns:
        Format string: 'mp3', 'wav', 'ogg', or 'pcm'
    """
    if not chunk or len(chunk) < 4:
        return 'pcm'

    # Check for MP3 (ID3 tag or MP3 frame header)
    if chunk[:3] == b'ID3':
        return 'mp3'
    if len(chunk) >= 2 and chunk[0] == 0xff and (chunk[1] & 0xe0) == 0xe0:
        return 'mp3'

    # Check for WAV
    if chunk[:4] == b'RIFF':
        return 'wav'

    # Check for OGG
    if chunk[:4] == b'OggS':
        return 'ogg'

    return 'pcm'


def convert_audio_to_pcm(
    audio_data: bytes,
    source_format: str,
    target_sample_rate: int = 24000
) -> tuple[bytes, int, int, int]:
    """Convert audio data from various formats to raw PCM.

    Args:
        audio_data: Raw audio bytes
        source_format: Format of the source audio (mp3, wav, ogg, pcm)
        target_sample_rate: Target sample rate for output (default 24000)

    Returns:
        tuple: (pcm_data, sample_rate, channels, sample_width)
    """
    if source_format == 'pcm':
        return (audio_data, target_sample_rate, 1, 2)

    if not PYDUB_AVAILABLE:
        print(f"Warning: Cannot convert {source_format} to PCM - pydub not available")
        return (audio_data, target_sample_rate, 1, 2)

    try:
        audio_io = io.BytesIO(audio_data)

        if source_format == 'mp3':
            audio = AudioSegment.from_mp3(audio_io)
        elif source_format == 'wav':
            audio = AudioSegment.from_wav(audio_io)
        elif source_format == 'ogg':
            audio = AudioSegment.from_ogg(audio_io)
        else:
            print(f"Warning: Unknown format {source_format}, returning original data")
            return (audio_data, target_sample_rate, 1, 2)

        print(f"Original audio: {audio.frame_rate}Hz, {audio.channels}ch, {audio.sample_width*8}bit")

        # Convert to PCM: mono, 16-bit, target sample rate
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)  # 16-bit

        if audio.frame_rate != target_sample_rate:
            print(f"Resampling from {audio.frame_rate}Hz to {target_sample_rate}Hz")
            audio = audio.set_frame_rate(target_sample_rate)

        print(f"Converted audio: {audio.frame_rate}Hz, {audio.channels}ch, {audio.sample_width*8}bit")

        return (audio.raw_data, audio.frame_rate, audio.channels, audio.sample_width)
    except Exception as e:
        print(f"Error converting {source_format} to PCM: {e}")
        import traceback
        traceback.print_exc()
        return (audio_data, target_sample_rate, 1, 2)


def create_complete_wav_file(
    pcm_data: bytes,
    sample_rate: int = 24000,
    num_channels: int = 1,
    sample_width: int = 2
) -> bytes:
    """Create a complete WAV file with proper headers and data.

    Args:
        pcm_data: Raw PCM audio data
        sample_rate: Sample rate in Hz
        num_channels: Number of audio channels
        sample_width: Sample width in bytes

    Returns:
        Complete WAV file as bytes
    """
    try:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)

        wav_buffer.seek(0)
        complete_wav = wav_buffer.read()
        wav_buffer.close()

        return complete_wav
    except Exception as e:
        print(f"Error creating WAV file: {e}")
        import traceback
        traceback.print_exc()
        raise


def create_wave_header_for_engine(engine) -> bytes:
    """Create a WAV header based on engine stream info.

    Args:
        engine: TTS engine instance with get_stream_info() method

    Returns:
        WAV header bytes
    """
    try:
        stream_info = engine.get_stream_info()
        print(f"Debug: stream_info = {stream_info}")
        _, _, sample_rate = stream_info
        print(f"Debug: extracted sample_rate = {sample_rate}")
    except Exception as e:
        print(f"Warning: Could not get stream info from engine: {e}")
        sample_rate = None

    # Default to 24kHz if sample_rate is not specified or invalid
    if sample_rate is None or sample_rate <= 0:
        sample_rate = 24000
        print(f"Warning: Sample rate not specified or invalid by engine, defaulting to {sample_rate}Hz")

    num_channels = 1
    sample_width = 2
    frame_rate = sample_rate

    print(f"Debug: Creating WAV header with frame_rate={frame_rate}, num_channels={num_channels}, sample_width={sample_width}")

    try:
        wav_header = io.BytesIO()
        with wave.open(wav_header, "wb") as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(frame_rate)

        wav_header.seek(0)
        wave_header_bytes = wav_header.read()
        wav_header.close()

        final_wave_header = io.BytesIO()
        final_wave_header.write(wave_header_bytes)
        final_wave_header.seek(0)

        return final_wave_header.getvalue()
    except Exception as e:
        print(f"Error creating wave header: {e}")
        raise
