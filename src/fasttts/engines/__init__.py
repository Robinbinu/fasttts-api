"""TTS Engine management and initialization."""

import logging
import sys
import threading
from pathlib import Path
from typing import Any

# Add local lib to path for vendored RealtimeTTS
_lib_path = Path(__file__).parent.parent.parent.parent / "lib"
if str(_lib_path) not in sys.path:
    sys.path.insert(0, str(_lib_path))

# Lazy import engines to handle missing dependencies gracefully
AzureEngine = None
ElevenlabsEngine = None
SystemEngine = None
OpenAIEngine = None
KokoroEngine = None
NeuTTSEngine = None
PocketTTSEngine = None

def _import_engines():
    """Import engines lazily, skipping those with missing dependencies."""
    global AzureEngine, ElevenlabsEngine, SystemEngine, OpenAIEngine, KokoroEngine, NeuTTSEngine, PocketTTSEngine

    try:
        from RealtimeTTS import AzureEngine as _AzureEngine
        AzureEngine = _AzureEngine
    except ImportError as e:
        logging.warning(f"AzureEngine not available: {e}")

    try:
        from RealtimeTTS import ElevenlabsEngine as _ElevenlabsEngine
        ElevenlabsEngine = _ElevenlabsEngine
    except ImportError as e:
        logging.warning(f"ElevenlabsEngine not available: {e}")

    try:
        from RealtimeTTS import SystemEngine as _SystemEngine
        SystemEngine = _SystemEngine
    except ImportError as e:
        logging.warning(f"SystemEngine not available: {e}")

    try:
        from RealtimeTTS import OpenAIEngine as _OpenAIEngine
        OpenAIEngine = _OpenAIEngine
    except ImportError as e:
        logging.warning(f"OpenAIEngine not available: {e}")

    try:
        from RealtimeTTS import KokoroEngine as _KokoroEngine
        KokoroEngine = _KokoroEngine
    except (ImportError, AttributeError) as e:
        logging.warning(f"KokoroEngine not available: {e}")

    try:
        from RealtimeTTS import NeuTTSEngine as _NeuTTSEngine
        NeuTTSEngine = _NeuTTSEngine
    except ImportError as e:
        logging.warning(f"NeuTTSEngine not available: {e}")

    try:
        from RealtimeTTS import PocketTTSEngine as _PocketTTSEngine
        PocketTTSEngine = _PocketTTSEngine
    except ImportError as e:
        logging.warning(f"PocketTTSEngine not available: {e}")

from ..config import settings


class EngineManager:
    """Manages TTS engines and their voices."""

    def __init__(self):
        self.engines: dict[str, Any] = {}
        self.voices: dict[str, list] = {}
        self.current_engine: Any = None
        self.current_engine_name: str | None = None
        self.tts_lock = threading.Lock()
        self.play_semaphore = threading.Semaphore(1)

    def set_engine(self, engine_name: str) -> bool:
        """Switch to a different TTS engine.

        Args:
            engine_name: Name of the engine to switch to

        Returns:
            True if successful, False otherwise
        """
        if engine_name not in self.engines:
            print(f"Warning: Engine '{engine_name}' not available")
            return False

        self.current_engine = self.engines[engine_name]
        self.current_engine_name = engine_name

        if self.voices.get(engine_name):
            self.engines[engine_name].set_voice(self.voices[engine_name][0].name)

        return True

    def init_engines(self) -> None:
        """Initialize all supported TTS engines based on available API keys."""
        print("Initializing TTS Engines")

        # Import engines (handles missing dependencies gracefully)
        _import_engines()

        for engine_name in settings.supported_engines:
            try:
                if engine_name == "azure":
                    if AzureEngine is None:
                        print("Azure engine skipped - dependencies not available")
                    elif settings.azure_speech_key and settings.azure_speech_region:
                        print("Initializing azure engine")
                        self.engines["azure"] = AzureEngine(
                            settings.azure_speech_key,
                            settings.azure_speech_region
                        )
                    else:
                        print("Azure engine skipped - missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION")

                elif engine_name == "elevenlabs":
                    if ElevenlabsEngine is None:
                        print("Elevenlabs engine skipped - dependencies not available")
                    elif settings.elevenlabs_api_key:
                        print("Initializing elevenlabs engine")
                        self.engines["elevenlabs"] = ElevenlabsEngine(settings.elevenlabs_api_key)
                    else:
                        print("Elevenlabs engine skipped - missing ELEVENLABS_API_KEY")

                elif engine_name == "system":
                    if SystemEngine is None:
                        print("System engine skipped - dependencies not available")
                    else:
                        print("Initializing system engine")
                        self.engines["system"] = SystemEngine()

                elif engine_name == "kokoro":
                    if KokoroEngine is None:
                        print("Kokoro engine skipped - dependencies not available")
                    else:
                        print("Initializing kokoro engine")
                        self.engines["kokoro"] = KokoroEngine()

                elif engine_name == "openai":
                    if OpenAIEngine is None:
                        print("OpenAI engine skipped - dependencies not available")
                    elif settings.openai_api_key:
                        print("Initializing openai engine")
                        self.engines["openai"] = OpenAIEngine()
                    else:
                        print("OpenAI engine skipped - missing OPENAI_API_KEY")

                elif engine_name == "neutts":
                    if NeuTTSEngine is None:
                        print("NeuTTS engine skipped - dependencies not available")
                    else:
                        print("Initializing neutts engine")
                        self.engines["neutts"] = NeuTTSEngine(
                            model=settings.neutts_model,
                            device=settings.neutts_device,
                            voices_dir=str(settings.neutts_voices_dir) if settings.neutts_voices_dir else None,
                        )

                elif engine_name == "pocket_tts":
                    if PocketTTSEngine is None:
                        print("PocketTTS engine skipped - dependencies not available")
                    else:
                        print("Initializing pocket_tts engine")
                        self.engines["pocket_tts"] = PocketTTSEngine(
                            voice=settings.pocket_tts_voice,
                        )

            except Exception as e:
                logging.error(f"Error initializing {engine_name} engine: {e}")

        if not self.engines:
            raise RuntimeError("No TTS engines initialized")

        # Retrieve voices for each engine
        for name, engine in self.engines.items():
            print(f"Retrieving voices for TTS Engine {name}")
            try:
                self.voices[name] = engine.get_voices()
            except Exception as e:
                self.voices[name] = []
                logging.error(f"Error retrieving voices for {name}: {str(e)}")

        # Set default engine to first available
        for engine_name in settings.supported_engines:
            if engine_name in self.engines:
                self.set_engine(engine_name)
                print(f"Default engine set to: {engine_name}")
                break

    def get_engine_names(self) -> list[str]:
        """Get list of initialized engine names."""
        return list(self.engines.keys())

    def get_voices(self) -> list[str]:
        """Get list of voice names for current engine."""
        if not self.current_engine_name or self.current_engine_name not in self.voices:
            return []

        return [voice.name for voice in self.voices[self.current_engine_name]]

    def set_voice(self, voice_name: str) -> bool:
        """Set voice for current engine.

        Args:
            voice_name: Name of the voice to use

        Returns:
            True if successful, False otherwise
        """
        if not self.current_engine:
            return False

        try:
            self.current_engine.set_voice(voice_name)
            return True
        except Exception as e:
            logging.error(f"Error setting voice: {str(e)}")
            return False


# Global engine manager instance
engine_manager = EngineManager()
