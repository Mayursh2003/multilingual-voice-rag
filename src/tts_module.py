import os
from pathlib import Path
import torch
from TTS.api import TTS
from typing import Optional, Dict, List
import logging
from dataclasses import dataclass
import soundfile as sf
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TTSResult:
    """Data class to store TTS generation results"""
    audio_path: str
    sample_rate: int
    duration: float
    language: str
    speaker: str
    error: Optional[str] = None


class MultilingualTTS:
    """
    Multilingual Text-to-Speech synthesis using Coqui TTS.
    Supports multiple languages and speakers with different models.
    """
    
    def __init__(
        self,
        output_dir: str = "audio_output",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = device
        
        # Initialize TTS engines
        self._initialize_tts_engines()

    def _initialize_tts_engines(self) -> None:
        """
        Initialize different TTS models for various languages.
        Models are loaded on-demand to save memory.
        """
        self.tts_engines: Dict[str, Dict] = {
            "multilingual": {
                "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
                "instance": None,
                "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn"],
            },
            "en": {
                "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
                "instance": None,
                "languages": ["en"],
            },
            "ja": {
                "model_name": "tts_models/ja/kokoro/tacotron2-DDC",
                "instance": None,
                "languages": ["ja"],
            }
        }

    def _load_tts_engine(self, language: str) -> Optional[TTS]:
        """
        Load appropriate TTS engine for the given language.
        
        Args:
            language: Language code (e.g., 'en', 'es')
            
        Returns:
            TTS instance or None if language not supported
        """
        # First, try language-specific model
        if language in self.tts_engines and self.tts_engines[language]["instance"] is None:
            logger.info(f"Loading language-specific model for {language}")
            self.tts_engines[language]["instance"] = TTS(
                model_name=self.tts_engines[language]["model_name"],
                progress_bar=True,
                gpu=self.device == "cuda"
            )
            return self.tts_engines[language]["instance"]
            
        # Fallback to multilingual model
        if (language in self.tts_engines["multilingual"]["languages"] and 
            self.tts_engines["multilingual"]["instance"] is None):
            logger.info("Loading multilingual model")
            self.tts_engines["multilingual"]["instance"] = TTS(
                model_name=self.tts_engines["multilingual"]["model_name"],
                progress_bar=True,
                gpu=self.device == "cuda"
            )
            return self.tts_engines["multilingual"]["instance"]
            
        # Return existing instance if already loaded
        for engine in self.tts_engines.values():
            if (engine["instance"] is not None and 
                language in engine["languages"]):
                return engine["instance"]
                
        return None

    def generate_speech(
        self,
        text: str,
        language: str,
        speaker: str = None,
        output_filename: str = None,
        speech_rate: float = 1.0
    ) -> TTSResult:
        """
        Generate speech from text in specified language.
        
        Args:
            text: Input text to synthesize
            language: Language code (e.g., 'en', 'es')
            speaker: Speaker ID/name (if supported by model)
            output_filename: Custom filename for output
            speech_rate: Speed of speech (1.0 is normal)
            
        Returns:
            TTSResult object containing generation results
        """
        try:
            # Load appropriate TTS engine
            tts_engine = self._load_tts_engine(language)
            if not tts_engine:
                raise ValueError(f"No TTS engine available for language: {language}")

            # Generate timestamp-based filename if not provided
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"speech_{language}_{timestamp}.wav"
            
            output_path = self.output_dir / output_filename

            # Generate speech
            logger.info(f"Generating speech for language: {language}")
            
            # Handle different model types and capabilities
            if speaker and hasattr(tts_engine, 'speakers'):
                wav = tts_engine.tts(
                    text=text,
                    speaker=speaker,
                    language=language
                )
            else:
                wav = tts_engine.tts(text=text, language=language)

            # Apply speech rate modification if needed
            if speech_rate != 1.0:
                wav = self._modify_speech_rate(wav, speech_rate)

            # Save audio
            if isinstance(wav, tuple):
                wav, sample_rate = wav
            else:
                sample_rate = 22050  # Default sample rate

            sf.write(output_path, wav, sample_rate)
            
            duration = len(wav) / sample_rate
            
            return TTSResult(
                audio_path=str(output_path),
                sample_rate=sample_rate,
                duration=duration,
                language=language,
                speaker=speaker or "default"
            )

        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            return TTSResult(
                audio_path="",
                sample_rate=0,
                duration=0,
                language=language,
                speaker=speaker or "default",
                error=str(e)
            )

    def _modify_speech_rate(
        self,
        wav: np.ndarray,
        rate: float
    ) -> np.ndarray:
        """Modify the speech rate of the audio"""
        try:
            import librosa
            return librosa.effects.time_stretch(wav, rate=1/rate)
        except ImportError:
            logger.warning("librosa not installed, speech rate modification unavailable")
            return wav

    def list_available_languages(self) -> List[str]:
        """Return list of available language codes"""
        available_languages = set()
        for engine in self.tts_engines.values():
            available_languages.update(engine["languages"])
        return sorted(list(available_languages))

    def list_available_speakers(self, language: str) -> List[str]:
        """Return list of available speakers for a language"""
        tts_engine = self._load_tts_engine(language)
        if tts_engine and hasattr(tts_engine, 'speakers'):
            return tts_engine.speakers
        return ["default"]


# Example usage and testing
def main():
    # Initialize TTS
    tts = MultilingualTTS()
    
    # Example texts in different languages
    texts = {
        "en": "Hello, this is a test of the English text-to-speech system.",
        "es": "Hola, esta es una prueba del sistema de texto a voz en español.",
        "fr": "Bonjour, ceci est un test du système de synthèse vocale en français.",
        "de": "Hallo, dies ist ein Test des deutschen Text-zu-Sprache-Systems.",
    }
    
    # Generate speech for each language
    for lang, text in texts.items():
        print(f"\nGenerating speech for {lang}")
        result = tts.generate_speech(
            text=text,
            language=lang,
            speech_rate=1.0
        )
        
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Generated: {result.audio_path}")
            print(f"Duration: {result.duration:.2f}s")


if __name__ == "__main__":
    main()