import whisper
import pathlib
from typing import Optional

"""
Installation requirements:
pip install openai-whisper
# For FFmpeg dependency on Ubuntu/Debian:
# sudo apt update && sudo apt install ffmpeg
# For FFmpeg on MacOS:
# brew install ffmpeg
# For FFmpeg on Windows:
# Download from https://ffmpeg.org/download.html

Supported languages include (but not limited to):
- 'en': English
- 'es': Spanish
- 'fr': French
- 'de': German
- 'zh': Chinese
- 'ja': Japanese
"""

def transcribe_audio(
    audio_path: str,
    language: str = None,
    model_size: str = "base"
) -> Optional[str]:
    """
    Transcribe audio file to text using OpenAI's Whisper model.
    
    Args:
        audio_path (str): Path to the audio file
        language (str, optional): Language code (e.g., 'en', 'es', 'fr')
        model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    
    Returns:
        str: Transcribed text or None if error occurs
    
    Example:
        text = transcribe_audio("speech.mp3", language="en")
        text = transcribe_audio("discurso.wav", language="es")
        text = transcribe_audio("discours.ogg", language="fr")
    """
    try:
        # Validate audio file exists
        audio_file = pathlib.Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load Whisper model (downloads automatically on first run)
        print(f"Loading Whisper {model_size} model...")
        model = whisper.load_model(model_size)
        
        # Perform transcription
        print(f"Transcribing audio file: {audio_path}")
        result = model.transcribe(
            audio_path,
            language=language,  # None for auto-detection
            fp16=False  # Set to True if using GPU
        )
        
        return result["text"].strip()

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return None


# Example usage and test function
def test_transcription():
    """Test the transcription function with different languages"""
    test_files = {
        "english.mp3": "en",
        "spanish.mp3": "es",
        "french.mp3": "fr"
    }
    
    for file, lang in test_files.items():
        print(f"\nTesting {lang} transcription...")
        try:
            text = transcribe_audio(file, language=lang)
            if text:
                print(f"Transcribed text ({lang}): {text}")
            else:
                print(f"Transcription failed for {file}")
        except Exception as e:
            print(f"Error testing {file}: {str(e)}")


if __name__ == "__main__":
    # Example usage
    audio_path = "sample.mp3"
    transcribed_text = transcribe_audio(audio_path, language="en")
    if transcribed_text:
        print(f"Transcribed text: {transcribed_text}")
        