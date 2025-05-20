import os
from pathlib import Path
import logging
from typing import Optional, Dict
from dataclasses import dataclass
import sounddevice as sd
import soundfile as sf
from datetime import datetime

# Import custom modules
from stt_module import transcribe_audio
from retrieval import SemanticSearchEngine
from llm_module import get_llm_response
from tts_module import MultilingualTTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConversationResult:
    """Store all outputs from the conversation pipeline"""
    input_audio: str
    transcription: str
    retrieved_documents: Dict[str, float]  # filename: similarity_score
    llm_response: str
    output_audio: str
    timestamp: str

class VoiceChatSystem:
    """Main system that orchestrates all components"""
    
    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "output",
        language: str = "en"
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.language = language
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        for subdir in ['audio', 'text', 'transcripts']:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing system components...")
            
            # Initialize semantic search
            self.search_engine = SemanticSearchEngine(
                documents_dir=self.data_dir
            )
            
            # Initialize TTS
            self.tts_engine = MultilingualTTS(
                output_dir=str(self.output_dir / 'audio')
            )
            
            logger.info("System initialization complete")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    def record_audio(
        self,
        duration: int = 10,
        sample_rate: int = 16000
    ) -> Optional[str]:
        """Record audio from microphone"""
        try:
            logger.info(f"Recording for {duration} seconds...")
            
            # Record audio
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1
            )
            sd.wait()
            
            # Save recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / 'audio' / f'input_{timestamp}.wav'
            sf.write(output_path, recording, sample_rate)
            
            logger.info(f"Recording saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error during recording: {str(e)}")
            return None

    def process_conversation(
        self,
        audio_path: str,
        language: str
    ) -> Optional[ConversationResult]:
        """
        Process the entire conversation pipeline
        
        Args:
            audio_path: Path to input audio file
            language: Language code (e.g., 'en', 'es')
            
        Returns:
            ConversationResult object or None if error occurs
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"Processing conversation, timestamp: {timestamp}")
            
            # 1. Speech to Text
            logger.info("Converting speech to text...")
            transcription = transcribe_audio(audio_path, language)
            if not transcription:
                raise ValueError("Speech-to-text conversion failed")
            
            # Save transcription
            transcript_path = self.output_dir / 'transcripts' / f'transcript_{timestamp}.txt'
            transcript_path.write_text(transcription)
            
            # 2. Semantic Search
            logger.info("Retrieving relevant documents...")
            search_results = self.search_engine.search(
                query=transcription,
                top_k=2
            )
            
            # Combine retrieved documents
            context = "\n\n".join([doc.content for doc in search_results])
            
            # 3. LLM Response
            logger.info("Generating LLM response...")
            llm_result = get_llm_response(
                query=transcription,
                context=context,
                llm_type="openai"  # or "local" for local model
            )
            
            if llm_result.error:
                raise ValueError(f"LLM error: {llm_result.error}")
            
            # Save LLM response
            response_path = self.output_dir / 'text' / f'response_{timestamp}.txt'
            response_path.write_text(llm_result.answer)
            
            # 4. Text to Speech
            logger.info("Converting response to speech...")
            tts_result = self.tts_engine.generate_speech(
                text=llm_result.answer,
                language=language
            )
            
            if tts_result.error:
                raise ValueError(f"TTS error: {tts_result.error}")
            
            # Create result object
            result = ConversationResult(
                input_audio=audio_path,
                transcription=transcription,
                retrieved_documents={
                    doc.filename: doc.similarity_score
                    for doc in search_results
                },
                llm_response=llm_result.answer,
                output_audio=tts_result.audio_path,
                timestamp=timestamp
            )
            
            logger.info("Conversation processing complete")
            return result
            
        except Exception as e:
            logger.error(f"Error processing conversation: {str(e)}")
            return None


def main():
    """Main execution function"""
    try:
        # Initialize system
        system = VoiceChatSystem()
        
        # Get language preference
        language = input("Enter language code (e.g., en, es, fr) [default: en]: ").strip()
        if not language:
            language = "en"
            
        # Get input method preference
        input_method = input("Record new audio (r) or use existing file (f)? [r/f]: ").strip().lower()
        
        if input_method == 'r':
            # Record new audio
            duration = int(input("Enter recording duration in seconds [default: 10]: ") or 10)
            audio_path = system.record_audio(duration=duration)
            if not audio_path:
                raise ValueError("Recording failed")
        else:
            # Use existing file
            audio_path = input("Enter path to audio file: ").strip()
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio file not found: {audio_path}")
        
        # Process conversation
        result = system.process_conversation(audio_path, language)
        
        if result:
            # Print results
            print("\nProcessing Complete!")
            print(f"Input Audio: {result.input_audio}")
            print(f"Transcription: {result.transcription}")
            print("\nRetrieved Documents:")
            for doc, score in result.retrieved_documents.items():
                print(f"- {doc} (score: {score:.3f})")
            print(f"\nLLM Response: {result.llm_response}")
            print(f"Output Audio: {result.output_audio}")
            print(f"\nAll outputs saved with timestamp: {result.timestamp}")
        else:
            print("\nError occurred during processing")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        logger.exception("Unexpected error in main")


if __name__ == "__main__":
    main()