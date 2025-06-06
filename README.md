
# Multilingual Voice Chat RAG System

A voice-based conversational system that can understand speech in multiple languages, search through documents, generate contextual responses, and speak back in the user's language.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Module Details](#module-details)
- [Usage Guide](#usage-guide)
- [Customization](#customization)
- [Deployment Options](#deployment-options)

## Architecture Overview

```
                        ┌─────────────┐
                        │  Audio In   │
                        └──────┬──────┘
                               ▼
┌───────────┐    ┌──────────────────┐    ┌─────────────┐
│   Audio   │ => │  Speech-to-Text  │ => │   Query     │
│  Files    │    │    (Whisper)     │    │   Text      │
└───────────┘    └──────────────────┘    └──────┬──────┘
                                                 ▼
┌───────────┐    ┌──────────────────┐    ┌─────────────┐
│ Document  │ <= │    Semantic      │ <= │   Search     │
│   Store   │    │    Search        │    │   Query      │
└───────────┘    └──────────────────┘    └─────────────┘
                          │
                          ▼
┌───────────┐    ┌──────────────────┐    ┌─────────────┐
│   LLM     │ => │    Response      │ => │    Text     │
│  API/Local│    │   Generation     │    │  Response   │
└───────────┘    └──────────────────┘    └──────┬──────┘
                                                 ▼
┌───────────┐    ┌──────────────────┐    ┌─────────────┐
│  Audio    │ <= │   Text-to-Speech │ <= │  Response   │
│   Out     │    │    (Coqui TTS)   │    │    Text     │
└───────────┘    └──────────────────┘    └─────────────┘
```

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for audio processing)
- Git
- Virtual environment tool

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/your-username/voice-chat-rag
cd voice-chat-rag
```

2. **Create and activate virtual environment**
```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
cp .env.example .env

# Edit .env with your API keys
OPENAI_API_KEY=your-key-here
COHERE_API_KEY=your-key-here
```

### Requirements
```
# Core dependencies
openai-whisper>=0.5.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
TTS>=0.14.0
openai>=1.0.0
python-dotenv>=1.0.0

# Audio processing
soundfile>=0.12.1
sounddevice>=0.4.5
librosa>=0.10.0

# Utilities
numpy>=1.24.0
torch>=2.0.0
tqdm>=4.65.0
```

## Module Details

### Speech-to-Text (stt_module.py)
- Uses OpenAI's Whisper for multilingual STT
- Supports 99+ languages
- Handles various audio formats
- Can process files or live recordings

```python
from stt_module import transcribe_audio

text = transcribe_audio("audio.mp3", language="en")
```

### Document Retrieval (retrieval.py)
- Uses sentence-transformers for embeddings
- FAISS for efficient similarity search
- Supports multilingual documents
- Returns relevant context for queries

```python
from retrieval import SemanticSearchEngine

search_engine = SemanticSearchEngine("data/")
results = search_engine.search("query", top_k=3)
```

### LLM Response (llm_module.py)
- Supports OpenAI API or local models
- Contextual response generation
- Error handling and retry logic
- Multiple model options

```python
from llm_module import get_llm_response

response = get_llm_response(
    query="question",
    context="retrieved_text",
    llm_type="openai"
)
```

### Text-to-Speech (tts_module.py)
- Uses Coqui TTS for multilingual synthesis
- Multiple voice options
- Adjustable speech parameters
- Audio post-processing

```python
from tts_module import MultilingualTTS

tts = MultilingualTTS()
result = tts.generate_speech("text", language="en")
```

## Usage Guide

### Running the System

1. **Start the main script**
```bash
python main.py
```

2. **Follow the prompts**
```
Enter language code (e.g., en, es, fr) [default: en]: es
Record new audio (r) or use existing file (f)? [r/f]: r
Enter recording duration in seconds [default: 10]: 15
```

3. **Check outputs**
```
Processing Complete!
Input Audio: output/audio/input_20231225_123456.wav
Transcription: Your question in text...
Retrieved Documents:
- doc1.txt (score: 0.856)
- doc2.txt (score: 0.743)
LLM Response: Generated answer...
Output Audio: output/audio/response_20231225_123456.wav
```

### Testing with Sample Data

1. **Add test documents**
```bash
# Copy sample documents
cp samples/documents/* data/

# Structure:
data/
├── document1.txt
├── document2.txt
└── document3.txt
```

2. **Test with sample audio**
```bash
# Run with sample audio
python main.py
# Choose 'f' for file input
# Enter: samples/audio/question_en.mp3
```

## Customization

### Adding New Documents
```bash
# Add documents to data directory
cp your-documents/*.txt data/

# Rebuild index (automatic on next run)
python -c "from retrieval import SemanticSearchEngine; SemanticSearchEngine('data/').build_index()"
```

### Adding New Languages

1. **Speech-to-Text**
- Whisper supports 99+ languages automatically

2. **Document Retrieval**
- Add documents in new language to `data/`
- Multilingual embeddings handle translation

3. **Text-to-Speech**
```python
# Add new TTS model
tts_engines["new_lang"] = {
    "model_name": "tts_models/new_lang/...",
    "instance": None,
    "languages": ["new_lang"]
}
```

### Adding New Voices
```python
# Fine-tune TTS model
from TTS.trainer import Trainer

config = load_config("config.json")
trainer = Trainer(
    config,
    output_path="custom_voice",
    training_data="voice_data/"
)
trainer.fit()
```

## Deployment Options

### Local Deployment
- Recommended for development
- Full control over components
- Lower latency
- No API costs

Setup:
```bash
# Install CUDA for GPU support (optional)
# Set larger model sizes in config
# Use local LLM models
```

### Google Colab Deployment
- Good for testing
- Free GPU access
- Limited runtime

Changes needed:
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Update paths
data_dir = "/content/drive/MyDrive/data"
output_dir = "/content/drive/MyDrive/output"

# Install dependencies
!pip install -r requirements.txt
```

### Cloud Deployment Tips
- Use environment variables for credentials
- Implement proper logging
- Add API endpoints
- Consider containerization
- Monitor resource usage

Example Docker setup:
```dockerfile
FROM python:3.8
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.