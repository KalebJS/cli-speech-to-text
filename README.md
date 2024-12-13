# Live Speech Transcription CLI

This is a command-line application that performs real-time speech transcription using OpenAI's Whisper model. It continuously listens to your microphone input and transcribes your speech to text in real-time.

## Prerequisites

- Python 3.8 or higher
- portaudio (installed via brew on macOS)
- pip (Python package manager)

## Installation

1. Install portaudio (on macOS):
```bash
brew install portaudio
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Simply run the script:
```bash
python live_transcribe.py
```

The application will:
1. Start listening to your microphone immediately
2. Process speech in ~2-second chunks
3. Display transcribed text in real-time
4. Continue until you press Ctrl+C to stop

## Notes

- The application uses the "base" Whisper model by default, which provides a good balance between accuracy and performance
- Audio is processed in ~2-second chunks to provide near real-time transcription
- The application requires a working microphone
- Internet connection is not required as Whisper runs locally
