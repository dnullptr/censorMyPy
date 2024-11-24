# Audio Censorship Tool with Whisper

This project is a Python-based tool that censors bad words in audio files by applying effects like reversing the audio at specific timestamps. It uses OpenAI's Whisper model to transcribe the audio, detect bad words based on a provided dictionary, and modify the audio accordingly.

## Features
- **Transcription**: Uses Whisper for word-level transcription with accurate timestamps.
- **Bad Word Detection**: Matches transcript words with a custom bad words dictionary.
- **Censorship Effects**: Applies a reversing effect to the detected bad words' segments.
- **Lyrics Alignment**: Optionally aligns provided lyrics with transcript for improved accuracy.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/dnullptr/censorMyPy.git
   cd censorMyPy
