# Censor My Py! [Music Censorship Tool]

This project is a Python-based tool that censors bad words in audio files by applying effects like reversing the audio or acapella separation at specific timestamps. It uses OpenAI's Whisper model to transcribe the audio, detect bad words based on a provided dictionary, and modify the audio accordingly.
For the acapella and instrumental stem separation I used spleeter and TF of course

## Features
- **Transcription**: Uses Whisper for word-level transcription with accurate timestamps.
- **Bad Word Detection**: Matches transcript words with a custom bad words dictionary.
- **Backspin Censorship Effect**: Choose between applying a reversing effect to the detected bad words' segments that behaves like a '90s backspin!
- **Acapella Stem Separation**: Use acapella separation method like all clean song versions in the industry to censor, leaving the 'dirty' section instrumental
- **WIP: Custom timestamps and Lyrics Alignment**: Optionally aligns provided lyrics with transcript for improved accuracy. Still in progress though.
- **
## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/dnullptr/censorMyPy.git
   cd censorMyPy
