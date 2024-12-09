# Censor My Py! [Music Censorship Tool]

This project is a Python-based tool that censors bad words in audio files by applying effects like reversing the audio or acapella separation at specific timestamps. It uses OpenAI's Whisper model to transcribe the audio, detect bad words based on a provided dictionary, and modify the audio accordingly.

For the acapella and instrumental stem separation I used spleeter and TF of course.

## Features
- **Transcription**: Uses Whisper for word-level transcription with accurate timestamps.
- **Bad Word Detection**: Matches transcript words with a custom bad words dictionary.
- **Backspin Censorship Effect**: Choose between applying a reversing effect to the detected bad words' segments that behaves like a '90s backspin!
- **Acapella Stem Separation**: Use acapella separation method like all clean song versions in the industry to censor, leaving the 'dirty' section instrumental.
- **Downpitched Acapella**: Use down-pitched vocals to censor, leaving the 'dirty' section blurry and interview-like unclear.
- **WIP: Custom timestamps and Lyrics Alignment**: Optionally aligns provided lyrics with transcript for improved accuracy. Still in progress though.


## Pre-Requirements

   1. FFmpeg - Pydub requires FFmpeg to process audio files; You need to install FFmpeg separately. (Add FFmpeg to your system PATH if it isn't already there).
   2. If you're willing to use separation method, Install CUDA 11.x and cuDNN 8.x
   3. If you're having issues of DLLs not found while using the tool, download and place them inside your CUDA/bin (instead of inside their original NVIDIA intended dir).
 

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/dnullptr/censorMyPy.git
   cd censorMyPy

2. Install Python 3.10.x
   
     On GUI Linux or Windows, Just manually create a virtual environment with Python 3.10.x (using your favorite editor).
   
     On Linux servers/terminal-based distros, Install a virtual environment for Python 3.10.x (PyEnv working method below):
      ```bash
      echo -e 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
      echo -e 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc
      exec "$SHELL"
      pyenv install 3.10.0
      pyenv global 3.10.0
      ```
      
3. Install requirements.txt (if you haven't setup a GPU, ignore failing on tensorflow-gpu installation)
   ```bash
   pip install -r requirements.txt
   ```
4. Call the script using a song file and a bad list file; i.e., my song file is named 'song.mp3' and my bad words file is named 'bad_list.txt', choosing 'v' mode for vocal method:
   ```bash
   python censormy.py "song.mp3" "bad_words.txt" --method=v
