# Censor My Py! [Music Censorship Tool]
<p align="center"><img width="768" height="512" alt="20251221_2354_Python Music Censorship_remix_01kd1edwnyew2vcvtt3myst2mm" src="https://github.com/user-attachments/assets/5b0d6ea9-ff11-4051-bf7a-48996e23c701" />
</p>

This project is a Python-based tool that censors bad words in audio files by applying effects like reversing the audio or acapella separation at specific timestamps. It uses OpenAI's Whisper model to transcribe the audio, detect bad words based on a provided dictionary, and modify the audio accordingly.

For the acapella and instrumental stem separation I used spleeter and TF of course.

## Features
- **Fully Asynchronous Pipeline**: All audio processing and censorship methods are now async for faster, concurrent execution.
- **Transcription**: Uses Whisper (and optionally GenAI) for word-level transcription with accurate timestamps.
- **Bad Word & Slur Detection**: Detects both bad words and slurs using custom dictionaries, with support for separate handling.
- **Multiple Censorship Methods**:
    - **Vocal Separation**: Replace vocals with instrumentals for censored sections.
    - **Backspin Effect**: Reverse audio segments containing bad words or slurs.
    - **Vocal + Backspin**: Overlay reversed vocals on instrumentals for enhanced censorship.
    - **Downpitched Vocals**: Down-pitch vocals in censored sections for a muffled/interview-like effect.
    - **Slur + Vocal/Instrumental**: Special handling for slurs, combining instrumental and down-pitched vocal effects.
    - **GenAI Vocal Separation**: Use generative AI for advanced vocal separation and censorship.
- **Concurrent Processing**: Audio separation and censorship run in parallel for improved speed.
- **Easy Migration**: Switch from the old synchronous pipeline to the new async pipeline with minimal changes.
- **Custom timestamps and Lyrics Alignment**: [Implemented now!]Optionally aligns provided lyrics with transcript for improved accuracy. Still in progress though.
- **Cleanup Utility**: Automated cleanup of temporary files and directories after processing.


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
   ```

## [NEW] Asynchronous Pipeline (Recommended)

> **Note:** The regular (non-async) method is now deprecated. The entire pipeline and toolset have been re-written using asynchronous functions for improved performance and scalability.

### Usage
Run the new async tool with:
```bash
python async_censormy.py "song.mp3" "bad_words.txt" "slurs.txt" --method=<method> [--output=output.mp3]
```
- **audio_file**: Path to the audio file to be censored.
- **bad_words_file**: Path to the bad words file.
- **slurs_file**: Path to the slurs file.
- **--method**: Censorship method. Choices:
    - `v`: Async vocal separation
    - `Gv`: GenAI vocal separation
    - `b`: Async backspin
    - `vb`: Async vocal + backspin
    - `p`: Async vocal downpitch
    - `sv`: Async slur + vocal
    - `sb`: Async slur + vocal + backspin
- **--output**: Output file path (default: `censored_output.mp3`)

### New Async Methods
- All censorship and audio processing methods are now async, including:
    - `separate_audio`
    - `censor_with_instrumentals`
    - `censor_with_backspin`
    - `censor_with_both`
    - `censor_with_downpitch`
    - `censor_with_instrumentals_and_downpitch`
    - `censor_with_both_and_downpitch`
    - `cleanup`
- The async pipeline allows concurrent separation and censorship for faster processing.

### Migration
- **Old method**: `censormy.py` (deprecated)
- **New method**: `async_censormy.py` (recommended)

---
