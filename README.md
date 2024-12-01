# Censor My Py! [Music Censorship Tool]

This project is a Python-based tool that censors bad words in audio files by applying effects like reversing the audio or acapella separation at specific timestamps. It uses OpenAI's Whisper model to transcribe the audio, detect bad words based on a provided dictionary, and modify the audio accordingly.

For the acapella and instrumental stem separation I used spleeter and TF of course.

## Features
- **Transcription**: Uses Whisper for word-level transcription with accurate timestamps.
- **Bad Word Detection**: Matches transcript words with a custom bad words dictionary.
- **Backspin Censorship Effect**: Choose between applying a reversing effect to the detected bad words' segments that behaves like a '90s backspin!
- **Acapella Stem Separation**: Use acapella separation method like all clean song versions in the industry to censor, leaving the 'dirty' section instrumental
- **WIP: Custom timestamps and Lyrics Alignment**: Optionally aligns provided lyrics with transcript for improved accuracy. Still in progress though.


## Pre-Requirements
1. If you're willing to use separation method, Install CUDA
   ```bash
   Windows:
   https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_561.17_windows.exe
   Linux:
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
   sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.3-560.35.05-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.3-560.35.05-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-6
## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/dnullptr/censorMyPy.git
   cd censorMyPy
