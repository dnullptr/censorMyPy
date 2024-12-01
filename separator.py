from spleeter.separator import Separator
import os

def separate_audio(input_audio_path, output_dir="output"):
    """
    Separate the input audio into vocals and instrumental using Spleeter.
    """
    print(f"Separating audio: {input_audio_path}")
    separator = Separator('spleeter:2stems')  # 2 stems: vocals + instrumental
    separator.separate_to_file(input_audio_path, output_dir)
    print(f"Separation completed! Files saved in: {output_dir}")

if __name__ == "__main__":
    input_audio_path = "song.mp3"  # Replace with your audio file path
    output_dir = "separated_audio"  # Directory to save separated files

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Perform separation
    separate_audio(input_audio_path, output_dir)