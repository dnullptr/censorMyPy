from pydub import AudioSegment
import os
import argparse

# Useful for uploading to Apple Music, iTunes, or any platform that supports ALAC (Apple Lossless Audio Codec)

def convert_wav_to_alac(input_file, output_file):
    """
    Converts a WAV file to an ALAC (M4A) file using pydub and FFmpeg.

    Args:
        input_file (str): Path to the input .wav file.
        output_file (str): Path for the output .m4a file.
    """
    if not input_file.lower().endswith('.wav'):
        print(f"Error: Input file must be a .wav file: {input_file}")
        return

    # Ensure output file has the correct .m4a extension
    if not output_file.lower().endswith('.m4a'):
        output_file = os.path.splitext(output_file)[0] + '.m4a'

    try:
        # Load the WAV file
        audio = AudioSegment.from_wav(input_file)
        
        # Export as ALAC within an M4A container
        # Use 'ipod' format for the M4A container and specify the 'alac' codec
        audio.export(
            output_file, 
            format="ipod", # pydub uses 'ipod' format for M4A files
            codec="alac"   # explicitly specify the Apple Lossless codec
        )
        
        print(f"Successfully converted '{input_file}' to '{output_file}'")

    except FileNotFoundError:
        print("Error: ffmpeg or ffprobe not found. Ensure FFmpeg is installed and in your PATH.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WAV to ALAC M4A")
    parser.add_argument('--file', required=True, help='Input WAV file path')
    args = parser.parse_args()

    input_wav = args.file
    output_alac = os.path.splitext(input_wav)[0] + '.m4a'

    convert_wav_to_alac(input_wav, output_alac)
