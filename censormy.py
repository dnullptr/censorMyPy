import whisper
import argparse
import os
from pydub import AudioSegment
from spleeter.separator import Separator

model = whisper.load_model("large")  # "small", "medium", "large" for better accuracy, I can use "base" but it's shitty

def separate_audio(input_audio_path, output_dir="separated"):
    """
    Separates the input audio into vocals and instrumental using Spleeter.
    """
    separator = Separator('spleeter:2stems')  # 2 stems: vocals + instrumental
    separator.separate_to_file(input_audio_path, output_dir)
    return f"{output_dir}/separated_audio/vocals.wav", f"{output_dir}/separated_audio/accompaniment.wav"

def censor_with_instrumentals(audio_file_path, bad_words, output_file="censored_output.mp3"):
    """
    Censors bad words by replacing vocal segments with instrumentals.
    """
    # Step 1: Separate vocals and instrumentals
    print(f'[+] Separation in Progress..')
    if os.path.exists(f'separated_audio/song/accompaniment.wav'):
        instrumental_path = f'separated_audio/song/accompaniment.wav'
    else:
        print(f'Error! Separated intrumental not found. Had the separator not worked firstly?')
    

    # Step 2: Transcribe vocals to find bad words
    print(f'[+] Transcribe vocals to find bad words in Progress..')
    bad_word_timestamps = get_bad_word_timestamps(audio_file_path, bad_words)

#try start
    
    audio = AudioSegment.from_mp3(audio_file_path)
    instrumental = AudioSegment.from_file(instrumental_path)
    censored_audio = AudioSegment.empty()  # Start with an empty audio segment
    previous_end_time = 0  # Keep track of the end of the last processed segment
  
    # Process each bad word segment
    for start_time, end_time in bad_word_timestamps:
        # Add the audio before the bad word
        censored_audio += audio[previous_end_time:start_time]
        print(f"[-] Processing segment: {start_time} ms to {end_time} ms")
        # Reverse only the segment containing the bad word
        censored_segment = instrumental[start_time:end_time]
        censored_audio += censored_segment

        # Update the end time of the last processed segment
        previous_end_time = end_time

    # Add the remaining audio after the last bad word
    censored_audio += audio[previous_end_time:]

    # Save the censored audio to the output file
    censored_audio.export(output_file, format="mp3")
    print(f"Censored audio saved to {output_file}")
    

def get_bad_word_timestamps(audio_file_path, bad_words):
    result = model.transcribe(audio_file_path, fp16=False)
    bad_word_timestamps = []
    
    # Check for bad words in the segments, probably 5-sec ones.
    print(f'[+] Bad words segmentation method running..')
    for segment in result['segments']:
        start_time = int(segment['start'] * 1000)  # ms
        end_time = int(segment['end'] * 1000)
        if any(bad_word in segment['text'].lower() for bad_word in bad_words):
            bad_word_timestamps.append((start_time, end_time))

    return bad_word_timestamps

def print_transcribed_words(audio_file_path):
    # Transcribe the audio using Whisper
    result = model.transcribe(audio_file_path, fp16=False)

    print("Recognized words and their timestamps:")
    for segment in result['segments']:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        print(f"From {start_time:.2f}s to {end_time:.2f}s: {text}")

# Function to censor bad words in audio with a backspin effect
def censor_with_backspin(audio_file_path, bad_words, output_file_path="censored_output.mp3"):
    audio = AudioSegment.from_mp3(audio_file_path)
    print(f'[+] Transcribe vocals to find bad words in Progress..')
    bad_word_timestamps = get_bad_word_timestamps(audio_file_path, bad_words)
   

    censored_audio = AudioSegment.empty()  # Start with an empty audio segment
    previous_end_time = 0  # Keep track of the end of the last processed segment
  
    # Process each bad word segment
    for start_time, end_time in bad_word_timestamps:
        # Add the audio before the bad word
        censored_audio += audio[previous_end_time:start_time]
        print(f"[-] Processing segment: {start_time} ms to {end_time} ms")
        # Reverse only the segment containing the bad word
        censored_segment = audio[start_time:end_time].reverse()
        censored_audio += censored_segment

        # Update the end time of the last processed segment
        previous_end_time = end_time

    # Add the remaining audio after the last bad word
    censored_audio += audio[previous_end_time:]

    # Save the censored audio to the output file
    censored_audio.export(output_file_path, format="mp3")
    print(f"Censored audio saved to {output_file_path}")
    


def main():
    parser = argparse.ArgumentParser(description="Kudsha's Sound System")
    parser.add_argument("audio_file",
        default="song.mp3",
        help="Path to the audio file to be censored. Will use 'song.mp3' as default")
    parser.add_argument("bad_words_file",help="Path to the bad words file.")
    parser.add_argument(
        "--method",
        choices=["v", "b"],
        required=True,
        help="Censorship method: 'v' for vocal separation, 'b' for backspin.",
    )
    parser.add_argument("--output", default="censored_output.mp3", help="Output file path.")
    args = parser.parse_args()


    # Read bad words from file
    with open(args.bad_words_file, "r") as f:
        bad_words = [line.strip().lower() for line in f]

    if args.method == "v":
        print("Using vocal separation method...")
        separate_audio(args.audio_file)
        #censor_with_instrumentals(args.audio_file, bad_words, args.output)
    elif args.method == "b":
        print("Using backspin method...")
        censor_with_backspin(args.audio_file, bad_words, args.output)

    

if __name__ == "__main__":
    main()
    