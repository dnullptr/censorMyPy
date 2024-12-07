import whisper
import argparse
import os
import librosa
import soundfile as sf
from pydub import AudioSegment

from module_context import ModuleContext


def separate_audio(input_audio_path, output_dir="separated"):
    """
    Separates the input audio into vocals and instrumental using Spleeter.
    """
    with ModuleContext("spleeter.separator") as modules:
        Separator = modules["spleeter.separator"].Separator
        separator = Separator('spleeter:2stems')  # 2 stems: vocals + instrumental
        separator.separate_to_file(input_audio_path, output_dir)
        return f"{output_dir}/separated_audio/vocals.wav", f"{output_dir}/separated_audio/accompaniment.wav"

def down_pitch(input_path, output_path, semitones):
    """
    Down-pitch an audio file by a given number of semitones using librosa.
    :param input_path: Path to the input audio file.
    :param output_path: Path to save the down-pitched audio.
    :param semitones: Number of semitones to shift down (positive for down-pitching).
    """
    # Load the audio file
    y, sr = librosa.load(input_path, sr=None)

    # Down-pitch the audio
    y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-semitones)
    print(f"[-] Down-shifted the pitch, saving..")
    # Save the processed audio
    sf.write(output_path, y_shifted, sr)

def censor_with_instrumentals(audio_file_path, bad_words, output_file="censored_output.mp3"):
    """
    Censors bad words by replacing vocal segments with instrumentals.
    """
    # Step 1: Separate vocals and instrumentals
    print(f'[+] Separation in Progress..')
    if os.path.exists(f'separated/song/accompaniment.wav'):
        instrumental_path = f'separated/song/accompaniment.wav'
    else:
        print(f'Error! Separated intrumental not found. Had the separator not worked firstly?')
    

    # Step 2: Transcribe vocals to find bad words
    print(f'[+] Transcribe vocals to find bad words in Progress..')
    bad_word_timestamps = get_bad_word_timestamps(audio_file_path, bad_words)

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
    censored_audio.export(output_file, format="mp3", bitrate='320k')
    print(f"Censored audio saved to {output_file}")

def censor_with_both(audio_file_path, bad_words, output_file="censored_output.mp3"):
    """
    Censors bad words by reversing vocal segments with the song original instrumentals.
    """
    # Step 1: Separate vocals and instrumentals
    print(f'[+] Separation in Progress..')
    if all([os.path.exists(f'separated/song/accompaniment.wav'), os.path.exists(f'separated/song/vocals.wav')]):
        instrumental_path = f'separated/song/accompaniment.wav'
        vocal_path = f'separated/song/vocals.wav'
    else:
        print(f'Error! Separated files not found. Had the separator not worked firstly?')
    
    

    # Step 2: Transcribe vocals to find bad words
    print(f'[+] Transcribe vocals to find bad words in Progress..')
    bad_word_timestamps = get_bad_word_timestamps(audio_file_path, bad_words)

    audio = AudioSegment.from_mp3(audio_file_path)
    instrumental = AudioSegment.from_file(instrumental_path)
    vocals = AudioSegment.from_file(vocal_path)

    censored_audio = AudioSegment.empty()  # Start with an empty audio segment
    previous_end_time = 0  # Keep track of the end of the last processed segment
  
    # Process each bad word segment
    for start_time, end_time in bad_word_timestamps:
        # Add the audio before the bad word
        censored_audio += audio[previous_end_time:start_time]
        print(f"[-] Processing segment: {start_time} ms to {end_time} ms")
        # Reverse only the segment containing the bad word
        censored_segment : AudioSegment = instrumental[start_time:end_time]
        censored_audio += censored_segment.overlay(vocals[start_time:end_time].reverse())

        # Update the end time of the last processed segment
        previous_end_time = end_time

    # Add the remaining audio after the last bad word
    censored_audio += audio[previous_end_time:]

    # Save the censored audio to the output file
    censored_audio.export(output_file, format="mp3", bitrate='320k')
    print(f"Censored audio saved to {output_file}")

def censor_with_downpitch(audio_file_path, bad_words, output_file="censored_output.mp3"):
    """
    Censors bad words by downpitching vocal segments with the song original instrumentals.
    """
    # Step 1: Separate vocals and instrumentals
    print(f'[+] Separation in Progress..')
    if all([os.path.exists(f'separated/song/accompaniment.wav'), os.path.exists(f'separated/song/vocals.wav')]):
        instrumental_path = f'separated/song/accompaniment.wav'
        vocal_path = f'separated/song/vocals.wav'
    else:
        print(f'Error! Separated files not found. Had the separator not worked firstly?')
    
    

    # Step 2: Transcribe vocals to find bad words
    print(f'[+] Transcribe vocals to find bad words in Progress..')
    bad_word_timestamps = get_bad_word_timestamps(audio_file_path, bad_words)

    audio = AudioSegment.from_mp3(audio_file_path)
    instrumental = AudioSegment.from_file(instrumental_path)
    vocals = AudioSegment.from_file(vocal_path)

    censored_audio = AudioSegment.empty()  # Start with an empty audio segment
    previous_end_time = 0  # Keep track of the end of the last processed segment
  
    # Process each bad word segment
    for start_time, end_time in bad_word_timestamps:
        # Add the audio before the bad word
        censored_audio += audio[previous_end_time:start_time]
        print(f"[+] Processing segment: {start_time} ms to {end_time} ms")
        # Reverse only the segment containing the bad word
        censored_segment : AudioSegment = instrumental[start_time:end_time]

        print(f"[-] Preparing current segment for down-pitch..")
        cur_vocal_to_downpitch = vocals[start_time:end_time]
        cur_vocal_to_downpitch.export('temp.mp3',format="mp3",bitrate='320k')
        print(f"[-] Calling downpitch... ")
        
        down_pitch('temp.mp3','down_temp.mp3',semitones=12) # 12 semi-tones should be enough to sound screwed.
        print(f"[-] Mixing segment as censored...")
        downpitched = AudioSegment.from_file('down_temp.mp3')
        censored_audio += censored_segment.overlay(downpitched)

        # Update the end time of the last processed segment
        previous_end_time = end_time

    # Add the remaining audio after the last bad word
    censored_audio += audio[previous_end_time:]

    # Save the censored audio to the output file
    censored_audio.export(output_file, format="mp3", bitrate='320k')
    print(f"Censored audio saved to {output_file}")

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
    
def get_bad_word_timestamps(audio_file_path, bad_words):

    model = whisper.load_model("large")  # "small", "medium", "large" for better accuracy, I can use "base" but it's shitty
    result = model.transcribe(audio_file_path, fp16=False)
    bad_word_timestamps = []
    
    # Check for bad words in the segments
    print(f'[+] Bad words segmentation method running..')
    for segment in result['segments']:
        start_time = int(segment['start'] * 1000)  # ms
        end_time = int(segment['end'] * 1000)
        if any(bad_word in segment['text'].lower() for bad_word in bad_words):
            bad_word_timestamps.append((start_time, end_time))

    return bad_word_timestamps

def print_transcribed_words(audio_file_path):
    # Transcribe the audio using Whisper
    model = whisper.load_model("large")  # "small", "medium", "large" for better accuracy, I can use "base" but it's shitty
    result = model.transcribe(audio_file_path, fp16=False,word_timestamps=True)

    print("Recognized words and their timestamps:")
    for segment in result['segments']:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        print(f"From {start_time:.2f}s to {end_time:.2f}s: {text}")



def main():
    parser = argparse.ArgumentParser(description="Kudsha's Sound System")
    parser.add_argument("audio_file",
        default="song.mp3",
        help="Path to the audio file to be censored. Will use 'song.mp3' as default")
    parser.add_argument("bad_words_file",help="Path to the bad words file.")
    parser.add_argument(
        "--method",
        choices=["v", "b", "vb", "p"],
        required=True,
        help="Censorship method: 'v' for vocal separation, 'b' for backspin, 'vb' for combination of both, 'p' for down-pitch.",
    )
    parser.add_argument("--output", default="censored_output.mp3", help="Output file path.")
    args = parser.parse_args()


    # Read bad words from file
    with open(args.bad_words_file, "r") as f:
        bad_words = [line.strip().lower() for line in f]

    if args.method == "v":
        print("Using vocal separation method...")
        separate_audio(args.audio_file)
        censor_with_instrumentals(args.audio_file, bad_words, args.output)

    elif args.method == "b":
        print("Using backspin method...")
        censor_with_backspin(args.audio_file, bad_words, args.output)

    elif args.method == "vb":
        print("Using vocal + backspin method...")
        separate_audio(args.audio_file)
        censor_with_both(args.audio_file, bad_words, args.output)
    
    elif args.method == "p":
        print("Using vocal downpitch method...")
        separate_audio(args.audio_file)
        censor_with_downpitch(args.audio_file, bad_words, args.output)

if __name__ == "__main__":
   main()