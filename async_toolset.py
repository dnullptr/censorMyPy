import os
import whisper
import librosa
import soundfile as sf
import asyncio
from pydub import AudioSegment
from shutil import rmtree
from module_context import ModuleContext


async def separate_audio(input_audio_path, output_dir="separated"):
    """
    Separates the input audio into vocals and instrumental using Spleeter.
    """
    print(f'[+] Separation in Progress..')
    with ModuleContext("spleeter.separator") as modules:
        Separator = modules["spleeter.separator"].Separator
        separator = Separator('spleeter:2stems-16kHz')  # 2 stems: vocals + instrumental
        separator.separate_to_file(input_audio_path, output_dir)
        return f"{output_dir}/separated_audio/vocals.wav", f"{output_dir}/separated_audio/accompaniment.wav"

async def down_pitch(input_path, output_path, semitones):
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

async def get_bad_word_timestamps(audio_file_path, bad_words):
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

async def get_bad_word_and_slurs_timestamps(audio_file_path, bad_words, slurs):
    model = whisper.load_model("large")  # "small", "medium", "large" for better accuracy, I can use "base" but it's shitty
    result = model.transcribe(audio_file_path, fp16=False)
    bad_word_timestamps = []
    slurs_timestamps = []
    
    # Check for bad words in the segments
    print(f'[+] Bad words segmentation method running..')
    for segment in result['segments']:
        start_time = int(segment['start'] * 1000)  # ms
        end_time = int(segment['end'] * 1000)
        if any(bad_word in segment['text'].lower() for bad_word in bad_words):
            bad_word_timestamps.append((start_time, end_time))
        if any(slur in segment['text'].lower() for slur in slurs):
            slurs_timestamps.append((start_time, end_time))

    return bad_word_timestamps, slurs_timestamps

async def get_separated_paths(audio_file_path, both=False):
    """
    Get the paths for separated instrumental and vocal files.
    """
    filename = audio_file_path.split('.')[0]  # without extension
    instrumental_path = f'separated/{filename}/accompaniment.wav'
    vocal_path = f'separated/{filename}/vocals.wav'

    if both:
        if os.path.exists(instrumental_path) and os.path.exists(vocal_path):
            return instrumental_path, vocal_path
        else:
            return None, None
    else:
        if os.path.exists(instrumental_path):
            return instrumental_path
        else:
            return None

async def censor_with_instrumentals(audio_file_path, bad_words, output_file="censored_output.mp3"):
    """
    Censors bad words by replacing vocal segments with instrumentals.
    """
    instrumental_path = await get_separated_paths(audio_file_path)
    if not instrumental_path:
        print(f'Error! Separated instrumental not found. Had the separator not worked firstly?')
        return

    # Step 2: Transcribe vocals to find bad words
    print(f'[+] Transcribe vocals to find bad words in Progress..')
    bad_word_timestamps = await get_bad_word_timestamps(audio_file_path, bad_words)

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

async def censor_with_both(audio_file_path, bad_words, output_file="censored_output.mp3"):
    """
    Censors bad words by reversing vocal segments with the song original instrumentals.
    """
    instrumental_path, vocal_path = await get_separated_paths(audio_file_path, both=True)
    if not (instrumental_path and vocal_path):
        print(f'Error! Separated files not found. Had the separator not worked firstly?')
        return

    # Step 2: Transcribe vocals to find bad words
    print(f'[+] Transcribe vocals to find bad words in Progress..')
    bad_word_timestamps = await get_bad_word_timestamps(audio_file_path, bad_words)

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

async def censor_with_downpitch(audio_file_path, bad_words, output_file="censored_output.mp3"):
    """
    Censors bad words by downpitching vocal segments with the song original instrumentals.
    """
    instrumental_path, vocal_path = await get_separated_paths(audio_file_path, both=True)
    if not (instrumental_path and vocal_path):
        print(f'Error! Separated files not found. Had the separator not worked firstly?')
        return

    # Step 2: Transcribe vocals to find bad words
    print(f'[+] Transcribe vocals to find bad words in Progress..')
    bad_word_timestamps = await get_bad_word_timestamps(audio_file_path, bad_words)

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
        
        await down_pitch('temp.mp3','down_temp.mp3',semitones=10) # 10 semi-tones should be enough to sound screwed.
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

async def censor_with_instrumentals_and_downpitch(audio_file_path, bad_words, slurs, output_file="censored_output.mp3"):
    """
    Censors bad words by replacing vocal segments with instrumentals.
    """
    

    # Step 2: Transcribe vocals to find bad words
    print(f'[+] Transcribe vocals to find bad words and slurs in Progress..')
    both_timestamps = await get_bad_word_and_slurs_timestamps(audio_file_path, bad_words, slurs)
    bad_word_timestamps, slurs_timestamps = both_timestamps
    
    instrumental_path, vocal_path = await get_separated_paths(audio_file_path, both=True)
    if not (instrumental_path and vocal_path):
        print(f'Error! Separated files not found. Had the separator not worked firstly?')
        return

    audio = AudioSegment.from_mp3(audio_file_path)
    instrumental = AudioSegment.from_file(instrumental_path)
    vocals = AudioSegment.from_file(vocal_path)

    censored_audio = AudioSegment.empty()  # Start with an empty audio segment
    previous_end_time = 0  # Keep track of the end of the last processed segment
  
    # Process each bad word segment
    for start_time, end_time in sorted(bad_word_timestamps + slurs_timestamps):
        # Add the audio before the bad word
        if (start_time, end_time) in bad_word_timestamps:
            censored_audio += audio[previous_end_time:start_time]
            print(f"[+] Processing bad word segment: {start_time} ms to {end_time} ms")
            # Reverse only the segment containing the bad word
            censored_segment = instrumental[start_time:end_time]
            censored_audio += censored_segment

        else:
            censored_audio += audio[previous_end_time:start_time]
            print(f"[+] Processing slur segment: {start_time} ms to {end_time} ms")
            # Reverse only the segment containing the bad word
            censored_segment : AudioSegment = instrumental[start_time:end_time]

            print(f"[-] Preparing current segment for down-pitch..")
            cur_vocal_to_downpitch = vocals[start_time:end_time]
            cur_vocal_to_downpitch.export('temp.mp3',format="mp3",bitrate='320k')
            print(f"[-] Calling downpitch... ")
            
            await down_pitch('temp.mp3','down_temp.mp3',semitones=10) # 10 semi-tones should be enough to sound screwed.
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

async def censor_with_backspin(audio_file_path, bad_words, output_file_path="censored_output.mp3"):
    # Oldest method in the book
    audio = AudioSegment.from_mp3(audio_file_path)
    print(f'[+] Transcribe vocals to find bad words in Progress..')
    bad_word_timestamps = await get_bad_word_timestamps(audio_file_path, bad_words)
   

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

async def print_transcribed_words(audio_file_path):
    # Transcribe the audio using Whisper
    model = whisper.load_model("large")  # "small", "medium", "large" for better accuracy, I can use "base" but it's shitty
    result = model.transcribe(audio_file_path, fp16=False, word_timestamps=True)

    print("Recognized words and their timestamps:")
    for segment in result['segments']:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        print(f"From {start_time:.2f}s to {end_time:.2f}s: {text}")

async def cleanup():
    print(f'[=] Running clean-up..')
    if os.path.exists('down_temp.mp3'):
        os.remove('down_temp.mp3')
    if os.path.exists('separated'):
        rmtree('separated')

async def run_in_thread(coro):
    await asyncio.to_thread(asyncio.run, coro)