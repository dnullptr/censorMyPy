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
    # Load model
    model = whisper.load_model("large") 
    
    # --- 1. CACHE HANDLER (Unchanged) ---
    # Checks if a pre-calculated timestamp list exists
    if os.path.exists(f'{audio_file_path}.json'):
        print(f'[+] Using cached transcription from {audio_file_path}.json')
        with open(f'{audio_file_path}.json', 'r') as f:
            data = json.load(f) 
            return [tuple(item) for item in data]
            
    # --- 2. TRANSCRIPTION (Updated) ---
    else:
        print(f'[+] Transcribing {audio_file_path} with word-level timestamps...')
        # CRITICAL CHANGE: word_timestamps=True allows access to individual word timing
        result = model.transcribe(audio_file_path, fp16=False, word_timestamps=True)

    bad_word_timestamps = []
    
    print(f'[+] Precision segmentation running...')

    # --- 3. SURGICAL FILTERING (New Logic) ---
    import string
    for segment in result['segments']:
        # Dig into the 'words' list instead of just the segment text
        for word_obj in segment['words']:
            
            # Extract the raw word string
            raw_word = word_obj['word']
            
            # CLEANUP: 
            # 1. Lowercase
            # 2. Strip whitespace
            # 3. Strip punctuation (so "sh*t!" becomes "sh*t")
            clean_word = raw_word.lower().strip().strip(string.punctuation)
            
            # Check against your list
            if clean_word in bad_words:
                
                # Convert to milliseconds (int)
                start_time = int(word_obj['start'] * 1000)
                end_time = int(word_obj['end'] * 1000)
                
                # Optional: Added a tiny buffer (50ms is enough) as the censor is too super accurate now lol
                BUFFER_MS = 50
                start_time = max(0, start_time - BUFFER_MS)
                end_time = end_time + BUFFER_MS
                
                bad_word_timestamps.append((start_time, end_time))


    return bad_word_timestamps

async def get_bad_word_and_slurs_timestamps(audio_file_path, bad_words, slurs):
    # Load model
    model = whisper.load_model("large") 
    
    # Enable word_timestamps to get the "surgical" data
    print(f'[+] Transcribing {audio_file_path} for bad words and slurs...')
    result = model.transcribe(audio_file_path, fp16=False, word_timestamps=True)
    
    bad_word_timestamps = []
    slurs_timestamps = []

    # Buffer settings (in milliseconds) - Adjust these if it feels too tight
    BUFFER_MS = 50 
    import string
    
    print(f'[+] Precision segmentation running..')
    
    for segment in result['segments']:
        # Dig into the words list
        for word_obj in segment['words']:
            
            # Clean the word (lowercase, strip whitespace, strip punctuation)
            raw_word = word_obj['word']
            clean_word = raw_word.lower().strip().strip(string.punctuation)
            
            # Get timestamps
            start_time = int(word_obj['start'] * 1000)
            end_time = int(word_obj['end'] * 1000)

            # Apply buffer (padding)
            # Ensure start_time doesn't go below 0
            start_time = max(0, start_time - BUFFER_MS) 
            end_time = end_time + BUFFER_MS
            
            # Check 1: Bad Words
            if clean_word in bad_words:
                bad_word_timestamps.append((start_time, end_time))
            
            # Check 2: Slurs (Independent check, so a word can technically be both)
            if clean_word in slurs:
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

async def censor_with_instrumentals(audio_file_path, bad_words, output_file="censored_output.mp3", sep_task : asyncio.Task = None, genai=False):
    """
    Censors bad words by replacing vocal segments with instrumentals.
    """
    instrumental_path = await get_separated_paths(audio_file_path)
    

    # Step 2: Transcribe vocals to find bad words
    print(f'[+] Transcribe vocals to find bad words in Progress..')
    if genai:
        bad_word_timestamps = await get_bad_word_timestamps_genai(audio_file_path, bad_words)
    else:
        bad_word_timestamps = await get_bad_word_timestamps(audio_file_path, bad_words)

   # Step 3: Block the code until the paths are found (from the separator simultaneously running thread)
    # Wait up to 60 seconds for instrumental_path to become available
    import time
    timeout = 60  # seconds
    start_time = time.time()
    while instrumental_path is None:
        await asyncio.sleep(1)
        instrumental_path = await get_separated_paths(audio_file_path)
    if not instrumental_path:
        print(f'Error! Separated instrumental not found after waiting. Had the separator not worked firstly?')
        return

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

async def censor_with_both(audio_file_path, bad_words, output_file="censored_output.mp3", sep_task : asyncio.Task = None):
    """
    Censors bad words by reversing vocal segments with the song original instrumentals.
    """
    instrumental_path, vocal_path = None,None
    # Step 1: Block the code until the paths are found (from the separator simultaneously running thread)
    while not sep_task.done():
        await asyncio.sleep(1)
        instrumental_path, vocal_path = await get_separated_paths(audio_file_path, both=True)
    
    # Step 2: Transcribe vocals to find bad words
    print(f'[+] Transcribe vocals to find bad words in Progress..')
    bad_word_timestamps = await get_bad_word_timestamps(audio_file_path, bad_words)

    if not (instrumental_path and vocal_path):
        print(f'Error! Separated files not found. Had the separator not worked firstly?')
        return

    audio = AudioSegment.from_file(audio_file_path)
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
    if audio_file_path.endswith(".wav"):
        censored_audio.export(output_file, format="wav")
    else:
        censored_audio.export(output_file, format="mp3", bitrate='320k')
    print(f"Censored audio saved to {output_file}")

async def censor_with_downpitch(audio_file_path, bad_words, output_file="censored_output.mp3", sep_task : asyncio.Task = None):
    """
    Censors bad words by downpitching vocal segments with the song original instrumentals.
    """
    instrumental_path, vocal_path = None,None
    # Step 1: Block the code until the paths are found (from the separator simultaneously running thread)
    while not sep_task.done():
            await asyncio.sleep(1)
            instrumental_path, vocal_path = await get_separated_paths(audio_file_path, both=True)

    # Step 2: Transcribe vocals to find bad words
    print(f'[+] Transcribe vocals to find bad words in Progress..')
    bad_word_timestamps = await get_bad_word_timestamps(audio_file_path, bad_words)

    if not (instrumental_path and vocal_path):
        print(f'Error! Separated files not found. Had the separator not worked firstly?')
        return

    audio = AudioSegment.from_file(audio_file_path)
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
        cur_vocal_to_downpitch.export('temp.wav',format="wav")
        print(f"[-] Calling downpitch... ")
        
        await down_pitch('temp.wav','down_temp.wav',semitones=10) # 10 semi-tones should be enough to sound screwed.
        print(f"[-] Mixing segment as censored...")
        downpitched = AudioSegment.from_file('down_temp.wav')
        censored_audio += censored_segment.overlay(downpitched)

        # Update the end time of the last processed segment
        previous_end_time = end_time

    # Add the remaining audio after the last bad word
    censored_audio += audio[previous_end_time:]

    # Save the censored audio to the output file
    if audio_file_path.endswith(".wav"):
        censored_audio.export(output_file, format="wav")
    else:
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

async def censor_with_both_and_downpitch(audio_file_path, bad_words, slurs, output_file="censored_output.mp3", sep_task : asyncio.Task = None):
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

    audio = AudioSegment.from_file(audio_file_path)
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
            censored_segment : AudioSegment = instrumental[start_time:end_time]
            censored_audio += censored_segment.overlay(vocals[start_time:end_time].reverse())

        else:
            censored_audio += audio[previous_end_time:start_time]
            print(f"[+] Processing slur segment: {start_time} ms to {end_time} ms")
            # Reverse only the segment containing the bad word
            censored_segment : AudioSegment = instrumental[start_time:end_time]

            print(f"[-] Preparing current segment for down-pitch..")
            cur_vocal_to_downpitch = vocals[start_time:end_time]
            cur_vocal_to_downpitch.export('temp.wav',format="wav")
            print(f"[-] Calling downpitch... ")
            
            await down_pitch('temp.wav','down_temp.wav',semitones=10) # 10 semi-tones should be enough to sound screwed.
            print(f"[-] Mixing segment as censored...")
            downpitched = AudioSegment.from_file('down_temp.wav')
            censored_audio += censored_segment.overlay(downpitched)

        # Update the end time of the last processed segment
        previous_end_time = end_time

    # Add the remaining audio after the last bad word
    censored_audio += audio[previous_end_time:]

    # Save the censored audio to the output file
    censored_audio.export(output_file, format="wav")
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
    model = whisper.load_model("large") 
    
    # 1. Ensure word_timestamps is True
    result = model.transcribe(audio_file_path, fp16=False, word_timestamps=True)

    print("Recognized words and their timestamps:")
    
    for segment in result['segments']:
        # Optional: Print the full sentence first so you know context
        print(f"\n--- Segment: {segment['text'].strip()} ---")
        
        # 2. THE FIX: Iterate through the 'words' list inside the segment
        for word_info in segment['words']:
            start_time = word_info['start']
            end_time = word_info['end']
            text = word_info['word']
            
            # 3. Print the granular timestamps
            print(f"   [{start_time:.2f}s -> {end_time:.2f}s]: {text}")
     
async def get_bad_word_timestamps_genai(audio_file_path, bad_words):
    # Using GenAI for transcription and not Whisper
    import json
    import genai
    bad_word_timestamps = []
    
   
    # check if transcription.json exists, if not, print error and exit
    if not os.path.exists('transcription.json'):
        print(f'Error! transcription.json not found. Running transcription..')
        await genai.transcribe_audio_file(audio_file_path, 'transcription.json')

    with open('transcription.json', 'r') as f:
        result = json.load(f)
    # If your JSON has a list of words with 'start', 'end' (in ms), and 'text'
    for word in result: 
        word_text = word['text'].lower()
        if any(bad_word in word_text for bad_word in bad_words):
            start_time = int(word['start'])  # in sec
            end_time = int(word['end'])      # in sec
            bad_word_timestamps.append((start_time, end_time))

    return bad_word_timestamps

async def cleanup():
    print(f'[=] Running clean-up..')
    files = ['down_temp.wav','down_temp.mp3','temp.wav','temp.mp3']
    for file in files:
        if os.path.exists(file):
            os.remove(file)
    if os.path.exists('separated'):
        rmtree('separated')

async def run_in_thread(coro):
    await asyncio.to_thread(asyncio.run, coro)