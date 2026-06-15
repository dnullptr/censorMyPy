import os
import whisper
import librosa
import torch
import soundfile as sf
import asyncio
import string
import json
from pydub import AudioSegment
from shutil import rmtree
from module_context import ModuleContext
from faster_whisper import WhisperModel


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
    # 1. CACHE HANDLER
    # Checks if a pre-calculated timestamp list exists
    def _check_cache():
        if os.path.exists(f'{audio_file_path}.json'):
            print(f'[+] Using cached transcription from {audio_file_path}.json')
            with open(f'{audio_file_path}.json', 'r') as f:
                data = json.load(f) 
                return [tuple(item) for item in data]
        return None

    cached_timestamps = _check_cache()
    if cached_timestamps is not None:
        return cached_timestamps

    # 2. TRANSCRIPTION (Updated for Faster-Whisper)
    print(f'[+] Transcribing {audio_file_path} with word-level timestamps (Faster Engine)...')

    model = WhisperModel(
        "medium", 
        device="cuda", 
        compute_type="int8_float16"
    )

    segments, info = model.transcribe(
        audio_file_path,
        word_timestamps=True,
        beam_size=5
    )

    # 3. PREPROCESS WORDS
    # Convert generator to list to consume it completely before cleanup
    segments_list = list(segments)
    all_words = []   # each element: {'raw': str, 'clean': str, 'start': float, 'end': float}
    for segment in segments_list:
        if segment.words:
            for word_obj in segment.words:
                raw_word = word_obj.word
                clean_word = raw_word.lower().strip().strip(string.punctuation)
                all_words.append({
                    'raw': raw_word,
                    'clean': clean_word,
                    'start': word_obj.start,
                    'end': word_obj.end
                })

    # 4. PREPROCESS BAD WORDS: split each bad word into tokens (cleaned similarly)
    bad_phrases = []
    for phrase in bad_words:
        clean_phrase = phrase.lower().strip().strip(string.punctuation)
        tokens = clean_phrase.split()
        if tokens:  # ignore empty phrases
            bad_phrases.append(tokens)

    # 5. FIND PHRASE MATCHES
    n = len(all_words)
    intervals = []   # list of (start_idx, end_idx) in word indices

    for i in range(n):
        for tokens in bad_phrases:
            L = len(tokens)
            if i + L > n:
                continue
            # Check if the next L words match the tokens
            match = True
            for j in range(L):
                if all_words[i+j]['clean'] != tokens[j]:
                    match = False
                    break
            if match:
                intervals.append( (i, i+L-1) )

    # 6. CONVERT WORD INDICES TO TIME INTERVALS WITH BUFFER
    BUFFER_MS = 85   # same as before
    time_intervals = []
    for (start_idx, end_idx) in intervals:
        start_time = all_words[start_idx]['start']
        end_time   = all_words[end_idx]['end']
        # Convert to milliseconds and apply buffer
        start_time_ms = int(start_time * 1000) - BUFFER_MS
        end_time_ms   = int(end_time   * 1000) + BUFFER_MS
        if start_time_ms < 0:
            start_time_ms = 0
        time_intervals.append( (start_time_ms, end_time_ms) )

    # 7. MERGE OVERLAPPING OR ADJACENT INTERVALS
    if not time_intervals:
        bad_word_timestamps = []
    else:
        # Sort by start time
        time_intervals.sort(key=lambda x: x[0])
        merged = []
        current_start, current_end = time_intervals[0]
        for i in range(1, len(time_intervals)):
            s, e = time_intervals[i]
            if s <= current_end:   # overlapping or adjacent
                if e > current_end:
                    current_end = e
            else:
                merged.append( (current_start, current_end) )
                current_start, current_end = s, e
        merged.append( (current_start, current_end) )
        bad_word_timestamps = merged

    # 8. SAVE THE RESULTS TO JSON FOR CACHING
    with open(f'{audio_file_path}.json', 'w') as f:
        json.dump(bad_word_timestamps, f)
    print(f'[+] Saved transcription cache to {audio_file_path}.json')

    # 9. CLEAN UP MODEL TO FREE GPU MEMORY
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 10. RETURN THE TIMESTAMPS
    return _check_cache() or bad_word_timestamps

async def get_bad_word_and_slurs_timestamps(audio_file_path, bad_words, slurs):
    # 1. CACHE HANDLER
    # Checks if a pre-calculated timestamp list exists
    cache_file = f'{audio_file_path}_bad_slurs.json'
    if os.path.exists(cache_file):
        print(f'[+] Using cached transcription from {cache_file}')
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return [tuple(item) for item in data['bad_words']], [tuple(item) for item in data['slurs']]

    # 2. Load Faster-Whisper Model
    # Using 'int8_float16' for massive VRAM savings (1.5GB-ish on 8GB GPU)
    model = WhisperModel(
        "medium",
        device="cuda",
        compute_type="int8_float16"
    )

    print(f'[+] Transcribing {audio_file_path} for bad words and slurs...')

    # 3. Run Transcription
    # word_timestamps=True is mandatory for the 'surgical' data you need
    segments, info = model.transcribe(
        audio_file_path,
        word_timestamps=True,
        beam_size=5
    )

    # 4. PREPROCESS WORDS
    segments_list = list(segments)
    all_words = []   # each element: {'raw': str, 'clean': str, 'start': float, 'end': float}
    for segment in segments_list:
        if segment.words:
            for word_obj in segment.words:
                raw_word = word_obj.word
                clean_word = raw_word.lower().strip().strip(string.punctuation)
                all_words.append({
                    'raw': raw_word,
                    'clean': clean_word,
                    'start': word_obj.start,
                    'end': word_obj.end
                })

    # 5. PREPROCESS BAD WORDS AND SLURS: split each into tokens (cleaned similarly)
    def preprocess_terms(terms):
        phrases = []
        for term in terms:
            clean_term = term.lower().strip().strip(string.punctuation)
            tokens = clean_term.split()
            if tokens:  # ignore empty terms
                phrases.append(tokens)
        return phrases

    bad_phrases = preprocess_terms(bad_words)
    slur_phrases = preprocess_terms(slurs)

    # 6. FIND PHRASE MATCHES FOR BAD WORDS AND SLURS SEPARATELY
    n = len(all_words)
    bad_intervals = []   # list of (start_idx, end_idx) for bad words
    slur_intervals = []  # list of (start_idx, end_idx) for slurs

    for i in range(n):
        # Check bad words
        for tokens in bad_phrases:
            L = len(tokens)
            if i + L > n:
                continue
            match = True
            for j in range(L):
                if all_words[i+j]['clean'] != tokens[j]:
                    match = False
                    break
            if match:
                bad_intervals.append( (i, i+L-1) )
        # Check slurs
        for tokens in slur_phrases:
            L = len(tokens)
            if i + L > n:
                continue
            match = True
            for j in range(L):
                if all_words[i+j]['clean'] != tokens[j]:
                    match = False
                    break
            if match:
                slur_intervals.append( (i, i+L-1) )

    # 7. CONVERT WORD INDICES TO TIME INTERVALS WITH BUFFER
    BUFFER_MS = 85   # same as before
    def convert_intervals(intervals):
        time_intervals = []
        for (start_idx, end_idx) in intervals:
            start_time = all_words[start_idx]['start']
            end_time   = all_words[end_idx]['end']
            # Convert to milliseconds and apply buffer
            start_time_ms = int(start_time * 1000) - BUFFER_MS
            end_time_ms   = int(end_time   * 1000) + BUFFER_MS
            if start_time_ms < 0:
                start_time_ms = 0
            time_intervals.append( (start_time_ms, end_time_ms) )
        return time_intervals

    bad_time_intervals = convert_intervals(bad_intervals)
    slur_time_intervals = convert_intervals(slur_intervals)

    # 8. MERGE OVERLAPPING OR ADJACENT INTERVALS FOR EACH LIST
    def merge_intervals(intervals):
        if not intervals:
            return []
        # Sort by start time
        intervals.sort(key=lambda x: x[0])
        merged = []
        current_start, current_end = intervals[0]
        for i in range(1, len(intervals)):
            s, e = intervals[i]
            if s <= current_end:   # overlapping or adjacent
                if e > current_end:
                    current_end = e
            else:
                merged.append( (current_start, current_end) )
                current_start, current_end = s, e
        merged.append( (current_start, current_end) )
        return merged

    merged_bad = merge_intervals(bad_time_intervals)
    merged_slur = merge_intervals(slur_time_intervals)

    # 9. SAVE THE RESULTS TO JSON FOR CACHING
    cache_data = {
        'bad_words': merged_bad,
        'slurs': merged_slur
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)
    print(f'[+] Saved transcription cache to {cache_file}')

    # 10. CLEAN UP MODEL TO FREE GPU MEMORY
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 11. RETURN THE TIMESTAMPS
    return merged_bad, merged_slur

async def get_separated_paths(audio_file_path, both=False):
    """
    Get the paths for separated instrumental and vocal files.
    """
    import os
    filename = os.path.splitext(os.path.basename(audio_file_path))[0]  # without extension
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

    audio = AudioSegment.from_file(audio_file_path)
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
    # if the original file is wav, save as wav, otherwise save as mp3 with high bitrate
    if audio_file_path.endswith(".wav"):
        censored_audio.export(output_file, format="wav")
    else:
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

async def censor_with_instrumentals_and_downpitch(audio_file_path, bad_words, slurs, output_file="censored_output.mp3", sep_task : asyncio.Task = None):
    """
    Censors bad words by replacing vocal segments with instrumentals.
    """
    # Step 1: Wait for separation to complete if sep_task is provided
    instrumental_path, vocal_path = None, None
    if sep_task is not None:
        # Wait for the separation task to complete
        while not sep_task.done():
            await asyncio.sleep(1)
            instrumental_path, vocal_path = await get_separated_paths(audio_file_path, both=True)
        # One final check after task completes
        instrumental_path, vocal_path = await get_separated_paths(audio_file_path, both=True)
    else:
        # If no sep_task provided, get paths directly (for backward compatibility)
        instrumental_path, vocal_path = await get_separated_paths(audio_file_path, both=True)
    
    if not (instrumental_path and vocal_path):
        print(f'Error! Separated files not found. Had the separator not worked firstly?')
        return

    # Step 2: Transcribe vocals to find bad words and slurs
    print(f'[+] Transcribe vocals to find bad words and slurs in Progress..')
    both_timestamps = await get_bad_word_and_slurs_timestamps(audio_file_path, bad_words, slurs)
    bad_word_timestamps, slurs_timestamps = both_timestamps
    
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
            censored_segment = instrumental[start_time:end_time]
            censored_audio += censored_segment
        
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
    if audio_file_path.endswith(".wav"):
        censored_audio.export(output_file, format="wav")
    else:
        censored_audio.export(output_file, format="mp3", bitrate='320k')
    print(f"Censored audio saved to {output_file}")

async def censor_with_both_and_downpitch(audio_file_path, bad_words, slurs, output_file="censored_output.mp3", sep_task : asyncio.Task = None):
    """
    Censors bad words by replacing vocal segments with instrumentals.
    """
    while not sep_task.done():
        await asyncio.sleep(1)
        instrumental_path, vocal_path = await get_separated_paths(audio_file_path, both=True)

    # Step 2: Transcribe vocals to find bad words
    print(f'[+] Transcribe vocals to find bad words and slurs in Progress..')
    both_timestamps = await get_bad_word_and_slurs_timestamps(audio_file_path, bad_words, slurs)
    bad_word_timestamps, slurs_timestamps = both_timestamps
    
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
    if audio_file_path.endswith(".wav"):
        censored_audio.export(output_file, format="wav")
    else:
        censored_audio.export(output_file, format="mp3", bitrate='320k')
    print(f"Censored audio saved to {output_file}")

async def censor_with_backspin(audio_file_path, bad_words, output_file_path="censored_output.mp3"):
    # Oldest method in the book
    audio = AudioSegment.from_file(audio_file_path)
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
    # Load model with high-performance int8_float16 quantization
    model = WhisperModel(
        "medium", 
        device="cuda", 
        compute_type="int8_float16"
    )
    
    print(f"[#] Debug: Transcribing {audio_file_path} (Faster Engine)")
    
    # 1. Faster-Whisper returns a generator of segments
    segments, info = model.transcribe(
        audio_file_path, 
        word_timestamps=True,
        beam_size=5
    )

    print("Recognized words and their timestamps:")
    
    # 2. Iterate through the generator
    for segment in segments:
        # Print the full segment text for context
        print(f"\n--- Segment: {segment.text.strip()} ---")
        
        # 3. Access 'words' attribute (only exists if word_timestamps=True)
        if segment.words:
            for word_info in segment.words:
                # Note: Attributes are accessed with dot notation, not brackets
                start_time = word_info.start
                end_time = word_info.end
                text = word_info.word
                
                # 4. Print the granular timestamps
                print(f"   [{start_time:.2f}s -> {end_time:.2f}s]: {text}")

    print("\n[#] Debug: End of transcription.")
     
async def get_bad_word_timestamps_genai(audio_file_path, bad_words):
    # Using GenAI for transcription and not Whisper
    import json
    import genai
    bad_word_timestamps = []
    
    print(f'[+] GenAI toolset bridge function running..')
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