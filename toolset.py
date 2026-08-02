import whisper
import os
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from shutil import rmtree
from module_context import ModuleContext



def separate_audio(input_audio_path, output_dir="separated"):
    """
    Separates the input audio into vocals and instrumental using Spleeter.
    """
    print(f'[+] Separation in Progress..')
    with ModuleContext("spleeter.separator") as modules:
        Separator = modules["spleeter.separator"].Separator
        separator = Separator('spleeter:2stems-16kHz')  # 2 stems: vocals + instrumental
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
    # Step 1: Separated vocals and instrumentals should be in their dir.
    filename = audio_file_path.split('.')[0] # without extension
    if os.path.exists(f'separated/{filename}/accompaniment.wav'):
        instrumental_path = f'separated/{filename}/accompaniment.wav'
    else:
        print(f'Error! Separated intrumental not found. Had the separator not worked firstly?')
        exit()
    

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
    # Step 1: Separated vocals and instrumentals should be in their dir.
    filename = audio_file_path.split('.')[0] # without extension
    if all([os.path.exists(f'separated/{filename}/accompaniment.wav'), os.path.exists(f'separated/{filename}/vocals.wav')]):
        instrumental_path = f'separated/{filename}/accompaniment.wav'
        vocal_path = f'separated/{filename}/vocals.wav'
    else:
        print(f'Error! Separated files not found. Had the separator not worked firstly?')
        exit()
    
    

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
    # Step 1: Separated vocals and instrumentals should be in their dir.
    filename = audio_file_path.split('.')[0] # without extension
    if all([os.path.exists(f'separated/{filename}/accompaniment.wav'), os.path.exists(f'separated/{filename}/vocals.wav')]):
        instrumental_path = f'separated/{filename}/accompaniment.wav'
        vocal_path = f'separated/{filename}/vocals.wav'
    else:
        print(f'Error! Separated files not found. Had the separator not worked firstly?')
        exit()
    

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
        
        down_pitch('temp.mp3','down_temp.mp3',semitones=10) # 10 semi-tones should be enough to sound screwed.
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

def censor_with_instrumentals_and_downpitch(audio_file_path, bad_words, slurs, output_file="censored_output.mp3"):
    """
    Censors bad words by replacing vocal segments with instrumentals.
    """
     # Step 1: Separated vocals and instrumentals should be in their dir.
    filename = audio_file_path.split('.')[0] # without extension
    if all([os.path.exists(f'separated/{filename}/accompaniment.wav'), os.path.exists(f'separated/{filename}/vocals.wav')]):
        instrumental_path = f'separated/{filename}/accompaniment.wav'
        vocal_path = f'separated/{filename}/vocals.wav'
    else:
        print(f'Error! Separated files not found. Had the separator not worked firstly?')
        exit()
    
    
    
    # Step 2: Transcribe vocals to find bad words
    print(f'[+] Transcribe vocals to find bad words and slurs in Progress..')
    both_timestamps = get_bad_word_and_slurs_timestamps(audio_file_path, bad_words, slurs)
    bad_word_timestamps, slurs_timestamps = both_timestamps
    
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
            
            down_pitch('temp.mp3','down_temp.mp3',semitones=10) # 10 semi-tones should be enough to sound screwed.
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
    # Oldest method in the book
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

def apply_tape_stop_effect(
    segment: AudioSegment,
    fade_in_ms: int = 5,
    fade_out_ms: int = 15,
    speed_end: float = 0.25,
    curve_exponent: float = 1.5
) -> AudioSegment:
    """
    Applies a fluent, musical DJ Tape Stop / Vinyl Break pitch-drop effect
    to an AudioSegment buffer, supporting both Mono and Stereo channels.
    Includes gain normalization to prevent volume boosting/clipping,
    and micro fade-in/fade-out for smooth transitions.
    """
    if len(segment) == 0:
        return segment

    # Extract raw PCM samples as float64 for high precision processing
    orig_raw = np.array(segment.get_array_of_samples())
    samples = orig_raw.astype(np.float64)
    num_channels = segment.channels

    if num_channels > 1:
        samples = samples.reshape((-1, num_channels))

    num_frames = len(samples)
    if num_frames < 2:
        return segment

    # Calculate input peak to preserve natural volume level without boosting/clipping
    orig_peak = np.max(np.abs(samples))
    if orig_peak == 0:
        return segment

    # Normalized time u in [0, 1]
    u = np.linspace(0.0, 1.0, num_frames)

    # Smooth curve for speed decay from 1.0 down to speed_end
    # speed(u) = 1.0 - (1.0 - speed_end) * (u ** curve_exponent)
    p = curve_exponent + 1.0
    phi = u - ((1.0 - speed_end) / p) * (u ** p)
    phi_max = 1.0 - ((1.0 - speed_end) / p)

    # Map position to input sample indices
    in_indices = np.arange(num_frames, dtype=np.float64)
    out_indices = (num_frames - 1) * (phi / phi_max)

    # Interpolate samples for mono / stereo
    if num_channels == 1:
        out_samples = np.interp(out_indices, in_indices, samples)
    else:
        out_samples = np.zeros((num_frames, num_channels), dtype=np.float64)
        for c in range(num_channels):
            out_samples[:, c] = np.interp(out_indices, in_indices, samples[:, c])

    # Normalize output gain so peak matches original segment level (prevents volume boost)
    out_peak = np.max(np.abs(out_samples))
    if out_peak > 0:
        out_samples = out_samples * (orig_peak / out_peak)

    # Convert back to original integer dtype safely
    if np.issubdtype(orig_raw.dtype, np.integer):
        info = np.iinfo(orig_raw.dtype)
        out_samples = np.clip(out_samples, info.min, info.max).astype(orig_raw.dtype)
    else:
        out_samples = out_samples.astype(orig_raw.dtype)

    if num_channels > 1:
        out_samples = out_samples.flatten()

    tape_stopped = AudioSegment(
        out_samples.tobytes(),
        frame_rate=segment.frame_rate,
        sample_width=segment.sample_width,
        channels=segment.channels
    )

    # Apply micro fade-in & fade-out for smooth, pop-free transitions
    if fade_in_ms > 0 and len(tape_stopped) > fade_in_ms:
        tape_stopped = tape_stopped.fade_in(duration=fade_in_ms)
    if fade_out_ms > 0 and len(tape_stopped) > fade_out_ms:
        tape_stopped = tape_stopped.fade_out(duration=fade_out_ms)

    return tape_stopped

def censor_with_tape_stop(audio_file_path, bad_words, output_file_path="censored_output.mp3"):
    """
    Censors bad words by applying a dynamic Tape Stop / Vinyl Break pitch-drop effect.
    If separated instrumental and vocal stems exist, applies tape stop to the vocal stem
    while keeping the background instrumental playing for perfect musical rhythm.
    """
    import os
    filename = os.path.splitext(os.path.basename(audio_file_path))[0]
    instrumental_path = f'separated/{filename}/accompaniment.wav'
    vocal_path = f'separated/{filename}/vocals.wav'
    has_stems = os.path.exists(instrumental_path) and os.path.exists(vocal_path)

    audio = AudioSegment.from_file(audio_file_path)
    print(f'[+] Transcribe vocals to find bad words in Progress..')
    bad_word_timestamps = get_bad_word_timestamps(audio_file_path, bad_words)

    if has_stems:
        instrumental = AudioSegment.from_file(instrumental_path)
        vocals = AudioSegment.from_file(vocal_path)

    censored_audio = AudioSegment.empty()  # Start with an empty audio segment
    previous_end_time = 0  # Keep track of the end of the last processed segment

    # Process each bad word segment
    for start_time, end_time in bad_word_timestamps:
        # Add the audio before the bad word
        censored_audio += audio[previous_end_time:start_time]
        print(f"[-] Processing tape stop segment: {start_time} ms to {end_time} ms")
        
        if has_stems:
            inst_seg = instrumental[start_time:end_time]
            voc_seg = vocals[start_time:end_time]
            ts_vocal = apply_tape_stop_effect(voc_seg)
            censored_segment = inst_seg.overlay(ts_vocal)
        else:
            segment = audio[start_time:end_time]
            censored_segment = apply_tape_stop_effect(segment)

        censored_audio += censored_segment

        # Update the end time of the last processed segment
        previous_end_time = end_time

    # Add the remaining audio after the last bad word
    censored_audio += audio[previous_end_time:]

    # Save the censored audio to the output file
    if audio_file_path.endswith(".wav"):
        censored_audio.export(output_file_path, format="wav")
    else:
        censored_audio.export(output_file_path, format="mp3", bitrate='320k')
    print(f"Censored audio saved to {output_file_path}")


def get_bad_word_timestamps(audio_file_path, bad_words):


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

    return bad_word_timestamps

def get_bad_word_and_slurs_timestamps(audio_file_path, bad_words, slurs):

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

def cleanup():
    print(f'[=] Running clean-up..')
    os.remove('down_temp.mp3') if os.path.exists('down_temp.mp3') else None
    rmtree('separated') if os.path.exists('separated') else None