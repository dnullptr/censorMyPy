import whisper
from pydub import AudioSegment
from difflib import SequenceMatcher

# Load the Whisper model
model = whisper.load_model("medium")  # "small", "medium", "large" for better accuracy, I can use "base" but it's shitty


def get_bad_word_timestamps(audio_file_path, bad_words):
    result = model.transcribe(audio_file_path, fp16=False)
    bad_word_timestamps = []
    
    # Check for bad words in the transcribed segments
    for segment in result['segments']:
        start_time = int(segment['start'] * 1000)  # ms
        end_time = int(segment['end'] * 1000)
        if any(bad_word in segment['text'].lower() for bad_word in bad_words):
            bad_word_timestamps.append((start_time, end_time))

    return bad_word_timestamps

def align_word_with_lyrics(transcript_word, lyrics_words):
    """
    Align a single word from the transcript with the closest match in the lyrics.
    Returns the closest matching word or the original word if no close match is found.
    """
    # Calculate similarity for each word in the lyrics
    best_match = transcript_word
    max_similarity = 0

    for lyric_word in lyrics_words:
        similarity = SequenceMatcher(None, transcript_word.lower(), lyric_word.lower()).ratio()
        if similarity > max_similarity:
            best_match = lyric_word
            max_similarity = similarity

    return best_match
def align_with_lyrics(transcript_text, lyrics):
    """
    Align the model's transcript with provided lyrics to improve accuracy.
    """
    # Use SequenceMatcher to align transcript with lyrics
    matcher = SequenceMatcher(None, transcript_text.lower(), lyrics.lower())
    corrected_text = ""
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            corrected_text += transcript_text[i1:i2]  # Keep original
        elif op == "replace":
            corrected_text += lyrics[j1:j2]  # Use lyrics to fix errors
        elif op == "insert":
            corrected_text += lyrics[j1:j2]  # Add missing lyrics
        # Skip "delete" as we assume the transcript may miss content
    return corrected_text


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
def censor_audio_with_backspin(audio_file_path, bad_words, output_file_path="censored_output.mp3"):
    audio = AudioSegment.from_mp3(audio_file_path)
    bad_word_timestamps = get_bad_word_timestamps(audio_file_path, bad_words)
   

    censored_audio = AudioSegment.empty()  # Start with an empty audio segment
    previous_end_time = 0  # Keep track of the end of the last processed segment

    # Process each bad word segment
    for start_time, end_time in bad_word_timestamps:
        # Add the audio before the bad word
        censored_audio += audio[previous_end_time:start_time]
        print(f"Processing segment: {start_time} ms to {end_time} ms")
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
    

if __name__ == "__main__":
    audio_file_path = "song.mp3"  
    bad_words = ["shit","fucking","damn","bitch","drank","convict","pussy","bust"]  # Define your bad words
    print_transcribed_words(audio_file_path)
    # lyrics = open('lyrics.txt',"r+").read()
    # #print(lyrics)
    #censor_audio_with_backspin(audio_file_path, bad_words)
    