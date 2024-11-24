import whisper
from pydub import AudioSegment

# Load the Whisper model
model = whisper.load_model("base")  # Use "small", "medium", or "large" for more accuracy

# Function to recognize and timestamp bad words
def get_bad_word_timestamps(audio_file_path, bad_words):
    audio = AudioSegment.from_mp3(audio_file_path)
    duration_in_seconds = len(audio) / 1000

    # Transcribe audio using Whisper
    result = model.transcribe(audio_file_path, fp16=False)
    bad_word_timestamps = []

    # Check for bad words in the transcribed segments
    for segment in result['segments']:
        start_time = int(segment['start'] * 1000)  # Convert to milliseconds
        end_time = int(segment['end'] * 1000)

        if any(bad_word in segment['text'].lower() for bad_word in bad_words):
            bad_word_timestamps.append((start_time, end_time))

    return bad_word_timestamps

# Function to censor bad words in audio with a backspin effect
def censor_audio_with_backspin(audio_file_path, bad_words, output_file_path="censored_output.mp3"):
    audio = AudioSegment.from_mp3(audio_file_path)
    bad_word_timestamps = get_bad_word_timestamps(audio_file_path, bad_words)

    # Apply backspin effect to each bad word segment
    for start_time, end_time in bad_word_timestamps:
        censored_segment = audio[start_time:end_time].reverse()  # Reverse the segment
        audio = audio[:start_time] + censored_segment + audio[end_time:]

    # Save censored audio to output file
    audio.export(output_file_path, format="mp3")
    print(f"Censored audio saved to {output_file_path}")

# Main program
if __name__ == "__main__":
    # Example usage
    audio_file_path = "song.mp3"  # Replace with your file path
    bad_words = ["shit", "damn", "bitch","drank","convict","boosy"]  # Define your bad words

    censor_audio_with_backspin(audio_file_path, bad_words)