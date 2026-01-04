import google.generativeai as genai
import os
import json
import time
import argparse # For command-line arguments
from pathlib import Path

# --- Configuration ---
# It's best to set your API key as an environment variable
# or use a .env file with python-dotenv
# For simplicity here, we'll check an environment variable.
API_KEY = os.environ.get("GEMINI_API_KEY") 
MODEL_NAME = "gemini-2.5-flash"

def configure_gemini():
    """Configures the Gemini API with the API key."""
    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set it before running the script."
        )
    genai.configure(api_key=API_KEY)

def upload_audio_file(file_path: str):
    """Uploads the audio file to Gemini and waits for it to be ready."""
    print(f"Uploading '{file_path}' to Gemini...")
    audio_file = genai.upload_file(path=file_path)
    print(f"Uploaded file: {audio_file.name} (URI: {audio_file.uri})")

    # Wait for the file to be processed by Gemini
    while audio_file.state.name == "PROCESSING":
        print("Waiting for file processing...")
        time.sleep(5) # Check every 5 seconds
        # It's important to re-fetch the file object to get the updated state
        audio_file = genai.get_file(name=audio_file.name)

    if audio_file.state.name == "FAILED":
        raise ValueError(f"Audio file processing failed: {audio_file.name}")
    elif audio_file.state.name != "ACTIVE":
        raise ValueError(f"Audio file '{audio_file.name}' is not active. Current state: {audio_file.state.name}")

    print(f"File '{audio_file.name}' is active and ready for use.")
    return audio_file

def delete_uploaded_file(file_object):
    """Deletes the uploaded file from Gemini."""
    if file_object:
        try:
            print(f"Deleting uploaded file: {file_object.name}...")
            genai.delete_file(name=file_object.name)
            print(f"File '{file_object.name}' deleted successfully.")
        except Exception as e:
            print(f"Warning: Could not delete file '{file_object.name}': {e}")

def transcribe_audio_with_gemini(audio_file_object):
    """
    Sends the audio file to Gemini for transcription with timestamps.
    Returns a list of dicts with 'start', 'end', and 'text'.
    """
    model = genai.GenerativeModel(MODEL_NAME)

    prompt = f"""
    You are an expert audio transcription service.
    Transcribe the provided audio file.
    For each understandable phrase or line of lyrics, provide:
    1. 'start': The start time of the phrase in seconds (float, e.g., 0.5).
    2. 'end': The end time of the phrase in seconds (float, e.g., 3.2).
    3. 'text': The transcribed text of the phrase (string).

    Return the output STRICTLY as a JSON list of objects, where each object
    follows the structure: {{"start": S.SS, "end": E.EE, "text": "lyrics here"}}

    Example:
    [
      {{"start": 0.5, "end": 2.1, "text": "Hello, this is the first line."}},
      {{"start": 2.5, "end": 5.0, "text": "And this would be the second."}},
      {{"start": 5.2, "end": 7.8, "text": "Music playing for a bit."}}
    ]

    If a section is just music or unintelligible, you can either omit it or
    create an entry like: {{"start": X.XX, "end": Y.YY, "text": "[Music]"}} or
    {{"start": X.XX, "end": Y.YY, "text": "[Unintelligible]"}}

    Ensure the output is ONLY the JSON list and nothing else. No introductory text,
    no concluding remarks, just the JSON.
    """

    print("Generating transcription... This might take a while for longer audio.")
    # Pass the prompt and the file object to the model
    response = model.generate_content([prompt, audio_file_object])

    try:
        # Gemini might wrap the JSON in ```json ... ``` or have other text.
        # We try to extract the JSON part.
        raw_text = response.text
        print(f"\n--- Raw Gemini Response Start ---")
        print(raw_text)
        print("--- Raw Gemini Response End ---\n")

        # Attempt to find the JSON block
        json_start_markers = ["```json\n", "```JSON\n", "["]
        json_end_markers = ["\n```", "]"]
        
        extracted_json_str = None

        if raw_text.strip().startswith("[") and raw_text.strip().endswith("]"):
            extracted_json_str = raw_text.strip()
        else:
            for start_marker in json_start_markers:
                if start_marker in raw_text:
                    start_idx = raw_text.find(start_marker) + len(start_marker)
                    # Find corresponding end marker
                    # This is a bit naive, assumes only one JSON block
                    potential_json = raw_text[start_idx:]
                    for end_marker in json_end_markers:
                        if end_marker in potential_json:
                            end_idx = potential_json.rfind(end_marker)
                            extracted_json_str = potential_json[:end_idx].strip()
                            if start_marker == "[" and end_marker == "]": # if raw json
                                extracted_json_str = start_marker + extracted_json_str + end_marker
                            break
                    if extracted_json_str:
                        break
        
        if not extracted_json_str:
             print("Warning: Could not confidently extract JSON block. Attempting to parse raw response.")
             extracted_json_str = raw_text # Fallback to raw text if no markers found

        transcription_data = json.loads(extracted_json_str)

        # Basic validation
        if not isinstance(transcription_data, list):
            raise ValueError("Expected a list of transcription segments.")
        for item in transcription_data:
            if not all(k in item for k in ("start", "end", "text")):
                raise ValueError("Segment missing 'start', 'end', or 'text' key.")
            if not (isinstance(item["start"], (int, float)) and isinstance(item["end"], (int, float))):
                raise ValueError(f"Timestamps must be numbers. Got: start={item['start']}, end={item['end']}")
            if not isinstance(item["text"], str):
                raise ValueError(f"Text must be a string. Got: {item['text']}")

        return transcription_data

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from Gemini's response: {e}")
        print("Gemini's raw response might not be valid JSON or was not extracted correctly.")
        return {"error": "JSONDecodeError", "details": str(e), "raw_response": response.text}
    except (ValueError, TypeError) as e:
        print(f"Error: Invalid data structure or type in Gemini's response: {e}")
        return {"error": "DataValidationError", "details": str(e), "raw_response": response.text}
    except Exception as e:
        print(f"An unexpected error occurred while processing the response: {e}")
        # This can happen if response.text is not available (e.g. due to safety filters)
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            print(f"Prompt Feedback: {response.prompt_feedback}")
        return {"error": "UnexpectedError", "details": str(e), "raw_response": str(response)}


def main():
    parser = argparse.ArgumentParser(description="Transcribe an MP3 or WAV file using Gemini.")
    parser.add_argument("audio_file", help="Path to the MP3 or WAV audio file.")
    parser.add_argument(
        "-o", "--output",
        help="Path to save the JSON output. If not provided, prints to console."
    )
    args = parser.parse_args()

    audio_file_path = Path(args.audio_file)
    if not audio_file_path.is_file():
        print(f"Error: Audio file not found at '{audio_file_path}'")
        return

    supported_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac'] # Common audio formats Gemini supports
    if audio_file_path.suffix.lower() not in supported_extensions:
        print(f"Error: Unsupported file type '{audio_file_path.suffix}'. "
              f"Supported types: {', '.join(supported_extensions)}")
        return

    uploaded_file_object = None
    try:
        configure_gemini()
        uploaded_file_object = upload_audio_file(str(audio_file_path))
        transcription = transcribe_audio_with_gemini(uploaded_file_object)

        print("\n--- Transcription Result ---")
        if isinstance(transcription, dict) and "error" in transcription:
            print("Transcription failed.")
            print(json.dumps(transcription, indent=2))
        else:
            print(json.dumps(transcription, indent=2))
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(transcription, f, indent=2, ensure_ascii=False)
                print(f"\nTranscription saved to '{output_path}'")

    except ValueError as ve:
        print(f"Configuration or Input Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if uploaded_file_object:
            delete_uploaded_file(uploaded_file_object)

# create another function to run the main function with parameters and without argaprse so i can call it a s a module froma  different script
def transcribe_audio_file(file_path: str, output_path: str = None):
    """
    Transcribe an audio file using Gemini and save the result to a JSON file.
    
    :param file_path: Path to the audio file.
    :param output_path: Optional path to save the transcription JSON
    :return: Transcription data as a list of dicts or an error message.
    """
    print(f"GenAI method running for audio file: {file_path}..")
    configure_gemini()
    uploaded_file_object = upload_audio_file(file_path)
    transcription = transcribe_audio_with_gemini(uploaded_file_object)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)

    return transcription

if __name__ == "__main__":
    main()