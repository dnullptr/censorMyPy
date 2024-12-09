import argparse
import time
from toolset import *
import async_toolset as ats

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

    # Time now for execution benchmarking
    start = time.time()

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
    
    # End time
    end = time.time()
    print(f'[=] Took {end-start} seconds to run')

if __name__ == "__main__":
   main()
   