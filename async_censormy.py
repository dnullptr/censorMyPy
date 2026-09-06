import os
import argparse
import time
import asyncio
from async_toolset import *


async def main():
    parser = argparse.ArgumentParser(description="Kudsha's Sound System Asynchronous")
    parser.add_argument("audio_file",
        default="song.mp3",
        help="Path to the audio file to be censored. Will use 'song.mp3' as default")
    parser.add_argument("bad_words_file", help="Path to the bad words file.")
    parser.add_argument("slurs_file", nargs="?", default="slurs.txt", help="Path to the slurs file (default: slurs.txt).")
    parser.add_argument(
        "--method",
        choices=["v", "Gv", "b", "vb", "p", "sv", "sb", "ts", "tape_stop"],
        required=True,
        help="Censorship method: 'v' for vocal separation, 'b' for backspin, 'vb' for combination of both, 'Gv' for GenAI vocal separation, 'p' for down-pitch, 'sv' for slur + vocal, 'sb' for slur + both or 'ts'/'tape_stop' for tape stop / vinyl break.",
    )
    parser.add_argument("--output", default=None, help="Output file path. Defaults to <audio_file>_censored_<method>.<ext>")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper AI model (default: large-v3-turbo)")
    parser.add_argument("--ts-intensity", type=float, default=0.6, help="Tape stop intensity 0.0-1.0 (default: 0.6)")
    parser.add_argument("--chunk", action="store_true", help="Enable 5-minute chunking for long audio sets and mixtapes (> 5 min)")
    parser.add_argument("--chunk-size", type=int, default=300, help="Chunk duration in seconds (default: 300 = 5 minutes)")
    args = parser.parse_args()

    # Time now for execution benchmarking
    start = time.time()

    # Read bad words from file
    with open(args.bad_words_file, "r") as f:
        bad_words = [line.strip().lower() for line in f if line.strip()]

    # Read slurs if needed
    slurs = []
    if args.method in ("sv", "sb"):
        if os.path.exists(args.slurs_file):
            with open(args.slurs_file, "r") as f:
                slurs = [line.strip().lower() for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"Slurs file '{args.slurs_file}' not found (required for method '{args.method}').")

    # Determine default output filename: <base>_censored_<method>.<ext>
    if not args.output:
        base, ext = os.path.splitext(os.path.basename(args.audio_file))
        if not ext:
            ext = ".mp3"
        output_file = f"{base}_censored_{args.method}{ext}"
    else:
        output_file = args.output

    # Ensure output directory exists if output path includes a directory
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Input track: {args.audio_file}")
    print(f"[*] Output track: {output_file}")
    print(f"[*] Method: {args.method} | Model: {args.model} | Chunking: {args.chunk}")

    await process_audio_pipeline(
        audio_file=args.audio_file,
        bad_words=bad_words,
        slurs=slurs,
        method=args.method,
        output_path=output_file,
        ts_intensity=args.ts_intensity,
        whisper_model=args.model,
        enable_chunking=args.chunk,
        chunk_duration_sec=args.chunk_size
    )

    end = time.time()
    print(f'[=] Finished successfully in {end-start:.2f} seconds')


if __name__ == "__main__":
    asyncio.run(main())