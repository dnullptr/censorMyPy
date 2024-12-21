import argparse
import time
from async_toolset import *


async def main():
    parser = argparse.ArgumentParser(description="Kudsha's Sound System Asynchronous")
    parser.add_argument("audio_file",
        default="song.mp3",
        help="Path to the audio file to be censored. Will use 'song.mp3' as default")
    parser.add_argument("bad_words_file",help="Path to the bad words file.")
    parser.add_argument("slurs_file",help="Path to the slurs file.")
    parser.add_argument(
        "--method",
        choices=["v", "b", "vb", "p", "sv"],
        required=True,
         help="Censorship method: 'v' for vocal separation, 'b' for backspin, 'vb' for combination of both, 'p' for down-pitch or 'sv' for slur + vocal.",
    )
    parser.add_argument("--output", default="censored_output.mp3", help="Output file path.")
    args = parser.parse_args()

    # Time now for execution benchmarking
    start = time.time()

    # Read bad words from file
    with open(args.bad_words_file, "r") as f:
        bad_words = [line.strip().lower() for line in f]

    if args.method == "v":
        print("Using Async vocal separation method...")
        task1 = asyncio.create_task(run_in_thread(separate_audio(args.audio_file)))
        task2 = asyncio.create_task(run_in_thread(censor_with_instrumentals(args.audio_file, bad_words, args.output)))
        await asyncio.gather(task1, task2)

    elif args.method == "b":
        print("Using Async backspin method...")
        task1 = asyncio.create_task(run_in_thread(separate_audio(args.audio_file)))
        task2 = asyncio.create_task(run_in_thread(censor_with_backspin(args.audio_file, bad_words, args.output)))
        await asyncio.gather(task1, task2)
        

    elif args.method == "vb":
        print("Using Async vocal + backspin method...")
        task1 = asyncio.create_task(run_in_thread(separate_audio(args.audio_file)))
        task2 = asyncio.create_task(run_in_thread(censor_with_both(args.audio_file, bad_words, args.output)))
        await asyncio.gather(task1, task2)
        
        
    
    elif args.method == "p":
        print("Using Async vocal downpitch method...")
        task1 = asyncio.create_task(run_in_thread(separate_audio(args.audio_file)))
        task2 = asyncio.create_task(run_in_thread(censor_with_downpitch(args.audio_file, bad_words, args.output)))
        await asyncio.gather(task1, task2)
    
    
    elif args.method == "sv":
        print("Using Async Slur + Vocal method...")
        with open(args.slurs_file, "r") as f:
            slurs = [line.strip().lower() for line in f]
        task1 = asyncio.create_task(run_in_thread(separate_audio(args.audio_file)))
        task2 = asyncio.create_task(run_in_thread(censor_with_instrumentals_and_downpitch(args.audio_file, bad_words, slurs, args.output)))
        await asyncio.gather(task1, task2)
       
    # End time
    end = time.time()
    await cleanup()
    print(f'[=] Took {end-start} seconds to run')
   

if __name__ == "__main__":
   asyncio.run(main())
   