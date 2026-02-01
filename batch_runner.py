#!/usr/bin/env python3
"""
batch_runner.py

Split an input audio into N chunks, run async_censormy.py sequentially on each chunk,
then merge the processed chunk outputs into a single output file.

Usage example:
  python batch_runner.py songxxx.mp3 bad_words.txt slurs.txt --method sb --output songxCENS.mp3 --chunks 2
"""
import argparse
import os
import sys
import subprocess
import time
from pydub import AudioSegment
import gc


def split_audio(input_path, chunks, tmp_dir):
    audio = AudioSegment.from_file(input_path)
    duration_ms = len(audio)
    chunk_duration = int(duration_ms / chunks)
    chunk_paths = []
    base = os.path.splitext(os.path.basename(input_path))[0]
    # preserve original extension (wav/mp3/other)
    ext = os.path.splitext(input_path)[1].lower().lstrip('.')
    if not ext:
        ext = 'mp3'
    for i in range(chunks):
        start = i * chunk_duration
        end = (i + 1) * chunk_duration if i < chunks - 1 else duration_ms
        chunk = audio[start:end]
        chunk_path = os.path.join(tmp_dir, f"{base}_chunk_{i}.{ext}")
        if ext == 'mp3':
            chunk.export(chunk_path, format=ext, bitrate='320k')
        else:
            chunk.export(chunk_path, format=ext)
        chunk_paths.append(chunk_path)
    return chunk_paths


def merge_audios(paths, out_path):
    if not paths:
        raise ValueError("No chunk outputs to merge")
    out = AudioSegment.empty()
    for p in paths:
        seg = AudioSegment.from_file(p)
        out += seg
    # export matching requested output extension
    out_ext = os.path.splitext(out_path)[1].lower().lstrip('.')
    if not out_ext:
        out_ext = 'mp3'
    if out_ext == 'mp3':
        out.export(out_path, format=out_ext, bitrate='320k')
    else:
        out.export(out_path, format=out_ext)


def run_chunk_processor(python_exe, runner_script, chunk_path, bad_words, slurs, method, out_chunk_path):
    cmd = [python_exe, runner_script, chunk_path, bad_words, slurs, "--method", method, "--output", out_chunk_path]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    return res.returncode


def pre_chunk_cleanup():
    """Attempt to clear TensorFlow session, empty PyTorch CUDA cache, and run garbage collection."""
    print("[pre-cleanup] Clearing frameworks and running GC before next chunk")
    try:
        import tensorflow as tf
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
    except Exception:
        pass

    try:
        import torch
        try:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass

    try:
        gc.collect()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Batch runner that chunks audio and runs async_censormy per chunk")
    parser.add_argument("input", help="Input audio file (mp3/wav)")
    parser.add_argument("bad_words", help="Bad words file")
    parser.add_argument("slurs", help="Slurs file")
    parser.add_argument("--method", required=True, help="Censor method (v,Gv,b,vb,p,sv,sb)")
    parser.add_argument("--output", required=True, help="Final output file path")
    parser.add_argument("--chunks", type=int, default=2, help="Number of chunks to split into (default 2)")
    parser.add_argument("--runner", default="async_censormy.py", help="Path to the runner script (default async_censormy.py)")
    args = parser.parse_args()

    input_path = args.input
    chunks = max(1, args.chunks)
    tmp_dir = os.path.join(".", f"_batch_tmp_{int(time.time())}")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        print(f"Splitting {input_path} into {chunks} chunks in {tmp_dir}")
        chunk_paths = split_audio(input_path, chunks, tmp_dir)

        out_chunk_paths = []
        python_exe = sys.executable
        runner_script = os.path.abspath(args.runner)

        # Use same extension for chunk outputs as the input chunks
        chunk_ext = os.path.splitext(chunk_paths[0])[1].lower().lstrip('.')
        for i, cp in enumerate(chunk_paths):
            out_chunk = os.path.join(tmp_dir, f"chunk_{i}_out.{chunk_ext}")
            pre_chunk_cleanup()
            rc = run_chunk_processor(python_exe, runner_script, cp, args.bad_words, args.slurs, args.method, out_chunk)
            if rc != 0:
                print(f"Chunk {i} processing failed (rc={rc}). Aborting.")
                sys.exit(rc)
            out_chunk_paths.append(out_chunk)

        print("Merging chunk outputs...")
        merge_audios(out_chunk_paths, args.output)
        print(f"Final merged output written to {args.output}")

    finally:
        # cleanup temp files
        try:
            for f in os.listdir(tmp_dir):
                os.remove(os.path.join(tmp_dir, f))
            os.rmdir(tmp_dir)
        except Exception:
            pass


if __name__ == '__main__':
    main()
