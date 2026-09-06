import gradio as gr
import asyncio
import os
import time
import tempfile
from async_toolset import (
    process_audio_pipeline,
    get_audio_duration,
    format_time,
    cleanup,
    free_memory,
    run_in_thread
)


def compute_default_filename(audio_path, method):
    """
    Computes default output filename formatted as: <base>_censored_<method>.<ext>
    Preserves original extension (.mp3 or .wav), defaulting to .mp3 if not found.
    """
    if not audio_path:
        return f"censored_output_{method}.mp3"
    orig_name = os.path.basename(audio_path)
    base, ext = os.path.splitext(orig_name)
    if not ext:
        ext = ".mp3"
    return f"{base}_censored_{method}{ext}"


def load_words_from_file(file_path):
    """Load words from a file, one word per line."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


async def process_audio(
    audio_file,
    use_builtin_bad_words,
    bad_words_file,
    use_builtin_slurs,
    slurs_file,
    method,
    output_filename,
    ts_intensity=0.6,
    whisper_model="large-v3-turbo",
    enable_chunking=False,
    progress=None
):
    """
    Process audio file with the specified censorship method.
    """
    if whisper_model:
        os.environ["WHISPER_MODEL"] = whisper_model
    if audio_file is None:
        return None, "❌ Error: Please upload an audio file.", "0s"
    
    # Load bad words
    if use_builtin_bad_words:
        bad_words = load_words_from_file("bad_words.txt")
        if bad_words is None:
            return None, "❌ Error: Built-in bad_words.txt not found.", "0s"
    else:
        if bad_words_file is None:
            return None, "❌ Error: Please upload a bad words file or use built-in.", "0s"
        bad_words = [line.strip().lower() for line in bad_words_file.decode('utf-8').strip().split('\n') if line.strip()]
    
    if not bad_words:
        return None, "❌ Error: No valid bad words found.", "0s"
    
    # Load slurs (required for 'sv' and 'sb' methods)
    slurs = []
    if method in ['sv', 'sb']:
        if use_builtin_slurs:
            slurs = load_words_from_file("slurs.txt")
            if slurs is None:
                return None, "❌ Error: Built-in slurs.txt not found.", "0s"
        else:
            if slurs_file is None:
                return None, "❌ Error: Please upload a slurs file or use built-in.", "0s"
            slurs = [line.strip().lower() for line in slurs_file.decode('utf-8').strip().split('\n') if line.strip()]
        
        if not slurs:
            return None, "❌ Error: No valid slurs found.", "0s"
    
    # Set default output filename if not provided
    if not output_filename or not output_filename.strip():
        output_filename = compute_default_filename(audio_file, method)
    
    # Ensure output directory exists
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # Start timing
    start_time = time.time()
    
    def progress_callback(fraction, message):
        if progress:
            progress(fraction, desc=message)

    try:
        await process_audio_pipeline(
            audio_file=audio_file,
            bad_words=bad_words,
            slurs=slurs,
            method=method,
            output_path=output_path,
            ts_intensity=ts_intensity,
            whisper_model=whisper_model,
            enable_chunking=enable_chunking,
            chunk_duration_sec=300,
            progress_callback=progress_callback
        )
        
        end_time = time.time()
        processing_time = f"{end_time - start_time:.2f}s"
        
        if os.path.exists(output_path):
            chunk_str = " [5-min chunking active]" if enable_chunking else ""
            status_text = f"✅ Success! Processed with method '{method}'{chunk_str}.\n\nOutput saved to: {output_path}"
            return output_path, status_text, processing_time
        else:
            return None, "❌ Error: Output file was not created.", processing_time
    
    except Exception as e:
        end_time = time.time()
        processing_time = f"{end_time - start_time:.2f}s"
        return None, f"❌ Error during processing: {str(e)}", processing_time
    finally:
        free_memory()


def create_ui():
    """Create the Gradio UI for the audio censoring application."""
    
    # Method descriptions for the UI
    method_descriptions = {
        "v": "Vocal Separation - Replace bad words with instrumentals",
        "Gv": "GenAI Vocal Separation - Uses GenAI for transcription",
        "b": "Backspin - Reverse audio segments containing bad words",
        "ts": "Tape Stop / Vinyl Break - Dynamically pitch-down and decelerate bad word segments",
        "vb": "Vocal + Backspin - Combine vocal separation with reversed vocals",
        "p": "Down-Pitch - Lower the pitch of bad word segments",
        "sv": "Slur + Vocal - Censor bad words with instrumentals, slurs with down-pitch",
        "sb": "Slur + Both - Censor bad words with reversed vocals, slurs with down-pitch"
    }
    
    with gr.Blocks(
        title="CensorMyPy - Music Censhorship Tool",
        theme=gr.themes.Soft()
    ) as app:
        gr.Markdown(
            """
            # 🎵 CensorMyPy - Music Censhorship Tool
            
            Censor explicit content from your audio files using various methods.
            Upload an audio file, choose your preferred method and let us go!
            
            > ⚡ **Hardware Acceleration:** AMD BC-250 (RADV GFX1013) Vulkan GPU Enabled
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Audio file input
                audio_input = gr.Audio(
                    label="Audio File",
                    type="filepath",
                    sources=["upload", "microphone"],
                    interactive=True
                )

                # Duration & Chunking recommendation notice
                chunking_notice = gr.Markdown(
                    value="",
                    visible=False,
                    elem_id="chunking-notice"
                )

                # 5-minute Chunking Checkbox
                enable_chunking = gr.Checkbox(
                    label="⚡ Enable 5-Minute Audio Chunking (Recommended for sets & mixtapes > 5 min)",
                    value=False,
                    info="Splits long audio into 5-minute segments to prevent high VRAM/RAM usage, then seamlessly rejoins them.",
                    interactive=True
                )
                
                # Method selection
                method_dropdown = gr.Dropdown(
                    choices=[
                        ("Vocal Separation (v)", "v"),
                        ("GenAI Vocal Separation (Gv)", "Gv"),
                        ("Backspin (b)", "b"),
                        ("Tape Stop / Vinyl Break (ts)", "ts"),
                        ("Vocal + Backspin (vb)", "vb"),
                        ("Down-Pitch (p)", "p"),
                        ("Slur + Vocal (sv)", "sv"),
                        ("Slur + Both (sb)", "sb")
                    ],
                    label="Censorship Method",
                    value="v",
                    interactive=True
                )

                # Whisper Model Selection (Vulkan GPU)
                model_dropdown = gr.Dropdown(
                    choices=[
                        ("Large-v3-Turbo (🌟 Recommended - Highest Accuracy & Fast)", "large-v3-turbo"),
                        ("Medium (Standard)", "medium"),
                        ("Base (Fast Preview)", "base")
                    ],
                    label="Whisper AI Model (Vulkan GPU)",
                    value="large-v3-turbo",
                    interactive=True
                )

                
                # Tape Stop Intensity Slider (Visible only when 'ts' method is selected)
                ts_intensity_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.6,
                    step=0.05,
                    label="Tape Stop Break Intensity",
                    info="0.0 (0% - pure downpitch, no speed drop) to 1.0 (100% - full complete tape stop break)",
                    interactive=True,
                    visible=False
                )

                # Method description
                method_info = gr.Markdown(
                    value=f"**Selected Method:** {method_descriptions['v']}",
                    elem_id="method-info"
                )
                
                # Output filename
                output_filename = gr.Textbox(
                    label="Output Filename",
                    value="censored_output.mp3",
                    placeholder="Enter output filename (e.g., mysong_censored_v.mp3)",
                    interactive=True
                )
            
            with gr.Column(scale=1):
                # Bad words section
                use_builtin_bad_words = gr.Checkbox(
                    label="Use built-in Bad Words File",
                    value=True,
                    interactive=True
                )
                bad_words_file = gr.File(
                    label="Custom Bad Words File (one word per line)",
                    file_types=[".txt"],
                    type="binary",
                    interactive=True,
                    visible=False
                )
                
                # Slurs section
                use_builtin_slurs = gr.Checkbox(
                    label="Use built-in Slurs File (required for 'sv' and 'sb' methods)",
                    value=True,
                    interactive=True
                )
                slurs_file = gr.File(
                    label="Custom Slurs File (one word per line) - Required for 'sv' and 'sb' methods",
                    file_types=[".txt"],
                    type="binary",
                    interactive=True,
                    visible=False
                )
                
                # Toggle file upload visibility based on checkboxes
                def toggle_bad_words_file(use_builtin):
                    return gr.update(visible=not use_builtin)
                
                def toggle_slurs_file(use_builtin):
                    return gr.update(visible=not use_builtin)
                
                use_builtin_bad_words.change(
                    fn=toggle_bad_words_file,
                    inputs=[use_builtin_bad_words],
                    outputs=[bad_words_file]
                )
                
                use_builtin_slurs.change(
                    fn=toggle_slurs_file,
                    inputs=[use_builtin_slurs],
                    outputs=[slurs_file]
                )
        
        # Process button
        process_btn = gr.Button(
            "🎵 Process Audio",
            variant="primary",
            size="lg"
        )
        
        # Output section
        with gr.Row():
            with gr.Column():
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )
                
                time_output = gr.Textbox(
                    label="Processing Time",
                    interactive=False
            )
        
        # Audio output
        audio_output = gr.Audio(
            label="Censored Audio",
            type="filepath",
            interactive=False
        )
        
        # Update output filename and chunking recommendation when an audio file is uploaded/changed
        def update_on_audio(audio_path, current_method, current_filename):
            if not audio_path:
                return (
                    gr.update(value=f"censored_output_{current_method}.mp3"),
                    gr.update(value=False),
                    gr.update(value="", visible=False)
                )
            new_filename = compute_default_filename(audio_path, current_method)
            dur_sec = get_audio_duration(audio_path)
            dur_str = format_time(dur_sec)
            
            if dur_sec > 300.0:
                notice = f"> 💡 **Long Audio Detected ({dur_str} > 5 min):** 5-minute chunking has been enabled automatically below to conserve VRAM & RAM on the BC-250 and maintain peak accuracy."
                should_chunk = True
            elif dur_sec > 0:
                notice = f"> ℹ️ **Track Length:** {dur_str} (Standard single-pass processing is sufficient)."
                should_chunk = False
            else:
                notice = ""
                should_chunk = False
                
            return (
                gr.update(value=new_filename),
                gr.update(value=should_chunk),
                gr.update(value=notice, visible=bool(notice))
            )

        audio_input.change(
            fn=update_on_audio,
            inputs=[audio_input, method_dropdown, output_filename],
            outputs=[output_filename, enable_chunking, chunking_notice]
        )
        
        # Update method description, slider visibility, and default output filename when dropdown changes
        def update_method_ui(method, audio_path, current_filename):
            desc = f"**Selected Method:** {method_descriptions.get(method, 'Unknown method')}"
            show_slider = (method == "ts")
            if audio_path:
                new_filename = compute_default_filename(audio_path, method)
            else:
                new_filename = f"censored_output_{method}.mp3"
            return desc, gr.update(visible=show_slider), gr.update(value=new_filename)
        
        method_dropdown.change(
            fn=update_method_ui,
            inputs=[method_dropdown, audio_input, output_filename],
            outputs=[method_info, ts_intensity_slider, output_filename]
        )
        
        # Process audio when button is clicked
        def run_process(
            audio_file,
            use_builtin_bad_words,
            bad_words_file,
            use_builtin_slurs,
            slurs_file,
            method,
            output_name,
            ts_intensity,
            whisper_model,
            enable_chunking,
            progress=gr.Progress()
        ):
            return asyncio.run(
                process_audio(
                    audio_file=audio_file,
                    use_builtin_bad_words=use_builtin_bad_words,
                    bad_words_file=bad_words_file,
                    use_builtin_slurs=use_builtin_slurs,
                    slurs_file=slurs_file,
                    method=method,
                    output_filename=output_name,
                    ts_intensity=ts_intensity,
                    whisper_model=whisper_model,
                    enable_chunking=enable_chunking,
                    progress=progress
                )
            )
        
        process_btn.click(
            fn=run_process,
            inputs=[
                audio_input,
                use_builtin_bad_words,
                bad_words_file,
                use_builtin_slurs,
                slurs_file,
                method_dropdown,
                output_filename,
                ts_intensity_slider,
                model_dropdown,
                enable_chunking
            ],
            outputs=[audio_output, status_output, time_output]
        )
        
        # Examples section
        gr.Markdown("### 📝 Example Usage")
        gr.Markdown(
            """
            1. **Upload** an audio file (MP3, WAV, etc.)
            2. **Choose** whether to use built-in bad_words.txt or upload a custom file
            3. **Choose** whether to use built-in slurs.txt or upload a custom file (required for 'sv' and 'sb' methods)
            4. **Select a method** from the dropdown
            5. **Click Process Audio** and wait for the result
            
            **Method Guide:**
            - **v (Vocal Separation):** Separates vocals and replaces bad words with instrumentals
            - **Gv (GenAI Vocal):** Uses GenAI for transcription instead of Whisper
            - **b (Backspin):** Reverses audio segments containing bad words
            - **vb (Vocal + Backspin):** Combines instrumental replacement with reversed vocals
            - **p (Down-Pitch):** Lowers the pitch of bad word segments
            - **sv (Slur + Vocal):** Bad words → instrumentals, Slurs → down-pitched
            - **sb (Slur + Both):** Bad words → reversed vocals, Slurs → down-pitched
            """
        )
    
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CensorMyPy Gradio UI")
    parser.add_argument("--share", action="store_true", help="Enable public Gradio sharing URL")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--auth", type=str, default=None, help="Basic auth in format user:password")
    args, _ = parser.parse_known_args()

    share_enabled = args.share or os.getenv("SHARE", "").lower() in ("true", "1", "yes")
    auth_env = os.getenv("GRADIO_AUTH", args.auth)
    auth = tuple(auth_env.split(":", 1)) if auth_env and ":" in auth_env else None

    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=share_enabled,
        auth=auth
    )
