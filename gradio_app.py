import gradio as gr
import asyncio
import os
import time
import tempfile
from async_toolset import (
    separate_audio,
    censor_with_instrumentals,
    censor_with_backspin,
    censor_with_both,
    censor_with_downpitch,
    censor_with_instrumentals_and_downpitch,
    censor_with_both_and_downpitch,
    cleanup,
    run_in_thread
)


def load_words_from_file(file_path):
    """Load words from a file, one word per line."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        return [line.strip().lower() for line in f if line.strip()]


async def process_audio(
    audio_file,
    use_builtin_bad_words,
    bad_words_file,
    use_builtin_slurs,
    slurs_file,
    method,
    output_filename
):
    """
    Process audio file with the specified censorship method.
    
    Args:
        audio_file: Path to uploaded audio file
        use_builtin_bad_words: Whether to use built-in bad_words.txt
        bad_words_file: Uploaded custom bad words file (if not using built-in)
        use_builtin_slurs: Whether to use built-in slurs.txt
        slurs_file: Uploaded custom slurs file (if not using built-in)
        method: Censorship method ('v', 'Gv', 'b', 'vb', 'p', 'sv', 'sb')
        output_filename: Output filename
    
    Returns:
        Tuple of (output_file_path, status_message, processing_time)
    """
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
    if not output_filename.strip():
        output_filename = "censored_output.mp3"
    
    # Ensure output directory exists
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # Start timing
    start_time = time.time()
    
    try:
        if method == "v":
            status = "🎵 Using Vocal Separation method..."
            task1 = asyncio.create_task(run_in_thread(separate_audio(audio_file)))
            task2 = asyncio.create_task(run_in_thread(censor_with_instrumentals(audio_file, bad_words, output_path)))
            await asyncio.gather(task1, task2)
        
        elif method == "Gv":
            status = "🎵 Using GenAI Vocal Separation method..."
            task1 = asyncio.create_task(run_in_thread(separate_audio(audio_file)))
            task2 = asyncio.create_task(run_in_thread(censor_with_instrumentals(audio_file, bad_words, output_path, genai=True)))
            await asyncio.gather(task1, task2)
        
        elif method == "b":
            status = "🎵 Using Backspin method..."
            task1 = asyncio.create_task(run_in_thread(separate_audio(audio_file)))
            task2 = asyncio.create_task(run_in_thread(censor_with_backspin(audio_file, bad_words, output_path)))
            await asyncio.gather(task1, task2)
        
        elif method == "vb":
            status = "🎵 Using Vocal + Backspin method..."
            task1 = asyncio.create_task(run_in_thread(separate_audio(audio_file)))
            task2 = asyncio.create_task(run_in_thread(censor_with_both(audio_file, bad_words, output_path, sep_task=task1)))
            await asyncio.gather(task1, task2)
        
        elif method == "p":
            status = "🎵 Using Down-Pitch method..."
            task1 = asyncio.create_task(run_in_thread(separate_audio(audio_file)))
            task2 = asyncio.create_task(run_in_thread(censor_with_downpitch(audio_file, bad_words, output_path, sep_task=task1)))
            await asyncio.gather(task1, task2)
        
        elif method == "sv":
            status = "🎵 Using Slur + Vocal method..."
            task1 = asyncio.create_task(run_in_thread(separate_audio(audio_file)))
            task2 = asyncio.create_task(run_in_thread(censor_with_instrumentals_and_downpitch(audio_file, bad_words, slurs, output_path)))
            await asyncio.gather(task1, task2)
        
        elif method == "sb":
            status = "🎵 Using Slur + Vocal + Backspin method..."
            task1 = asyncio.create_task(run_in_thread(separate_audio(audio_file)))
            task2 = asyncio.create_task(run_in_thread(censor_with_both_and_downpitch(audio_file, bad_words, slurs, output_path, sep_task=task1)))
            await asyncio.gather(task1, task2)
        
        else:
            return None, f"❌ Error: Unknown method '{method}'.", "0s"
        
        # Cleanup temporary files
        await cleanup()
        
        # Calculate processing time
        end_time = time.time()
        processing_time = f"{end_time - start_time:.2f}s"
        
        # Check if output file exists
        if os.path.exists(output_path):
            return output_path, f"✅ Success! {status}\n\nOutput saved to: {output_path}", processing_time
        else:
            return None, "❌ Error: Output file was not created.", processing_time
    
    except Exception as e:
        end_time = time.time()
        processing_time = f"{end_time - start_time:.2f}s"
        return None, f"❌ Error during processing: {str(e)}", processing_time


def create_ui():
    """Create the Gradio UI for the audio censoring application."""
    
    # Method descriptions for the UI
    method_descriptions = {
        "v": "Vocal Separation - Replace bad words with instrumentals",
        "Gv": "GenAI Vocal Separation - Uses GenAI for transcription",
        "b": "Backspin - Reverse audio segments containing bad words",
        "vb": "Vocal + Backspin - Combine vocal separation with reversed vocals",
        "p": "Down-Pitch - Lower the pitch of bad word segments",
        "sv": "Slur + Vocal - Censor bad words with instrumentals, slurs with down-pitch",
        "sb": "Slur + Both - Censor bad words with reversed vocals, slurs with down-pitch"
    }
    
    with gr.Blocks(
        title="CensorMyPy - Music Censhorship Tool"
    ) as app:
        gr.Markdown(
            """
            # 🎵 CensorMyPy - Music Censhorship Tool
            
            Censor explicit content from your audio files using various methods.
            Upload an audio file, choose your preferred method and let's f***king go!.
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
                
                # Method selection
                method_dropdown = gr.Dropdown(
                    choices=[
                        ("Vocal Separation (v)", "v"),
                        ("GenAI Vocal Separation (Gv)", "Gv"),
                        ("Backspin (b)", "b"),
                        ("Vocal + Backspin (vb)", "vb"),
                        ("Down-Pitch (p)", "p"),
                        ("Slur + Vocal (sv)", "sv"),
                        ("Slur + Both (sb)", "sb")
                    ],
                    label="Censorship Method",
                    value="v",
                    interactive=True
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
                    placeholder="Enter output filename (e.g., censored_song.mp3)",
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
        
        # Update method description when dropdown changes
        def update_method_description(method):
            return f"**Selected Method:** {method_descriptions.get(method, 'Unknown method')}"
        
        method_dropdown.change(
            fn=update_method_description,
            inputs=[method_dropdown],
            outputs=[method_info]
        )
        
        # Process audio when button is clicked
        def run_process(audio_file, use_builtin_bad_words, bad_words_file, use_builtin_slurs, slurs_file, method, output_name):
            return asyncio.run(process_audio(audio_file, use_builtin_bad_words, bad_words_file, use_builtin_slurs, slurs_file, method, output_name))
        
        process_btn.click(
            fn=run_process,
            inputs=[audio_input, use_builtin_bad_words, bad_words_file, use_builtin_slurs, slurs_file, method_dropdown, output_filename],
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
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=False,
        theme=gr.themes.Soft()
    )
