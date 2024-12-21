from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import os
import shutil
from toolset import (
    separate_audio,
    down_pitch,
    censor_with_instrumentals,
    censor_with_both,
    censor_with_downpitch,
    censor_with_instrumentals_and_downpitch,
    censor_with_backspin,
    get_bad_word_timestamps,
    get_bad_word_and_slurs_timestamps,
    print_transcribed_words,
    cleanup
)

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Welcome to the Kudhsa Sound System API"}
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "file_path": file_path}

@app.post("/separate_audio/")
async def separate_audio_endpoint(file_path: str = Form(...)):
    vocals_path, instrumental_path = separate_audio(file_path)
    return {"vocals_path": vocals_path, "instrumental_path": instrumental_path}

@app.post("/down_pitch/")
async def down_pitch_endpoint(file_path: str = Form(...), semitones: int = Form(...)):
    output_path = f"{file_path.split('.')[0]}_downpitched.wav"
    down_pitch(file_path, output_path, semitones)
    return {"output_path": output_path}

@app.post("/censor_with_instrumentals/")
async def censor_with_instrumentals_endpoint(file_path: str = Form(...), bad_words: str = Form(...)):
    bad_words_list = bad_words.split(",")
    output_file = f"{file_path.split('.')[0]}_censored.mp3"
    censor_with_instrumentals(file_path, bad_words_list, output_file)
    return {"output_file": output_file}

@app.post("/censor_with_both/")
async def censor_with_both_endpoint(file_path: str = Form(...), bad_words: str = Form(...)):
    bad_words_list = bad_words.split(",")
    output_file = f"{file_path.split('.')[0]}_censored.mp3"
    censor_with_both(file_path, bad_words_list, output_file)
    return {"output_file": output_file}

@app.post("/censor_with_downpitch/")
async def censor_with_downpitch_endpoint(file_path: str = Form(...), bad_words: str = Form(...)):
    bad_words_list = bad_words.split(",")
    output_file = f"{file_path.split('.')[0]}_censored.mp3"
    censor_with_downpitch(file_path, bad_words_list, output_file)
    return {"output_file": output_file}

@app.post("/censor_with_instrumentals_and_downpitch/")
async def censor_with_instrumentals_and_downpitch_endpoint(file_path: str = Form(...), bad_words: str = Form(...), slurs: str = Form(...)):
    bad_words_list = bad_words.split(",")
    slurs_list = slurs.split(",")
    output_file = f"{file_path.split('.')[0]}_censored.mp3"
    censor_with_instrumentals_and_downpitch(file_path, bad_words_list, slurs_list, output_file)
    return {"output_file": output_file}

@app.post("/censor_with_backspin/")
async def censor_with_backspin_endpoint(file_path: str = Form(...), bad_words: str = Form(...)):
    bad_words_list = bad_words.split(",")
    output_file = f"{file_path.split('.')[0]}_censored.mp3"
    censor_with_backspin(file_path, bad_words_list, output_file)
    return {"output_file": output_file}

@app.post("/print_transcribed_words/")
async def print_transcribed_words_endpoint(file_path: str = Form(...)):
    print_transcribed_words(file_path)
    return {"message": "Transcription printed to console"}

@app.post("/cleanup/")
async def cleanup_endpoint():
    cleanup()
    return {"message": "Cleanup completed"}

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
