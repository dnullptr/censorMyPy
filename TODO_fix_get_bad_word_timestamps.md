# TODO: Fix get_bad_word_timestamps in async_toolset.py

## Tasks
- [x] Add try-except blocks around transcription and processing in get_bad_word_timestamps
- [x] Add more logging for debugging and error handling
- [x] Ensure word-level operation is maintained (word_timestamps=True, iterate over segment.words)
- [x] Keep device="cuda", compute_type="int8_float16"
- [x] Test the function after changes
