import asyncio
from async_toolset import get_bad_word_and_slurs_timestamps,get_bad_word_timestamps,print_transcribed_words

async def main():
    # Read bad words and slurs from files
    with open('bad_words.txt', 'r') as f:
        bad_words = [line.strip().lower() for line in f]
    with open('slurs.txt', 'r') as f:
        slurs = [line.strip().lower() for line in f]
    
    # Call the async function
    SONG_NAME = 'good_comf.mp3'
    #bad_word_timestamps = await get_bad_word_timestamps(SONG_NAME, bad_words)
    #print('Bad word timestamps:', bad_word_timestamps)
    await print_transcribed_words(SONG_NAME)

if __name__ == "__main__":
    asyncio.run(main())
