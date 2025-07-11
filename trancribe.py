import asyncio
from async_toolset import get_bad_word_and_slurs_timestamps,get_bad_word_timestamps

async def main():
    # Read bad words and slurs from files
    with open('bad_words.txt', 'r') as f:
        bad_words = [line.strip().lower() for line in f]
    with open('slurs.txt', 'r') as f:
        slurs = [line.strip().lower() for line in f]
    
    # Call the async function
    bad_word_timestamps = await get_bad_word_timestamps('bsx.mp3', bad_words)
    print('Bad word timestamps:', bad_word_timestamps)

if __name__ == "__main__":
    asyncio.run(main())
