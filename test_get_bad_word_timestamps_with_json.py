import asyncio
import sys
import os

# Add current directory to path to import async_toolset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from async_toolset import get_bad_word_timestamps

async def test_function():
    # Test the function with senseless.mp3 using bad_words.txt
    try:
        with open("bad_words.txt", "r") as f:
            bad_words = [line.strip().lower() for line in f if line.strip()]
        print(f"Loaded {len(bad_words)} bad words from bad_words.txt")
        print("Testing get_bad_word_timestamps with senseless.mp3...")
        result = await get_bad_word_timestamps("senseless.mp3", bad_words)
        print(f"Success! Found {len(result)} bad word timestamps: {result}")

        # Check if JSON was created
        json_path = "senseless.mp3.json"
        if os.path.exists(json_path):
            print(f"[+] JSON cache created at {json_path}")
        else:
            print(f"[-] JSON cache not found at {json_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_function())
