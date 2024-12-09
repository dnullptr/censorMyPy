import asyncio
import os,shutil
async def bluh():
    for i in range(100):
        print(i)

async def bla():
    for i in range(100,200):
        print(i)

async def run_in_thread(coro):
    await asyncio.to_thread(asyncio.run, coro)

async def main():
    task1 = asyncio.create_task(run_in_thread(bluh()))
    task2 = asyncio.create_task(run_in_thread(bla()))
    await asyncio.gather(task1, task2)

#asyncio.run(main())
os.remove('down_temp.mp3') if os.path.exists('down_temp.mp3') else None
shutil.rmtree('separated') if os.path.exists('separated') else None   