import sys


def main() -> None:
    print("Hello from civicsetu!")


if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())