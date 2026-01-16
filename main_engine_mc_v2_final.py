from __future__ import annotations
import asyncio


async def main() -> None:
    # Compatibility wrapper: keep existing entrypoint name but delegate to the
    # authoritative runner in `main.py`.
    import main as main_entry

    await main_entry.main()


if __name__ == "__main__":
    asyncio.run(main())
