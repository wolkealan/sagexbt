import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.telegram_scraper import run_telegram_scraper
from config.config import TelegramConfig

async def main():
    await run_telegram_scraper(
        TelegramConfig.API_ID,
        TelegramConfig.API_HASH,
        TelegramConfig.PHONE_NUMBER
    )

if __name__ == "__main__":
    asyncio.run(main())