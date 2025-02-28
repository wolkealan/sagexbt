import asyncio
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.telegram_scraper import run_enhanced_telegram_scraper
from config.config import TelegramConfig

async def main():
    try:
        # Fix UTF-8 encoding for console output (for emoji support)
        import sys
        import codecs
        
        if sys.stdout.encoding != 'utf-8':
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        
        # Run the enhanced telegram scraper
        await run_enhanced_telegram_scraper(
            TelegramConfig.API_ID,
            TelegramConfig.API_HASH,
            TelegramConfig.PHONE_NUMBER
        )
    except KeyboardInterrupt:
        print("Stopping Telegram scraper due to keyboard interrupt")
    except Exception as e:
        print(f"Error running telegram scraper: {e}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Run the main function
    asyncio.run(main())