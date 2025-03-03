import uvicorn
import asyncio
import threading
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import platform

# For Windows compatibility
if platform.system() == 'Windows':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()  # Load environment variables from .env file

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import AppConfig, DatabaseConfig
from discord.discord_bot import run_discord_bot
from utils.logger import get_logger
from utils.database import get_database
from data.market_data import get_market_data_provider
from data.news_provider import get_news_provider
from api.routes import app as api_app

logger = get_logger("main")

# Initialize database
db = get_database()
logger.info(f"Database initialized: {DatabaseConfig.DB_NAME}")

print("Starting Crypto Trading Advisor application...")

async def initialize_database():
    """Initialize database connection"""
    logger.info("Initializing database connection...")
    try:
        # Get database instance to trigger initialization
        db = get_database()
        logger.info(f"Connected to MongoDB: {DatabaseConfig.DB_NAME}")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        sys.exit(1)  # Exit if database connection fails

async def initialize_data():
    """Initialize data providers by prefetching some data"""
    logger.info("Initializing data providers...")
    
    try:
        # Get market data provider
        market_data = get_market_data_provider()
        
        # Prefetch data for major coins
        major_coins = ["BTC", "ETH"]
        for coin in major_coins:
            for timeframe in ["1d", "4h"]:
                await market_data.fetch_ohlcv(coin, timeframe)
            logger.info(f"Prefetched market data for {coin}")
        
        # Get news provider
        news_provider = get_news_provider()
        
        # Prefetch general market context
        _ = news_provider.get_market_context()
        logger.info("Prefetched market context")
        
        # Prefetch news for major coins
        for coin in major_coins:
            _ = news_provider.get_coin_news_summary(coin)
            logger.info(f"Prefetched news for {coin}")
        
        logger.info("Data initialization complete")
        
    except Exception as e:
        logger.error(f"Error initializing data: {e}")
        # Continue anyway, as we can fetch data on-demand later

def create_app_dirs():
    """Create necessary application directories"""
    dirs = [
        "logs",
        "data/cache",
        "data/vector_db"
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for directory in dirs:
        path = os.path.join(base_dir, directory)
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory: {path}")

def run_combined_api():
    """Run a single FastAPI instance that serves both the main API and Discord bot API"""
    api_port = int(os.getenv("API_PORT", "8000"))
    logger.info(f"Starting combined API server on {AppConfig.API_HOST}:{api_port}")
    uvicorn.run(
        "api.routes:app", 
        host=AppConfig.API_HOST, 
        port=api_port, 
        reload=AppConfig.DEBUG, 
        log_level=AppConfig.LOG_LEVEL.lower()
    )

def run_discord():
    """Run the Discord bot"""
    run_discord_bot()

def start_background_tasks():
    """Start background tasks like data refreshing"""
    # This would typically use something like APScheduler or asyncio.create_task
    # For simplicity, we're not implementing this in the initial version
    pass

def main():
    """Main entry point of the application"""
    logger.info("Starting Crypto Trading Advisor application")
    
    # Create necessary directories
    create_app_dirs()
    
    # Initialize database
    asyncio.run(initialize_database())
    
    # Initialize data asynchronously
    asyncio.run(initialize_data())
    
    # Start background tasks
    start_background_tasks()
    
    # Run Discord bot in a separate thread
    discord_thread = threading.Thread(target=run_discord)
    discord_thread.daemon = True
    discord_thread.start()
    
    # Log startup information
    logger.info(f"Running in {'DEBUG' if AppConfig.DEBUG else 'PRODUCTION'} mode")
    
    # Run the main API in the main thread
    # This will handle both frontend and Discord bot requests
    run_combined_api()

if __name__ == "__main__":
    main()