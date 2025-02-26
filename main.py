import uvicorn
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
# main.py

# main.py

from config.config import AppConfig, DatabaseConfig
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Example usage
print("AppConfig DEBUG:", AppConfig.DEBUG)
print("DatabaseConfig DB_NAME:", DatabaseConfig.DB_NAME)
# Import necessary modules
# from config.config import AppConfig, DatabaseConfig
# main.py

from utils.logger import get_logger
from utils.database import get_database
from data.market_data import get_market_data_provider
from data.news_provider import get_news_provider
from api.routes import app
logger = get_logger("main")
logger.info("Application started")

# Initialize database
db = get_database()
logger.info(f"Database initialized: {DatabaseConfig.DB_NAME}")
logger = get_logger("main")
# Add this at the beginning of your main() function in main.py
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
    
    # Log startup information
    logger.info(f"Running in {'DEBUG' if AppConfig.DEBUG else 'PRODUCTION'} mode")
    logger.info(f"Starting API server on {AppConfig.API_HOST}:{AppConfig.API_PORT}")
    
    # Start the FastAPI server
    uvicorn.run(
        "api.routes:app",
        host=AppConfig.API_HOST,
        port=AppConfig.API_PORT,
        reload=AppConfig.DEBUG,
        log_level=AppConfig.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main()