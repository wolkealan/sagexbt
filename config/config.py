import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent
print("Loading config.py...")
# Database configuration
# Database configuration
class DatabaseConfig:
    # MongoDB connection string
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/crypto_advisor")
    TELEGRAM_NEWS_COLLECTION = "telegram_news"
    # Database and collection names
    DB_NAME = os.getenv("DB_NAME", "walletTracker")
    
    # Collections
    MARKET_DATA_COLLECTION = "market_data"
    NEWS_COLLECTION = "news_data"
    RECOMMENDATIONS_COLLECTION = "recommendations"
    USER_PROFILES_COLLECTION = "user_profiles"
    
    # Cache TTL (Time-to-live) in seconds
    MARKET_DATA_TTL = 60 * 60  # 1 hour
    NEWS_DATA_TTL = 6 * 60 * 60  # 6 hours
    RECOMMENDATIONS_TTL = 1 * 60 * 60  # 1 hour
class TelegramConfig:
    API_ID = "20999811"  # You need to get this from my.telegram.org
    API_HASH = "2d8df81a847d782316cefe4a7f4b373a" # You need to get this from my.telegram.org  
    PHONE_NUMBER = "+919849929099"
    CHANNELS = ["sagenewsss","news_sage","cointelegraph","Cointelegraph","sage_aut"]  # Add more channels as needed
    
# API Keys and credentials
class APIConfig:
    # DeepSeek LLM API
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    
    
    
    # News API
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    
    # Crypto Exchange APIs
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
    
    COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
    COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
    
    # Social Media APIs
    TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
    TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
    TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
    TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

# Application settings
class AppConfig:
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Database settings
    VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # LLM settings
    LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    
    # Data refresh intervals (in seconds)
    MARKET_DATA_REFRESH = int(os.getenv("MARKET_DATA_REFRESH", "60"))
    NEWS_REFRESH = int(os.getenv("NEWS_REFRESH", "300"))
    SOCIAL_SENTIMENT_REFRESH = int(os.getenv("SOCIAL_SENTIMENT_REFRESH", "600"))

# Trading configuration
class TradingConfig:
    # Supported cryptocurrencies
    SUPPORTED_COINS = [
        "BTC", "ETH", "BNB", "SOL", "XRP", 
        "ADA", "DOGE", "SHIB", "DOT", "MATIC",
        "AVAX", "LINK", "UNI", "LTC", "ATOM"
    ]
    
    # Risk levels
    RISK_LEVELS = {
        "low": 0.3,      # Conservative
        "medium": 0.6,   # Balanced
        "high": 0.9      # Aggressive
    }
    
    # Default timeframes for analysis
    DEFAULT_TIMEFRAMES = ["1d", "4h", "1h"]
    
    # Technical indicators to consider
    TECHNICAL_INDICATORS = [
        "RSI", "MACD", "Bollinger", "MA", "Volume"
    ]

# Create a sample .env file template if it doesn't exist
def create_env_template():
    env_path = os.path.join(BASE_DIR, ".env.template")
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("""# MongoDB Connection
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/dbname?retryWrites=true&w=majority
DB_NAME=walletTracker

# API Keys
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_BASE=https://api.deepseek.com

# News API
NEWS_API_KEY=your_news_api_key

# Binance API
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# Coinbase API
COINBASE_API_KEY=your_coinbase_api_key
COINBASE_API_SECRET=your_coinbase_api_secret

# Twitter API
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_SECRET=your_twitter_access_secret

# App settings
DEBUG=false
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# LLM settings
LLM_MODEL=deepseek-chat
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2048

# Data refresh intervals (seconds)
MARKET_DATA_REFRESH=60
NEWS_REFRESH=300
SOCIAL_SENTIMENT_REFRESH=600
""")

if __name__ == "__main__":
    create_env_template()