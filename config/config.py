import os
from pathlib import Path
from dotenv import load_dotenv
import requests
import json
import time
from typing import List, Optional
from functools import lru_cache
import logging
logger = logging.getLogger("configs")
# Load environment variables from .env file
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent
print("Loading config.py...")

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
    CHANNELS = ["sage_news_cg"]  # Add more channels as needed
    
# API Keys and credentials
class APIConfig:
    # DeepSeek LLM API
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    
    # Grok LLM API
    GROK_API_KEY = os.getenv("GROK_API_KEY", "")
    GROK_API_BASE = os.getenv("GROK_API_BASE", "https://api.xai.com")
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
    
    X_ACCOUNT_URLS = [
        'https://x.com/Cointelegraph',  # Primary crypto news source
        'https://x.com/watcher_guru',   # Crypto market insights
        'https://x.com/TheCryptoLark',  # Crypto analyst
        'https://x.com/BinanceUS',      # Major exchange news
        'https://x.com/coindesk',       # Another crypto news source
        'https://x.com/DocumentingBTC'  # Bitcoin-focused account
    ]
    
    # Supported coins for filtering news
    SUPPORTED_COINS_CACHE = [
        'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 
        'ADA', 'DOGE', 'SHIB', 'DOT', 'MATIC', 
        'AVAX', 'UNI', 'LINK', 'LTC', 'XLM'
    ]
    
    # News collection name (match with your existing database configuration)
    NEWS_COLLECTION = 'news_data'
    
    # Fallback supported cryptocurrencies - Comprehensive list from Binance
    DEFAULT_SUPPORTED_COINS = [
        # Major Cryptocurrencies
        "BTC",  # Bitcoin
        "ETH",  # Ethereum
        "BNB",  # Binance Coin
        "SOL",  # Solana
        "XRP",  # XRP (Ripple)
        "ADA",  # Cardano
        "DOGE", # Dogecoin
        "SHIB", # Shiba Inu
        "DOT",  # Polkadot
        "MATIC", # Polygon
        "AVAX", # Avalanche
        "LINK", # Chainlink
        "UNI",  # Uniswap
        "LTC",  # Litecoin
        "ATOM", # Cosmos
        "TON",  # Toncoin
        "NEAR", # NEAR Protocol
        "ICP",  # Internet Computer
        "APT",  # Aptos
        "BCH",  # Bitcoin Cash
        
        # Layer-1 & Layer-2 Solutions
        "FTM",  # Fantom
        "ALGO", # Algorand
        "OP",   # Optimism
        "ARB",  # Arbitrum
        "STX",  # Stacks
        "HBAR", # Hedera
        "ETC",  # Ethereum Classic
        "FLOW", # Flow
        "EGLD", # MultiversX (Elrond)
        "ONE",  # Harmony
        "CELO", # Celo
        "KAVA", # Kava
        "KLAY", # Klaytn
        "ZIL",  # Zilliqa
        "KAS",  # Kaspa
        "SEI",  # Sei Network
        "SUI",  # Sui
        "TRX",  # TRON
        "IMX",  # Immutable X
        "ASTR", # Astar
        
        # DeFi Tokens
        "MKR",  # Maker
        "AAVE", # Aave
        "CRV",  # Curve
        "CAKE", # PancakeSwap
        "COMP", # Compound
        "SNX",  # Synthetix
        "1INCH", # 1inch
        "YFI",  # yearn.finance
        "SUSHI", # SushiSwap
        "CVX",  # Convex Finance
        "LDO",  # Lido DAO
        "BAL",  # Balancer
        "DYDX", # dYdX
        "QNT",  # Quant
        "GRT",  # The Graph
        "VET",  # VeChain
        "INJ",  # Injective
        
        # Stablecoins
        "USDT", # Tether
        "USDC", # USD Coin
        "BUSD", # Binance USD
        "DAI",  # Dai
        "TUSD", # TrueUSD
        "FDUSD", # First Digital USD
        
        # Gaming & Metaverse
        "SAND", # The Sandbox
        "MANA", # Decentraland
        "AXS",  # Axie Infinity
        "ENJ",  # Enjin Coin
        "GALA", # Gala Games
        "ILV",  # Illuvium
        "BLUR", # Blur
        "RNDR", # Render
        "CHZ",  # Chiliz
        "DUSK", # Dusk Network
        "GMT",  # STEPN
        "APE",  # ApeCoin
        "RUNE", # THORChain
        
        # Exchange Tokens
        "CRO",  # Crypto.com Coin
        "OKB",  # OKB
        "KCS",  # KuCoin Token
        "GT",   # GateToken
        "FTT",  # FTX Token
        "HT",   # Huobi Token
        
        # Privacy Coins
        "XMR",  # Monero
        "ZEC",  # Zcash
        "DASH", # Dash
        "ROSE", # Oasis Network
        
        # Storage & Computing
        "FIL",  # Filecoin
        "AR",   # Arweave
        
        # Newer & Trending Tokens
        "PYTH", # Pyth Network
        "JTO",  # Jito
        "BONK", # Bonk
        "BOME", # Book of Meme
        "PEPE", # Pepe
        "WIF",  # Dogwifhat
        "JUP",  # Jupiter
        "CYBER", # CyberConnect
        "TIA",  # Celestia
        "FET",  # Fetch.ai
        "ORDI", # Ordinals
        "STRK", # Starknet
        "BEAM", # Beam
        "BLAST", # Blast
        "MOUSE", # MousePad
        "AGIX", # SingularityNET
        "ID",   # Space ID
        "ACE",  # Ace
        
        # Other Significant Coins
        "AST",  # AirSwap
        "XTZ",  # Tezos
        "EOS",  # EOS
        "THETA", # Theta Network
        "NEO",  # Neo
        "IOTA", # IOTA
        "XLM",  # Stellar
        "ZRX",  # 0x
        "BAT",  # Basic Attention Token
        "RVN",  # Ravencoin
        "ICX",  # ICON
        "ONT",  # Ontology
        "WAVES", # Waves
        "DGB",  # DigiByte
        "QTUM", # Qtum
        "KSM",  # Kusama
        "DCR",  # Decred
        "ZEN",  # Horizen
        "SC",   # Siacoin
        "STG",  # Stargate Finance
        "WOO",  # WOO Network
        "CFX",  # Conflux
        "SKL",  # SKALE
        "MASK", # Mask Network
        "API3", # API3
        "OMG",  # OMG Network
        "ENS",  # Ethereum Name Service
        "MAGIC", # Magic
        "ANKR", # Ankr
        "SSV",  # SSV Network
        "BNX",  # BinaryX
        "XEM",  # NEM
        "HNT",  # Helium
        "SXP",  # Swipe
        "LINA", # Linear
        "LRC",  # Loopring
        "RPL",  # Rocket Pool
        "OGN",  # Origin Protocol
        "PEOPLE", # ConstitutionDAO
        "PAXG", # PAX Gold
        "POND", # Marlin
        "ETHW", # EthereumPoW
        "TWT",  # Trust Wallet Token
        "JASMY", # JasmyCoin
        "OCEAN", # Ocean Protocol
        "ALPHA", # Alpha Venture DAO
        "DODO", # DODO
        "IOTX", # IoTeX
        "XVG",  # Verge
        "STORJ", # Storj
        "BAKE", # BakeryToken
        "RSR",  # Reserve Rights
        "RIF",  # RSK Infrastructure Framework
        "CTK",  # CertiK
        "AUCTION", # Bounce Finance
        "SFP",  # SafePal
        "MDT",  # Measurable Data Token
        "MBOX", # MOBOX
        "BEL",  # Bella Protocol
        "WING", # Wing Finance
        "KMD",  # Komodo
        "RLC",  # iExec RLC
        "NKN",  # NKN
        "ARPA", # ARPA
    ]
    
    # Initialize Binance client at class level
    try:
        from binance.client import Client
        _binance_client = Client(
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET"),
            requests_params={'timeout': 30}  # Increase timeout
        )
        logger.info("Binance client initialized successfully")
    except ImportError:
        logger.error("Could not import Binance client. Make sure python-binance package is installed")
        _binance_client = None
    except Exception as e:
        logger.error(f"Error initializing Binance client: {e}")
        _binance_client = None
    
    # Class variable to store cached coins
    SUPPORTED_COINS_CACHE = None
    CACHE_TIMESTAMP = 0
    CACHE_TTL = 3600  # 1 hour
    
    @classmethod
    def get_supported_coins(cls, refresh=False) -> List[str]:
        """
        Dynamically fetch supported cryptocurrency coins from Binance
        
        Args:
            refresh (bool): Force refresh the coin list
            
        Returns:
            List[str]: List of coin symbols that have USDT trading pairs
        """
        # Check cache first if not refreshing
        current_time = time.time()
        if not refresh and cls.SUPPORTED_COINS_CACHE and (current_time - cls.CACHE_TIMESTAMP < cls.CACHE_TTL):
            logger.debug("Using memory-cached coin list")
            return cls.SUPPORTED_COINS_CACHE
        
        # If Binance client is unavailable, use fallback
        if cls._binance_client is None:
            logger.warning("Binance client unavailable, using default coin list")
            return cls.DEFAULT_SUPPORTED_COINS
        
        try:
            logger.info("Fetching supported coins from Binance API")
            
            # Get exchange info to get all trading pairs
            exchange_info = cls._binance_client.get_exchange_info()
            
            # Filter for USDT trading pairs and extract unique base assets
            supported_coins = set()
            for symbol_info in exchange_info.get('symbols', []):
                # Check for active trading pairs with USDT
                if (symbol_info.get('status') == 'TRADING' and 
                    symbol_info.get('quoteAsset') == 'USDT' and 
                    symbol_info.get('baseAsset') != 'USDT'):
                    supported_coins.add(symbol_info.get('baseAsset'))
            
            # Convert to sorted list for consistency
            coins_list = sorted(list(supported_coins))
            
            # Update cache
            cls.SUPPORTED_COINS_CACHE = coins_list
            cls.CACHE_TIMESTAMP = current_time
            
            # Always ensure our key coins are included
            for coin in cls.DEFAULT_SUPPORTED_COINS:
                if coin not in coins_list:
                    coins_list.append(coin)
            
            logger.info(f"Successfully fetched {len(coins_list)} coins from Binance")
            
            # Save to file cache
            cls._save_to_file_cache(coins_list)
            
            return coins_list
            
        except Exception as e:
            logger.error(f"Error fetching supported coins from Binance: {e}")
            # Try loading from file cache
            cached_coins = cls._load_from_file_cache()
            if cached_coins:
                return cached_coins
            # Ultimate fallback
            return cls.DEFAULT_SUPPORTED_COINS
    
    @classmethod
    def _save_to_file_cache(cls, coins_list: List[str]) -> None:
        """Save coins list to file cache"""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(base_dir, 'data', 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_file = os.path.join(cache_dir, 'supported_coins.json')
            with open(cache_file, 'w') as f:
                json.dump({
                    'coins': coins_list,
                    'timestamp': time.time()
                }, f)
                
            logger.debug(f"Saved {len(coins_list)} coins to cache file")
        except Exception as e:
            logger.warning(f"Could not write coins to cache file: {e}")
    
    @classmethod
    def _load_from_file_cache(cls) -> Optional[List[str]]:
        """Load coins list from file cache"""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(base_dir, 'data', 'cache')
            cache_file = os.path.join(cache_dir, 'supported_coins.json')
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check cache age
                if time.time() - cache_data.get('timestamp', 0) < cls.CACHE_TTL:
                    coins = cache_data.get('coins', [])
                    logger.info(f"Loaded {len(coins)} coins from cache file")
                    return coins
                else:
                    logger.info("Cache file expired")
        except Exception as e:
            logger.warning(f"Could not read coins from cache file: {e}")
        
        return None
    
    @property
    def SUPPORTED_COINS(self) -> List[str]:
        """
        Property to access supported coins
        This ensures compatibility with existing code
        """
        return self.get_supported_coins()

    # Risk levels
    RISK_LEVELS = {
        "low": 0.3,      # Conservative
        "medium": 0.6,   # Balanced
        "high": 0.9      # Aggressive
    }
    
    # Default timeframes for analysis
    DEFAULT_TIMEFRAMES = ["1d", "4h", "1h", "30m", "15m", "5m"]
    
    # Technical indicators to consider
    TECHNICAL_INDICATORS = [
        "RSI", "MACD", "Bollinger", "MA", "Volume"
    ]

# Initialize the cache on module load
try:
    TradingConfig.get_supported_coins()
except Exception as e:
    logger.error(f"Failed to initialize supported coins: {e}")
    
# Initialize SUPPORTED_COINS at class level - do this after the class definition
try:
    # Just initialize the cache without calling a non-existent method
    TradingConfig.SUPPORTED_COINS_CACHE = TradingConfig.get_supported_coins()
except Exception as e:
    logger.error(f"Failed to initialize supported coins cache: {e}")
    TradingConfig.SUPPORTED_COINS_CACHE = TradingConfig.DEFAULT_SUPPORTED_COINS

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
    # Print example of fetching coins
    trading_config = TradingConfig()
    print(f"Supported coins: {trading_config.SUPPORTED_COINS}")