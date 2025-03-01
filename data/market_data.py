import ccxt
import pandas as pd
import numpy as np
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

from config.config import APIConfig, TradingConfig, DatabaseConfig
from utils.logger import get_market_logger
from utils.database import get_database

logger = get_market_logger()

class MarketDataProvider:
    """Provides cryptocurrency market data from various exchanges"""
    
    def __init__(self):
        self.exchanges = {}
        self.market_data = {}
        self.last_update = {}
        self.db = get_database()
        self.initialize_exchanges()
    
    def initialize_exchanges(self):
        """Initialize connections to cryptocurrency exchanges"""
        try:
            # Initialize Binance
            if APIConfig.BINANCE_API_KEY and APIConfig.BINANCE_API_SECRET:
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': APIConfig.BINANCE_API_KEY,
                    'secret': APIConfig.BINANCE_API_SECRET,
                    'enableRateLimit': True,
                })
                logger.info("Binance exchange initialized")
            
            # Initialize Coinbase (corrected method)
            if APIConfig.COINBASE_API_KEY and APIConfig.COINBASE_API_SECRET:
                self.exchanges['coinbase'] = ccxt.coinbase({  # Changed from coinbasepro to coinbase
                    'apiKey': APIConfig.COINBASE_API_KEY,
                    'secret': APIConfig.COINBASE_API_SECRET,
                    'enableRateLimit': True,
                })
                logger.info("Coinbase exchange initialized")
            
            # If no API keys are available, initialize with public methods only
            if not self.exchanges:
                self.exchanges['binance'] = ccxt.binance({'enableRateLimit': True})
                logger.warning("No API keys provided, using public methods only")
        
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
            raise
    
    async def fetch_ticker(self, symbol: str, exchange_id: str = 'binance') -> Dict[str, Any]:
        """Fetch current ticker information for a symbol"""
        try:
            # Check if we have recent data in MongoDB
            cache_key = f"{exchange_id}_{symbol}_ticker"
            cached_data = self._get_from_db(cache_key, max_age_minutes=5)
            
            if cached_data:
                logger.debug(f"Using cached ticker data for {symbol} from {exchange_id}")
                return cached_data
            
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                raise ValueError(f"Exchange {exchange_id} not initialized")
            
            # Format symbol for the exchange (e.g., BTC to BTC/USDT)
            formatted_symbol = f"{symbol}/USDT" if not '/' in symbol else symbol
            
            # Fetch ticker data
            ticker = exchange.fetch_ticker(formatted_symbol)
            
            # Update last update timestamp
            self.last_update[cache_key] = datetime.now()
            
            # Store in MongoDB
            self._save_to_db(cache_key, ticker)
            
            logger.debug(f"Fetched ticker for {formatted_symbol} from {exchange_id}")
            return ticker
        
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol} from {exchange_id}: {e}")
            return {}
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1d', 
                          limit: int = 100, exchange_id: str = 'binance') -> pd.DataFrame:
        """Fetch OHLCV (Open, High, Low, Close, Volume) data for a symbol"""
        try:
            # Check if we have recent data in MongoDB
            cache_key = f"{exchange_id}_{symbol}_{timeframe}_ohlcv"
            cached_data = self._get_from_db(cache_key, max_age_minutes=15)
            
            if cached_data:
                # Convert to DataFrame
                df = pd.DataFrame(cached_data['data'], 
                                  columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Store in memory cache
                self.market_data[cache_key] = df
                
                logger.debug(f"Using cached OHLCV data for {symbol} ({timeframe}) from {exchange_id}")
                return df
            
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                raise ValueError(f"Exchange {exchange_id} not initialized")
            
            # Format symbol for the exchange
            formatted_symbol = f"{symbol}/USDT" if not '/' in symbol else symbol
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(formatted_symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Update last update timestamp
            self.last_update[cache_key] = datetime.now()
            self.market_data[cache_key] = df
            
            # Store in MongoDB
            self._save_to_db(cache_key, {'data': ohlcv, 'timeframe': timeframe, 'symbol': symbol})
            
            logger.debug(f"Fetched OHLCV for {formatted_symbol} ({timeframe}) from {exchange_id}")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} ({timeframe}) from {exchange_id}: {e}")
            return pd.DataFrame()
    
    async def fetch_order_book(self, symbol: str, limit: int = 20, 
                              exchange_id: str = 'binance') -> Dict[str, Any]:
        """Fetch order book for a symbol"""
        try:
            # Check if we have recent data in MongoDB
            cache_key = f"{exchange_id}_{symbol}_orderbook"
            cached_data = self._get_from_db(cache_key, max_age_minutes=1)  # Order book data is very time-sensitive
            
            if cached_data:
                logger.debug(f"Using cached order book data for {symbol} from {exchange_id}")
                return cached_data
            
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                raise ValueError(f"Exchange {exchange_id} not initialized")
            
            # Format symbol for the exchange
            formatted_symbol = f"{symbol}/USDT" if not '/' in symbol else symbol
            
            # Fetch order book
            order_book = exchange.fetch_order_book(formatted_symbol, limit)
            
            # Update last update timestamp
            self.last_update[cache_key] = datetime.now()
            
            # Store in MongoDB
            self._save_to_db(cache_key, order_book)
            
            logger.debug(f"Fetched order book for {formatted_symbol} from {exchange_id}")
            return order_book
        
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol} from {exchange_id}: {e}")
            return {}
    
    def get_supported_coins(self) -> List[str]:
        """Get list of supported coins for analysis"""
        return TradingConfig.SUPPORTED_COINS
    
    async def fetch_all_coins_data(self, timeframes: List[str] = None) -> Dict[str, Any]:
        """Fetch data for all supported coins across specified timeframes"""
        if not timeframes:
            timeframes = TradingConfig.DEFAULT_TIMEFRAMES
        
        all_data = {}
        tasks = []
        
        # Create tasks for parallel data fetching
        for coin in self.get_supported_coins():
            for timeframe in timeframes:
                tasks.append(self.fetch_ohlcv(coin, timeframe))
        
        # Execute all tasks concurrently
        await asyncio.gather(*tasks)
        
        logger.info(f"Fetched data for {len(self.get_supported_coins())} coins across {len(timeframes)} timeframes")
        return self.market_data
    
    def calculate_rsi(self, symbol: str, timeframe: str = '1d', 
                 periods: int = 14, exchange_id: str = 'binance') -> float:
        """Calculate Relative Strength Index (RSI) for a symbol"""
        key = f"{exchange_id}_{symbol}_{timeframe}_ohlcv"  # Corrected key to match fetch_ohlcv
        df = self.market_data.get(key)
        
        if df is None or df.empty:
            logger.warning(f"No data available for RSI calculation for {symbol}")
            return 50  # Default neutral value
        
        try:
            # Calculate price changes
            delta = df['close'].diff()
            
            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=periods).mean()
            avg_loss = loss.rolling(window=periods).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Return the most recent RSI value
            current_rsi = rsi.iloc[-1]
            logger.debug(f"Calculated RSI for {symbol}: {current_rsi}")
            
            return current_rsi
        
        except Exception as e:
            logger.error(f"Error calculating RSI for {symbol}: {e}")
            return 50  # Default neutral value
        
    def calculate_macd(self, symbol: str, timeframe: str = '1d', 
                  fast_period: int = 12, slow_period: int = 26, 
                  signal_period: int = 9, exchange_id: str = 'binance') -> Dict[str, float]:
        """Calculate Moving Average Convergence Divergence (MACD) for a symbol"""
        key = f"{exchange_id}_{symbol}_{timeframe}_ohlcv"  # Match key format with fetch_ohlcv
        df = self.market_data.get(key)
        
        if df is None or df.empty:
            logger.warning(f"No data available for MACD calculation for {symbol}")
            return {}
        
        try:
            # Calculate MACD components
            df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
            
            # MACD Line = Fast EMA - Slow EMA
            df['macd_line'] = df['ema_fast'] - df['ema_slow']
            
            # Signal Line = 9-day EMA of MACD Line
            df['signal_line'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
            
            # MACD Histogram = MACD Line - Signal Line
            df['macd_histogram'] = df['macd_line'] - df['signal_line']
            
            # Return the most recent values
            result = {
                'macd_line': df['macd_line'].iloc[-1],
                'signal_line': df['signal_line'].iloc[-1],
                'macd_histogram': df['macd_histogram'].iloc[-1]
            }
            
            logger.debug(f"Calculated MACD for {symbol}: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error calculating MACD for {symbol}: {e}")
            return {}
    
    def calculate_moving_averages(self, symbol: str, timeframe: str = '1d', 
                                exchange_id: str = 'binance') -> Dict[str, float]:
        """Calculate various moving averages for a symbol"""
        key = f"{exchange_id}_{symbol}_{timeframe}"
        df = self.market_data.get(key)
        
        if df is None or df.empty:
            logger.warning(f"No data available for MA calculation for {symbol}")
            return {}
        
        try:
            result = {}
            # Calculate short-term MA (20 periods)
            df['ma_20'] = df['close'].rolling(window=20).mean()
            result['ma_20'] = df['ma_20'].iloc[-1]
            
            # Calculate medium-term MA (50 periods)
            df['ma_50'] = df['close'].rolling(window=50).mean()
            result['ma_50'] = df['ma_50'].iloc[-1]
            
            # Calculate long-term MA (200 periods)
            df['ma_200'] = df['close'].rolling(window=200).mean()
            result['ma_200'] = df['ma_200'].iloc[-1]
            
            # Store current price for comparison
            result['current_price'] = df['close'].iloc[-1]
            
            logger.debug(f"Calculated moving averages for {symbol}")
            return result
        
        except Exception as e:
            logger.error(f"Error calculating moving averages for {symbol}: {e}")
            return {}
    
    async def get_market_summary(self, symbol: str) -> Dict[str, Any]:
        """Get a summary of market data and indicators for a symbol"""
        try:
            # Check if we have a recent summary in MongoDB
            summary_key = f"{symbol}_market_summary"
            cached_summary = self._get_from_db(summary_key, max_age_minutes=15)
            
            if cached_summary:
                logger.info(f"Using cached market summary for {symbol}")
                return cached_summary
            
            # Get the current price
            ticker_data = await self.fetch_ticker(symbol)
            
            # Calculate indicators for different timeframes
            indicators = {}
            for timeframe in TradingConfig.DEFAULT_TIMEFRAMES:
                await self.fetch_ohlcv(symbol, timeframe)
                
                # RSI
                rsi = self.calculate_rsi(symbol, timeframe)
                
                # Moving averages
                mas = self.calculate_moving_averages(symbol, timeframe)
                
                # MACD (add this)
                macd = self.calculate_macd(symbol, timeframe)
                
                # Add indicators for this timeframe
                indicators[timeframe] = {
                    'rsi': rsi,
                    'moving_averages': mas,
                    'macd': macd  # Add the MACD data
                }
            
            # Create market summary
            summary = {
                'symbol': symbol,
                'current_price': ticker_data.get('last', 0),
                'daily_change_pct': ticker_data.get('percentage', 0),
                'volume_24h': ticker_data.get('quoteVolume', 0),
                'indicators': indicators,
                'timestamp': datetime.now()
            }
            
            # Save to MongoDB
            self._save_to_db(summary_key, summary)
            
            logger.info(f"Generated market summary for {symbol}")
            return summary
        
        except Exception as e:
            logger.error(f"Error generating market summary for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _get_from_db(self, key: str, max_age_minutes: int = 15) -> Optional[Dict[str, Any]]:
        """Get data from MongoDB if it exists and is recent enough"""
        try:
            # Find data in the collection
            query = {
                "key": key,
                "timestamp": {"$gte": datetime.now() - timedelta(minutes=max_age_minutes)}
            }
            
            cached_data = self.db.find_one(DatabaseConfig.MARKET_DATA_COLLECTION, query)
            
            if cached_data and "data" in cached_data:
                return cached_data["data"]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting data from MongoDB for {key}: {e}")
            return None
    
    def _save_to_db(self, key: str, data: Dict[str, Any]) -> bool:
        """Save data to MongoDB"""
        try:
            # Create document for MongoDB
            document = {
                "key": key,
                "data": data,
                "timestamp": datetime.now()
            }
            
            # Check if document already exists
            query = {"key": key}
            
            # Update or insert
            self.db.update_one(
                DatabaseConfig.MARKET_DATA_COLLECTION,
                query,
                {"$set": document},
                upsert=True
            )
            
            logger.debug(f"Saved data to MongoDB for {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to MongoDB for {key}: {e}")
            return False
    
    def get_historical_data(self, symbol: str, timeframe: str = '1d', 
                          days: int = 30, exchange_id: str = 'binance') -> pd.DataFrame:
        """Get historical OHLCV data for a symbol"""
        try:
            # Fetch the most recent data
            df = asyncio.run(self.fetch_ohlcv(symbol, timeframe, limit=days, exchange_id=exchange_id))
            
            if df.empty:
                logger.warning(f"No historical data available for {symbol}")
                return pd.DataFrame()
            
            # Return the DataFrame
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()

# Singleton instance
market_data_provider = MarketDataProvider()

# Helper function to get the singleton instance
def get_market_data_provider():
    return market_data_provider

# Example usage
if __name__ == "__main__":
    provider = get_market_data_provider()
    # Fetch data for Bitcoin
    asyncio.run(provider.fetch_ohlcv("BTC", "1d"))
    # Calculate RSI
    rsi = provider.calculate_rsi("BTC")
    print(f"BTC RSI: {rsi}")
    # Get market summary
    summary = provider.get_market_summary("BTC")
    print(f"Market summary: {summary}")