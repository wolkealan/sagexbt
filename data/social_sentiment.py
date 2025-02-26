import tweepy
import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime, timedelta
import time
import re
from typing import Dict, List, Any, Optional
import statistics

from config.config import APIConfig, DatabaseConfig
from utils.logger import get_logger
from utils.database import get_database
from knowledge.entity_recognition import get_entity_recognition

logger = get_logger("social_sentiment")

class SocialSentimentAnalyzer:
    """Analyzes sentiment from social media platforms for cryptocurrencies"""
    
    def __init__(self):
        self.db = get_database()
        self.entity_recognition = get_entity_recognition()
        self.twitter_api = self._initialize_twitter_api()
        self.sentiment_cache = {}
        self.last_update = {}
    
    def _initialize_twitter_api(self) -> Optional[Any]:
        """Initialize Twitter API client"""
        try:
            if not (APIConfig.TWITTER_API_KEY and 
                   APIConfig.TWITTER_API_SECRET and 
                   APIConfig.TWITTER_ACCESS_TOKEN and 
                   APIConfig.TWITTER_ACCESS_SECRET):
                logger.warning("Twitter API credentials not fully configured")
                return None
            
            # Initialize Twitter client
            auth = tweepy.OAuth1UserHandler(
                APIConfig.TWITTER_API_KEY,
                APIConfig.TWITTER_API_SECRET,
                APIConfig.TWITTER_ACCESS_TOKEN,
                APIConfig.TWITTER_ACCESS_SECRET
            )
            api = tweepy.API(auth)
            
            # Test the connection
            api.verify_credentials()
            
            logger.info("Twitter API initialized successfully")
            return api
            
        except Exception as e:
            logger.error(f"Error initializing Twitter API: {e}")
            return None
    
    def get_sentiment(self, symbol: str, hours: int = 24, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get social sentiment analysis for a cryptocurrency
        
        Args:
            symbol: Cryptocurrency symbol
            hours: Number of hours to look back
            force_refresh: Force refresh of sentiment data
            
        Returns:
            Dictionary with sentiment analysis
        """
        try:
            # Check if we have a recent analysis in cache
            cache_key = f"{symbol}_{hours}"
            
            if not force_refresh and cache_key in self.sentiment_cache:
                cached_data = self.sentiment_cache[cache_key]
                cache_time = self.last_update.get(cache_key, 0)
                
                # Use cache if less than 30 minutes old
                if time.time() - cache_time < 1800:  # 30 minutes
                    logger.info(f"Using cached sentiment for {symbol}")
                    return cached_data
            
            # Check if we have recent data in the database
            if not force_refresh:
                db_data = self._get_from_db(symbol, hours)
                if db_data:
                    logger.info(f"Using database sentiment for {symbol}")
                    
                    # Update cache
                    self.sentiment_cache[cache_key] = db_data
                    self.last_update[cache_key] = time.time()
                    
                    return db_data
            
            # Get fresh data from social platforms
            logger.info(f"Fetching fresh sentiment for {symbol}")
            
            # Get Twitter sentiment
            twitter_sentiment = self._get_twitter_sentiment(symbol, hours)
            
            # Here you could add other platforms (Reddit, Discord, etc.)
            
            # Combine all platforms
            combined_sentiment = self._combine_sentiment_sources(twitter_sentiment)
            
            # Add metadata
            sentiment_analysis = {
                'symbol': symbol,
                'time_range': f"{hours} hours",
                'twitter_sentiment': twitter_sentiment,
                'combined_sentiment': combined_sentiment,
                'timestamp': datetime.now()
            }
            
            # Save to cache
            self.sentiment_cache[cache_key] = sentiment_analysis
            self.last_update[cache_key] = time.time()
            
            # Save to database
            self._save_to_db(sentiment_analysis)
            
            return sentiment_analysis
        
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def get_trending_coins(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get trending cryptocurrencies based on social media activity
        
        Args:
            limit: Maximum number of trending coins to return
            
        Returns:
            List of trending coin dictionaries with sentiment data
        """
        try:
            # Get trending data from database (from last 24 hours)
            from_time = datetime.now() - timedelta(hours=24)
            
            query = {
                "timestamp": {"$gte": from_time}
            }
            
            # Get all recent sentiment data
            collection = self.db.get_collection(DatabaseConfig.SOCIAL_SENTIMENT_COLLECTION)
            results = list(collection.find(query).sort("combined_sentiment.tweet_volume", -1))
            
            # Convert ObjectId to string
            for result in results:
                if '_id' in result:
                    result['_id'] = str(result['_id'])
            
            # Remove duplicates (keep most recent for each symbol)
            symbol_data = {}
            for result in results:
                symbol = result.get('symbol', '')
                if symbol and (symbol not in symbol_data or 
                               result.get('timestamp', datetime.min) > symbol_data[symbol].get('timestamp', datetime.min)):
                    symbol_data[symbol] = result
            
            # Sort by activity and sentiment
            trending = []
            for symbol, data in symbol_data.items():
                combined = data.get('combined_sentiment', {})
                tweet_volume = combined.get('tweet_volume', 0)
                sentiment_score = combined.get('sentiment_score', 0)
                
                # Calculate trending score (volume * magnitude of sentiment)
                trending_score = tweet_volume * (1 + abs(sentiment_score))
                
                trending.append({
                    'symbol': symbol,
                    'trending_score': trending_score,
                    'tweet_volume': tweet_volume,
                    'sentiment_score': sentiment_score,
                    'sentiment': combined.get('sentiment', 'neutral'),
                    'timestamp': data.get('timestamp', datetime.now())
                })
            
            # Sort by trending score
            trending.sort(key=lambda x: x.get('trending_score', 0), reverse=True)
            
            # Return top N
            return trending[:limit]
        
        except Exception as e:
            logger.error(f"Error getting trending coins: {e}")
            return []
    
    def get_sentiment_timeline(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get sentiment timeline for a cryptocurrency over time
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days to include
            
        Returns:
            Dictionary with sentiment timeline data
        """
        try:
            # Calculate date range
            from_time = datetime.now() - timedelta(days=days)
            
            # Query database for historical sentiment
            query = {
                "symbol": symbol,
                "timestamp": {"$gte": from_time}
            }
            
            collection = self.db.get_collection(DatabaseConfig.SOCIAL_SENTIMENT_COLLECTION)
            results = list(collection.find(query).sort("timestamp", 1))
            
            # Convert ObjectId to string
            for result in results:
                if '_id' in result:
                    result['_id'] = str(result['_id'])
            
            # Create timeline data
            timeline = []
            for result in results:
                combined = result.get('combined_sentiment', {})
                timestamp = result.get('timestamp', datetime.now())
                
                timeline.append({
                    'timestamp': timestamp,
                    'sentiment_score': combined.get('sentiment_score', 0),
                    'sentiment': combined.get('sentiment', 'neutral'),
                    'tweet_volume': combined.get('tweet_volume', 0),
                    'bullish_percentage': combined.get('bullish_percentage', 0),
                    'bearish_percentage': combined.get('bearish_percentage', 0)
                })
            
            # Create result
            result = {
                'symbol': symbol,
                'days': days,
                'data_points': len(timeline),
                'timeline': timeline,
                'timestamp': datetime.now()
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting sentiment timeline for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def compare_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Compare sentiment across multiple cryptocurrencies
        
        Args:
            symbols: List of cryptocurrency symbols to compare
            
        Returns:
            Dictionary with sentiment comparison
        """
        try:
            if not symbols:
                return {"error": "No symbols provided"}
            
            # Get current sentiment for each symbol
            sentiments = {}
            for symbol in symbols:
                sentiment = self.get_sentiment(symbol, hours=24)
                sentiments[symbol] = sentiment
            
            # Extract key metrics for comparison
            comparison = []
            for symbol, data in sentiments.items():
                combined = data.get('combined_sentiment', {})
                
                comparison.append({
                    'symbol': symbol,
                    'sentiment_score': combined.get('sentiment_score', 0),
                    'sentiment': combined.get('sentiment', 'neutral'),
                    'tweet_volume': combined.get('tweet_volume', 0),
                    'bullish_percentage': combined.get('bullish_percentage', 0),
                    'bearish_percentage': combined.get('bearish_percentage', 0)
                })
            
            # Sort by sentiment score (highest to lowest)
            comparison.sort(key=lambda x: x.get('sentiment_score', 0), reverse=True)
            
            # Create result
            result = {
                'symbols': symbols,
                'comparison': comparison,
                'most_bullish': comparison[0]['symbol'] if comparison else None,
                'most_bearish': comparison[-1]['symbol'] if comparison else None,
                'timestamp': datetime.now()
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error comparing sentiment for {symbols}: {e}")
            return {
                'symbols': symbols,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _get_twitter_sentiment(self, symbol: str, hours: int) -> Dict[str, Any]:
        """Get sentiment from Twitter for a cryptocurrency"""
        try:
            if not self.twitter_api:
                logger.warning("Twitter API not initialized, unable to fetch sentiment")
                return {"error": "Twitter API not available"}
            
            # Define search queries
            queries = [
                f"${symbol}",  # Stock/crypto notation
                f"#{symbol}",  # Hashtag notation
                symbol         # The symbol itself
            ]
            
            # Map common symbols to their names for better search
            symbol_names = {
                "BTC": ["Bitcoin", "#Bitcoin", "$BTC"],
                "ETH": ["Ethereum", "#Ethereum", "$ETH"],
                "SOL": ["Solana", "#Solana", "$SOL"],
                "BNB": ["Binance Coin", "#BNB", "$BNB"],
                "XRP": ["Ripple", "#XRP", "$XRP"],
                "ADA": ["Cardano", "#Cardano", "$ADA"],
                "DOGE": ["Dogecoin", "#Dogecoin", "$DOGE"],
                "DOT": ["Polkadot", "#Polkadot", "$DOT"]
            }
            
            # Add name-based queries if available
            if symbol in symbol_names:
                queries.extend(symbol_names[symbol])
            
            # Collect tweets
            all_tweets = []
            for query in queries:
                try:
                    # Search for tweets
                    tweets = self.twitter_api.search_tweets(
                        q=query,
                        lang="en",
                        count=100,
                        result_type="recent",
                        tweet_mode="extended"
                    )
                    
                    all_tweets.extend(tweets)
                    
                    # Add a delay to avoid rate limits
                    time.sleep(0.2)
                    
                except Exception as query_error:
                    logger.warning(f"Error searching for query '{query}': {query_error}")
                    continue
            
            # Remove duplicates
            unique_tweets = {}
            for tweet in all_tweets:
                if tweet.id not in unique_tweets:
                    unique_tweets[tweet.id] = tweet
            
            unique_tweet_list = list(unique_tweets.values())
            
            # Filter by time
            since_time = datetime.now() - timedelta(hours=hours)
            recent_tweets = [t for t in unique_tweet_list if t.created_at > since_time]
            
            # Analyze sentiment
            sentiment_scores = []
            bullish_tweets = 0
            bearish_tweets = 0
            neutral_tweets = 0
            
            for tweet in recent_tweets:
                # Clean tweet text
                text = self._clean_tweet_text(tweet.full_text)
                
                # Use TextBlob for sentiment analysis
                analysis = TextBlob(text)
                sentiment_score = analysis.sentiment.polarity
                
                # Categorize sentiment
                if sentiment_score > 0.2:
                    bullish_tweets += 1
                elif sentiment_score < -0.2:
                    bearish_tweets += 1
                else:
                    neutral_tweets += 1
                
                sentiment_scores.append(sentiment_score)
            
            # Calculate statistics
            tweet_count = len(recent_tweets)
            
            if tweet_count > 0:
                avg_sentiment = sum(sentiment_scores) / tweet_count
                
                # Determine overall sentiment
                sentiment = "neutral"
                if avg_sentiment > 0.1:
                    sentiment = "bullish"
                elif avg_sentiment < -0.1:
                    sentiment = "bearish"
                
                # Calculate percentages
                bullish_percentage = (bullish_tweets / tweet_count) * 100
                bearish_percentage = (bearish_tweets / tweet_count) * 100
                neutral_percentage = (neutral_tweets / tweet_count) * 100
                
                # Calculate sentiment strength
                sentiment_strength = "weak"
                if abs(avg_sentiment) > 0.3:
                    sentiment_strength = "strong"
                elif abs(avg_sentiment) > 0.15:
                    sentiment_strength = "moderate"
                
                # Calculate volatility of sentiment
                sentiment_volatility = statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0
            else:
                avg_sentiment = 0
                sentiment = "neutral"
                bullish_percentage = 0
                bearish_percentage = 0
                neutral_percentage = 100
                sentiment_strength = "weak"
                sentiment_volatility = 0
            
            # Create result
            twitter_sentiment = {
                'tweet_count': tweet_count,
                'sentiment_score': avg_sentiment,
                'sentiment': sentiment,
                'sentiment_strength': sentiment_strength,
                'sentiment_volatility': sentiment_volatility,
                'bullish_percentage': bullish_percentage,
                'bearish_percentage': bearish_percentage,
                'neutral_percentage': neutral_percentage,
                'trending_hashtags': self._extract_trending_hashtags(recent_tweets)
            }
            
            return twitter_sentiment
        
        except Exception as e:
            logger.error(f"Error getting Twitter sentiment for {symbol}: {e}")
            return {"error": str(e)}
    
    def _clean_tweet_text(self, text: str) -> str:
        """Clean tweet text for sentiment analysis"""
        try:
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            # Remove mentions (@username)
            text = re.sub(r'@\w+', '', text)
            # Remove hashtags
            text = re.sub(r'#\w+', '', text)
            # Remove cashtags ($BTC)
            text = re.sub(r'\$\w+', '', text)
            # Remove RT prefix
            text = re.sub(r'^RT ', '', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        except Exception as e:
            logger.error(f"Error cleaning tweet text: {e}")
            return text
    
    def _extract_trending_hashtags(self, tweets: List[Any]) -> List[Dict[str, Any]]:
        """Extract trending hashtags from tweets"""
        try:
            # Count hashtags
            hashtag_counts = {}
            
            for tweet in tweets:
                # Extract hashtags from tweet
                hashtags = re.findall(r'#(\w+)', tweet.full_text)
                
                for hashtag in hashtags:
                    hashtag = hashtag.lower()
                    if hashtag in hashtag_counts:
                        hashtag_counts[hashtag] += 1
                    else:
                        hashtag_counts[hashtag] = 1
            
            # Convert to list and sort
            hashtag_list = [{'hashtag': tag, 'count': count} for tag, count in hashtag_counts.items()]
            hashtag_list.sort(key=lambda x: x['count'], reverse=True)
            
            # Return top 10
            return hashtag_list[:10]
        
        except Exception as e:
            logger.error(f"Error extracting trending hashtags: {e}")
            return []
    
    def _combine_sentiment_sources(self, twitter_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Combine sentiment from multiple sources"""
        try:
            # In a more complete implementation, this would combine Twitter, Reddit, etc.
            # For now, we just use Twitter sentiment
            
            # Extract key metrics
            sentiment_score = twitter_sentiment.get('sentiment_score', 0)
            sentiment = twitter_sentiment.get('sentiment', 'neutral')
            bullish_percentage = twitter_sentiment.get('bullish_percentage', 0)
            bearish_percentage = twitter_sentiment.get('bearish_percentage', 0)
            tweet_volume = twitter_sentiment.get('tweet_count', 0)
            
            # Create combined sentiment
            combined_sentiment = {
                'sentiment_score': sentiment_score,
                'sentiment': sentiment,
                'bullish_percentage': bullish_percentage,
                'bearish_percentage': bearish_percentage,
                'tweet_volume': tweet_volume,
                'trending_hashtags': twitter_sentiment.get('trending_hashtags', []),
                'sources': ['twitter']
            }
            
            return combined_sentiment
        
        except Exception as e:
            logger.error(f"Error combining sentiment sources: {e}")
            return {
                'sentiment_score': 0,
                'sentiment': 'neutral',
                'error': str(e)
            }
    
    def _get_from_db(self, symbol: str, hours: int) -> Optional[Dict[str, Any]]:
        """Get sentiment data from database"""
        try:
            # Calculate minimum timestamp
            min_time = datetime.now() - timedelta(hours=hours, minutes=30)  # Add 30 min buffer
            
            # Query for sentiment data
            query = {
                "symbol": symbol,
                "timestamp": {"$gte": min_time}
            }
            
            # Get most recent matching record
            collection = self.db.get_collection(DatabaseConfig.SOCIAL_SENTIMENT_COLLECTION)
            result = collection.find_one(query, sort=[("timestamp", -1)])
            
            if result:
                # Convert ObjectId to string
                if '_id' in result:
                    result['_id'] = str(result['_id'])
                
                return result
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting sentiment from database for {symbol}: {e}")
            return None
    
    def _save_to_db(self, sentiment_data: Dict[str, Any]) -> bool:
        """Save sentiment data to database"""
        try:
            # Check if data already exists
            symbol = sentiment_data.get('symbol', '')
            if not symbol:
                return False
            
            # Query for existing data
            query = {
                "symbol": symbol,
                "timestamp": {
                    "$gt": datetime.now() - timedelta(hours=1)  # Only replace data older than 1 hour
                }
            }
            
            collection = self.db.get_collection(DatabaseConfig.SOCIAL_SENTIMENT_COLLECTION)
            existing = collection.find_one(query)
            
            if existing:
                # Update existing record
                collection.update_one({"_id": existing['_id']}, {"$set": sentiment_data})
            else:
                # Insert new record
                collection.insert_one(sentiment_data)
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving sentiment to database: {e}")
            return False

# Singleton instance
social_sentiment_analyzer = SocialSentimentAnalyzer()

# Helper function to get the singleton instance
def get_social_sentiment_analyzer():
    return social_sentiment_analyzer

# Example usage
if __name__ == "__main__":
    analyzer = get_social_sentiment_analyzer()
    # Get sentiment for Bitcoin
    btc_sentiment = analyzer.get_sentiment("BTC")
    print(f"BTC Sentiment: {btc_sentiment}")
    # Get trending coins
    trending = analyzer.get_trending_coins()
    print(f"Trending coins: {trending}")