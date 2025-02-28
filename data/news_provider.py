import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import hashlib
from typing import Dict, List, Optional, Any
import os
import json

from config.config import APIConfig, TradingConfig, BASE_DIR, DatabaseConfig
from utils.logger import get_news_logger
from utils.database import get_database

logger = get_news_logger()

class NewsDataProvider:
    """Provides news data from various sources related to cryptocurrency and financial markets"""
    
    def __init__(self):
        self.news_cache = {}
        self.last_update = {}
        self.db = get_database()
        
        # Local file-based cache directory for additional backup
        self.cache_dir = os.path.join(BASE_DIR, "data", "cache")
        self.ensure_cache_dir()
    
    def ensure_cache_dir(self):
        """Ensure the local cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory at {self.cache_dir}")
    
    def _generate_cache_key(self, query: str, days: int) -> str:
        """Generate a consistent cache key"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"newsapi_{query_hash}_{days}"
    
    def _get_from_cache(self, key: str, max_age_hours: int = 3) -> Optional[List[Dict[str, Any]]]:
        """Get data from local file cache if it exists and is recent enough"""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            # Check if cache is recent enough
            file_modified_time = os.path.getmtime(cache_file)
            file_modified_datetime = datetime.fromtimestamp(file_modified_time)
            age = datetime.now() - file_modified_datetime
            
            if age > timedelta(hours=max_age_hours):
                logger.debug(f"Local cache for {key} is too old ({age.total_seconds()/3600:.1f} hours)")
                return None
            
            # Read cache file
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            logger.debug(f"Retrieved {len(cached_data)} items from local cache for {key}")
            return cached_data
        
        except Exception as e:
            logger.error(f"Error reading from local cache {key}: {e}")
            return None
    
    def _save_to_local_cache(self, key: str, data: List[Dict[str, Any]]) -> bool:
        """Save data to local file cache"""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            logger.debug(f"Saved {len(data)} items to local cache for {key}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving to local cache {key}: {e}")
            return False
    
    def _get_from_db(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached news from MongoDB"""
        try:
            # Find data in the collection with a recent timestamp
            query = {
                "query_hash": query_hash,
                "timestamp": {"$gte": datetime.now() - timedelta(hours=3)}
            }
            
            # Retrieve from the news collection
            cached_data = self.db.find_one(
                DatabaseConfig.NEWS_COLLECTION, 
                query
            )
            
            if cached_data and "articles" in cached_data:
                return cached_data["articles"]
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving news from MongoDB: {e}")
            return None
    
    def _save_to_db(self, query_hash: str, articles: List[Dict[str, Any]]) -> bool:
        """Save news articles to MongoDB"""
        try:
            # More robust filtering of valid articles
            valid_articles = []
            for article in articles:
                # Ensure article has a valid URL and is not None
                url = article.get('url')
                if url and isinstance(url, str) and url.strip():
                    # Create a copy to avoid modifying the original
                    clean_article = article.copy()
                    
                    # Ensure all string fields are clean
                    for key in ['title', 'description', 'url']:
                        if key in clean_article:
                            clean_article[key] = str(clean_article.get(key, '')).strip()
                    
                    valid_articles.append(clean_article)
            
            # If no valid articles, return False
            if not valid_articles:
                logger.warning(f"No valid articles found for query hash {query_hash}")
                return False
            
            # Prepare document for MongoDB
            document = {
                "query_hash": query_hash,
                "articles": valid_articles,
                "timestamp": datetime.now()
            }
            
            # Upsert the document
            result = self.db.update_one(
                DatabaseConfig.NEWS_COLLECTION,
                {"query_hash": query_hash},
                {"$set": document},
                upsert=True
            )
            
            logger.debug(f"Saved {len(valid_articles)} valid news articles to MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"Error saving news to MongoDB: {e}")
            return False
    
    def fetch_news_api(self, query: str = "cryptocurrency OR bitcoin OR ethereum", 
                      days: int = 3, language: str = "en") -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI.org with multi-level caching"""
        # Generate a deterministic cache key
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_key = self._generate_cache_key(query, days)
        
        # Check in-memory cache first
        if cache_key in self.news_cache and time.time() - self.last_update.get(cache_key, 0) < 1800:  # 30 minutes
            logger.info(f"Using memory-cached news data for query: {query}")
            return self.news_cache[cache_key]
        
        # Check MongoDB cache
        cached_data = self._get_from_db(query_hash)
        if cached_data:
            logger.info(f"Using MongoDB-cached news data for query: {query}")
            # Update memory cache
            self.news_cache[cache_key] = cached_data
            self.last_update[cache_key] = time.time()
            return cached_data
        
        # Check local file cache
        local_cached_data = self._get_from_cache(cache_key)
        if local_cached_data:
            logger.info(f"Using local file-cached news data for query: {query}")
            # Update memory and MongoDB cache
            self.news_cache[cache_key] = local_cached_data
            self.last_update[cache_key] = time.time()
            self._save_to_db(query_hash, local_cached_data)
            return local_cached_data
        
        try:
            if not APIConfig.NEWS_API_KEY:
                logger.error("No NewsAPI key provided")
                return []
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Format dates for the API
            from_date_str = from_date.strftime('%Y-%m-%d')
            to_date_str = to_date.strftime('%Y-%m-%d')
            
            # Construct URL
            url = "https://newsapi.org/v2/everything"
            
            # Parameters
            params = {
                'q': query,
                'from': from_date_str,
                'to': to_date_str,
                'language': language,
                'sortBy': 'publishedAt',
                'apiKey': APIConfig.NEWS_API_KEY
            }
            
            # Make the request
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse the response
            data = response.json()
            articles = data.get('articles', [])
            
            # Cache the results in memory
            self.news_cache[cache_key] = articles
            self.last_update[cache_key] = time.time()
            
            # Save to MongoDB
            self._save_to_db(query_hash, articles)
            
            # Save to local file cache
            self._save_to_local_cache(cache_key, articles)
            
            logger.info(f"Fetched {len(articles)} news articles for query: {query}")
            return articles
        
        except Exception as e:
            logger.error(f"Error fetching news from NewsAPI: {e}")
            return []
    
    def fetch_crypto_news(self, coin: str = None) -> List[Dict[str, Any]]:
        """Fetch news specifically about a cryptocurrency or crypto in general"""
        query = "cryptocurrency OR blockchain OR crypto"
        
        # Add specific coin to query if provided
        if coin:
            if coin in TradingConfig.SUPPORTED_COINS:
                # Map coin symbols to full names for better search results
                coin_names = {
                    "BTC": "Bitcoin",
                    "ETH": "Ethereum",
                    "BNB": "Binance Coin",
                    "SOL": "Solana",
                    "XRP": "Ripple",
                    "ADA": "Cardano",
                    "DOGE": "Dogecoin",
                    "SHIB": "Shiba Inu",
                    "DOT": "Polkadot",
                    "MATIC": "Polygon",
                    "AVAX": "Avalanche",
                    "LINK": "Chainlink",
                    "UNI": "Uniswap",
                    "LTC": "Litecoin",
                    "ATOM": "Cosmos"
                }
                
                coin_name = coin_names.get(coin, coin)
                query = f"{coin_name} OR {coin} cryptocurrency"
        
        return self.fetch_news_api(query, days=2)
    
    def fetch_market_news(self) -> List[Dict[str, Any]]:
        """Fetch general financial market news"""
        query = "financial markets OR stock market OR economy OR federal reserve OR interest rates"
        return self.fetch_news_api(query, days=2)
    
    def fetch_geopolitical_news(self) -> List[Dict[str, Any]]:
        """Fetch news about geopolitical events that might impact markets"""
        query = "geopolitical OR international relations OR war OR sanctions OR trade war OR global economy"
        return self.fetch_news_api(query, days=3)
    
    def fetch_regulatory_news(self) -> List[Dict[str, Any]]:
        """Fetch news about crypto and financial regulation"""
        query = "cryptocurrency regulation OR bitcoin regulation OR SEC crypto OR crypto ban OR crypto tax"
        return self.fetch_news_api(query, days=5)  # Regulatory news has longer relevance
    
    def analyze_news_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of news articles"""
        if not articles:
            return {
                'sentiment_score': 0,
                'sentiment': 'neutral',
                'article_count': 0
            }
        
        try:
            # Very basic sentiment analysis based on titles
            # In a real implementation, you'd want to use a more sophisticated NLP approach
            positive_keywords = ['surge', 'jump', 'rise', 'gain', 'bull', 'rally', 'soar', 'up', 
                                'high', 'growth', 'positive', 'buy', 'support', 'adopt', 'approve']
            
            negative_keywords = ['drop', 'fall', 'crash', 'decline', 'bear', 'down', 'low', 'loss',
                                'negative', 'sell', 'ban', 'restrict', 'concern', 'worry', 'fear']
            
            sentiment_scores = []
            
            for article in articles:
                # Safely get title and description, defaulting to empty string
                title = str(article.get('title', '')).lower()
                description = str(article.get('description', '')).lower()
                content = title + ' ' + description
                
                # Count positive and negative keywords
                positive_count = sum(1 for word in positive_keywords if word in content)
                negative_count = sum(1 for word in negative_keywords if word in content)
                
                # Calculate sentiment score (-1 to 1)
                if positive_count == 0 and negative_count == 0:
                    score = 0  # neutral
                else:
                    score = (positive_count - negative_count) / (positive_count + negative_count)
                
                sentiment_scores.append(score)
            
            # Handle case where no valid scores were calculated
            if not sentiment_scores:
                return {
                    'sentiment_score': 0,
                    'sentiment': 'neutral',
                    'article_count': len(articles),
                    'sentiment_distribution': {
                        'positive': 0,
                        'neutral': len(articles),
                        'negative': 0
                    }
                }
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Determine sentiment category
            sentiment = 'neutral'
            if avg_sentiment > 0.2:
                sentiment = 'positive'
            elif avg_sentiment < -0.2:
                sentiment = 'negative'
            
            return {
                'sentiment_score': round(avg_sentiment, 2),
                'sentiment': sentiment,
                'article_count': len(articles),
                'sentiment_distribution': {
                    'positive': len([s for s in sentiment_scores if s > 0.2]),
                    'neutral': len([s for s in sentiment_scores if -0.2 <= s <= 0.2]),
                    'negative': len([s for s in sentiment_scores if s < -0.2])
                }
            }
        
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {
                'sentiment_score': 0,
                'sentiment': 'neutral',
                'article_count': len(articles),
                'error': str(e)
            }
   
    def fetch_telegram_news(self, coin: str = None, days: int = 3) -> List[Dict[str, Any]]:
        """
        Fetch news from both news_data and telegram_news collections
        
        Args:
            coin (str, optional): Specific coin to filter news for
            days (int, default=3): Number of days to look back
        
        Returns:
            List of news articles
        """
        try:
            # Determine cutoff date
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Prepare base query for both collections
            base_query = {
                "date": {"$gte": cutoff_date.isoformat()}
            }
            
            # Coin-specific filtering
            if coin:
                coin_filters = [
                    {"coins_mentioned": coin},
                    {"text": {"$regex": f"\\b{coin}\\b", "$options": "i"}},
                    {"title": {"$regex": f"\\b{coin}\\b", "$options": "i"}}
                ]
                base_query["$or"] = coin_filters
            
            # Query news_data collection first (prioritized)
            news_data_query = base_query.copy()
            news_data_query["source_type"] = "telegram"
            
            news_data_results = self.db.find_many(
                DatabaseConfig.NEWS_COLLECTION,
                news_data_query,
                sort=[("date", -1)],
                limit=50
            )
            
            # If not enough results, query telegram_news collection
            telegram_news_results = []
            if len(news_data_results) < 20:
                telegram_news_results = self.db.find_many(
                    DatabaseConfig.TELEGRAM_NEWS_COLLECTION,
                    base_query,
                    sort=[("date", -1)],
                    limit=50 - len(news_data_results)
                )
            
            # Combine and format results
            combined_news = []
            
            # Process news_data results
            for news in news_data_results:
                combined_news.append({
                    'title': news.get('title', ''),
                    'description': news.get('description', news.get('text', '')),
                    'publishedAt': news.get('publishedAt', news.get('date', '')),
                    'url': news.get('url', f"telegram://{news.get('channel_name', 'unknown')}"),
                    'source': {
                        'name': f"Telegram: {news.get('channel_name', 'Unknown')}"
                    },
                    'source_type': 'telegram',
                    'coins_mentioned': news.get('coins_mentioned', []),
                    'content_type': news.get('content_type', 'telegram')
                })
            
            # Process telegram_news results (if needed)
            for news in telegram_news_results:
                combined_news.append({
                    'title': news.get('title', ''),
                    'description': news.get('text', ''),
                    'publishedAt': news.get('date', ''),
                    'url': f"https://t.me/{news.get('username', 'unknown')}/{news.get('message_id', '')}",
                    'source': {
                        'name': f"Telegram: {news.get('channel_name', 'Unknown')}"
                    },
                    'source_type': 'telegram',
                    'coins_mentioned': news.get('coins_mentioned', []),
                    'content_type': 'telegram'
                })
            
            # Sort combined results by date
            combined_news.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
            
            logger.info(f"Fetched {len(combined_news)} Telegram news articles for {'specific coin' if coin else 'all coins'}")
            return combined_news
        
        except Exception as e:
            logger.error(f"Error fetching Telegram news from multiple sources: {e}")
            return []

    def get_combined_news(self, coin: str = None) -> List[Dict[str, Any]]:
        """Get news from Telegram channels (no NewsAPI)"""
        # Get Telegram news from both collections
        telegram_news = self.fetch_telegram_news(coin)
        
        # Log news sources for debugging
        logger.info(f"News sources for {coin or 'all'}:")
        logger.info(f"Telegram News count: {len(telegram_news)}")
        
        # Sort by published date (newest first)
        telegram_news.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
        
        return telegram_news
    # Override the existing method
    def get_coin_news_summary(self, coin: str) -> Dict[str, Any]:
        try:
            # Fetch combined news from different sources
            coin_news = self.get_combined_news(coin)
            
            # Analyze sentiment
            sentiment = self.analyze_news_sentiment(coin_news)
            
            # Get the most recent articles (top 5)
            recent_articles = []
            for article in coin_news[:5]:
                source_type = article.get('source_type', 'news_api')
                source_name = article.get('source', {}).get('name', 'Unknown')
                
                # Add an indicator for Telegram sources
                if source_type == 'telegram':
                    source_name = f"📱 {source_name}"
                    
                recent_article = {
                    'title': article.get('title', ''),
                    'source': {
                        'name': source_name
                    },
                    'url': article.get('url', ''),
                    'publishedAt': article.get('publishedAt', '')
                }
                recent_articles.append(recent_article)
            
            # Create news summary
            summary = {
                'coin': coin,
                'sentiment': sentiment,
                'recent_articles': recent_articles,
                'total_articles_found': len(coin_news),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Generated news summary for {coin} with {len(coin_news)} articles")
            return summary
        
        except Exception as e:
            logger.error(f"Error generating news summary for {coin}: {e}")
            return {
                'coin': coin,
                'sentiment': {
                    'sentiment': 'neutral', 
                    'sentiment_score': 0,
                    'article_count': 0
                },
                'recent_articles': [],
                'total_articles_found': 0,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def get_market_context(self) -> Dict[str, Any]:
        """Get overall market context from various news sources"""
        try:
            # Fetch different types of news
            market_news = self.fetch_market_news()
            geo_news = self.fetch_geopolitical_news()
            regulatory_news = self.fetch_regulatory_news()
            
            # Analyze sentiment for each category
            market_sentiment = self.analyze_news_sentiment(market_news)
            geo_sentiment = self.analyze_news_sentiment(geo_news)
            regulatory_sentiment = self.analyze_news_sentiment(regulatory_news)
            
            # Create market context summary
            context = {
                'market': {
                    'sentiment': market_sentiment,
                    'recent_headline': market_news[0].get('title', '') if market_news else ''
                },
                'geopolitical': {
                    'sentiment': geo_sentiment,
                    'recent_headline': geo_news[0].get('title', '') if geo_news else ''
                },
                'regulatory': {
                    'sentiment': regulatory_sentiment,
                    'recent_headline': regulatory_news[0].get('title', '') if regulatory_news else ''
                },
                'overall_sentiment': (
                    market_sentiment.get('sentiment_score', 0) * 0.5 +
                    geo_sentiment.get('sentiment_score', 0) * 0.3 +
                    regulatory_sentiment.get('sentiment_score', 0) * 0.2
                ),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info("Generated market context summary")
            return context
        
        except Exception as e:
            logger.error(f"Error generating market context: {e}")
            return {
                'error': str(e),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

# Singleton instance
news_data_provider = NewsDataProvider()

# Helper function to get the singleton instance
def get_news_provider():
    return news_data_provider

# Example usage
if __name__ == "__main__":
    provider = get_news_provider()
    # Get news for Bitcoin
    btc_news = provider.get_coin_news_summary("BTC")
    print(f"BTC News Summary: {btc_news}")
    # Get market context
    context = provider.get_market_context()
    print(f"Market Context: {context}")