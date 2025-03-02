from typing import Dict, List, Any, Optional
import time
import asyncio
from datetime import datetime, timedelta
import pymongo
import json
from config.config import TradingConfig, DatabaseConfig
from utils.logger import get_logger
from utils.database import get_database
from data.market_data import get_market_data_provider
from data.news_provider import get_news_provider
from deepseek.grok_interface import get_grok_llm
# from deepseek.llm_factory import get_llm_provider

from decision.pattern_recognition import get_pattern_recognition
logger = get_logger("recommendation_engine")

class RecommendationEngine:
    """Engine for generating trading recommendations based on market data and news"""
    
    def __init__(self):
        self.market_data = get_market_data_provider()
        self.news_provider = get_news_provider()
        self.llm = get_grok_llm()
        self.db = get_database()
        self.recommendations_cache = {}
    
    async def generate_recommendation(self, coin: str, action_type: str = "spot",
                                      risk_tolerance: Optional[str] = None,
                             force_refresh: bool = False) -> Dict[str, Any]:
        try:
            # Fetch market data
            market_summary = await self.market_data.get_market_summary(coin)
            
            # Fetch news summary (with robust error handling)
            try:
                news_summary = self.news_provider.get_coin_news_summary(coin)
            except Exception as e:
                logger.warning(f"Error fetching news summary for {coin}: {e}")
                news_summary = {
                    'coin': coin,
                    'sentiment': {'sentiment': 'neutral', 'sentiment_score': 0},
                    'recent_articles': [],
                    'total_articles_found': 0
                }
            
            # Get pattern recognition data
            try:
                pattern_recognizer = get_pattern_recognition()
                pattern_data = pattern_recognizer.identify_patterns(coin)
                logger.info(f"Pattern data retrieved for {coin}")
            except Exception as e:
                logger.warning(f"Error analyzing patterns for {coin}: {e}")
                pattern_data = {
                    "trend": {"overall": "neutral", "strength": 0},
                    "support_resistance": {},
                    "candlestick": {},
                    "chart_patterns": {}
                }
            
            # Fetch market context with more detailed logging
            try:
                market_context = self.news_provider.get_market_context()
                logger.info(f"Market Context Retrieved: {json.dumps(market_context, indent=2)}")
            except Exception as e:
                logger.warning(f"Error fetching market context: {e}")
                market_context = {
                    'market': {'sentiment': {'sentiment': 'neutral'}},
                    'geopolitical': {'sentiment': {'sentiment': 'neutral'}},
                    'regulatory': {'sentiment': {'sentiment': 'neutral'}},
                    'insights': [],
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            # Generate recommendation using LLM
            recommendation = self.llm.generate_recommendation(
                coin=coin,
                market_data=market_summary,
                news_data=news_summary,
                market_context=market_context,
                pattern_data=pattern_data,  # Add pattern data
                action_type=action_type,
                risk_tolerance=risk_tolerance
            )
            
            return recommendation
                
        except Exception as e:
            logger.error(f"Error generating recommendation for {coin}: {e}")
            return {
                'coin': coin,
                'action': 'HOLD',
                'confidence': 'Low',
                'action_type': action_type,
                'explanation': f"Unexpected error: {str(e)}",
                'timestamp': datetime.now()
            }
    
    async def generate_all_recommendations(self, action_type: str = "spot") -> Dict[str, Dict[str, Any]]:
        """Generate recommendations for all supported coins"""
        all_recommendations = {}
        tasks = []
        
        for coin in TradingConfig.SUPPORTED_COINS:
            tasks.append(self.generate_recommendation(coin, action_type))
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            coin = TradingConfig.SUPPORTED_COINS[i]
            if isinstance(result, Exception):
                logger.error(f"Error generating recommendation for {coin}: {result}")
                all_recommendations[coin] = {
                    'coin': coin,
                    'action': 'HOLD',
                    'confidence': 'Low',
                    'action_type': action_type,
                    'explanation': f"Error: {str(result)}",
                    'timestamp': datetime.now(),
                    'error': str(result)
                }
            else:
                all_recommendations[coin] = result
        
        logger.info(f"Generated recommendations for {len(all_recommendations)} coins")
        return all_recommendations
    
    def get_cached_recommendation(self, coin: str, action_type: str = "spot") -> Optional[Dict[str, Any]]:
        """Get a cached recommendation if available"""
        cache_key = f"{coin}_{action_type}"
        recommendation = self.recommendations_cache.get(cache_key)
        
        if recommendation:
            # Check if the recommendation is still relevant (less than 1 hour old)
            timestamp = recommendation.get('timestamp')
            if timestamp:
                if isinstance(timestamp, datetime):
                    age = datetime.now() - timestamp
                    if age.total_seconds() < 3600:  # 1 hour
                        return recommendation
                else:
                    if time.time() - timestamp < 3600:  # 1 hour
                        return recommendation
        
        # Try to load from database if not in memory
        return self._load_recommendation(coin, action_type)
    
    def _extract_key_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key market data points for context"""
        return {
            'price': market_data.get('current_price', 0),
            'daily_change': market_data.get('daily_change_pct', 0),
            'volume': market_data.get('volume_24h', 0),
            'rsi_1d': market_data.get('indicators', {}).get('1d', {}).get('rsi', 50)
        }
    
    def _extract_key_context(self, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key market context data with news titles"""
        context = {
            'market_sentiment': market_context.get('market', {}).get('sentiment', {}).get('sentiment', 'neutral'),
            'geo_sentiment': market_context.get('geopolitical', {}).get('sentiment', {}).get('sentiment', 'neutral'),
            'regulatory_sentiment': market_context.get('regulatory', {}).get('sentiment', {}).get('sentiment', 'neutral')
        }
        
        # Extract news titles from insights
        context['market_news_titles'] = [
            insight.get('title', '') 
            for insight in market_context.get('insights', [])
            if insight.get('title')
        ][:3]  # Limit to top 3 titles
        
        return context
    
    def _save_recommendation(self, recommendation: Dict[str, Any]) -> bool:
        """Save recommendation to MongoDB"""
        try:
            coin = recommendation.get('coin', 'unknown')
            action_type = recommendation.get('action_type', 'spot')
            
            # Check if recommendation already exists in the database
            query = {
                "coin": coin, 
                "action_type": action_type
            }
            
            # Save to MongoDB with upsert (update if exists, insert if not)
            # Use $set to update or insert
            self.db.update_one(
                DatabaseConfig.RECOMMENDATIONS_COLLECTION,
                query,
                {"$set": recommendation},
                upsert=True
            )
            
            logger.debug(f"Saved recommendation for {coin} ({action_type}) to MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"Error saving recommendation to MongoDB: {e}")
            return False
    
    def _load_recommendation(self, coin: str, action_type: str = "spot") -> Optional[Dict[str, Any]]:
        """Load the most recent recommendation from MongoDB"""
        try:
            # Find the most recent recommendation
            query = {
                "coin": coin,
                "action_type": action_type
            }
            
            recommendation = self.db.find_one(
                DatabaseConfig.RECOMMENDATIONS_COLLECTION,
                query
            )
            
            if recommendation:
                logger.debug(f"Loaded recommendation for {coin} ({action_type}) from MongoDB")
                
                # Update the cache
                cache_key = f"{coin}_{action_type}"
                self.recommendations_cache[cache_key] = recommendation
                
                return recommendation
            else:
                logger.debug(f"No recommendation found for {coin} ({action_type}) in MongoDB")
                return None
            
        except Exception as e:
            logger.error(f"Error loading recommendation for {coin} ({action_type}) from MongoDB: {e}")
            return None
    
    def get_historical_recommendations(self, coin: str, action_type: str = "spot", 
                                      days: int = 7) -> List[Dict[str, Any]]:
        """Get historical recommendations for a coin"""
        try:
            # Find recommendations from the last N days
            query = {
                "coin": coin,
                "action_type": action_type
            }
            
            recommendations = self.db.find_recent(
                DatabaseConfig.RECOMMENDATIONS_COLLECTION,
                query,
                hours=days * 24
            )
            
            logger.debug(f"Found {len(recommendations)} historical recommendations for {coin}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting historical recommendations for {coin}: {e}")
            return []

# Singleton instance
recommendation_engine = RecommendationEngine()

# Helper function to get the singleton instance
def get_recommendation_engine():
    return recommendation_engine

# Example usage
if __name__ == "__main__":
    engine = get_recommendation_engine()
    # Generate recommendation for Bitcoin
    import asyncio
    recommendation = asyncio.run(engine.generate_recommendation("BTC"))
    print(f"BTC Recommendation: {recommendation}")