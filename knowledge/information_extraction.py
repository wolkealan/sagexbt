import re
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import json

from config.config import TradingConfig, DatabaseConfig
from utils.logger import get_logger
from utils.database import get_database
from knowledge.entity_recognition import get_entity_recognition

logger = get_logger("information_extraction")

class InformationExtraction:
    """Extracts structured information from unstructured text sources"""
    
    def __init__(self):
        self.entity_recognition = get_entity_recognition()
        self.db = get_database()
    
    def extract_from_news(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured information from a news article
        
        Args:
            article: News article dictionary
            
        Returns:
            Extracted information dictionary
        """
        try:
            # Get article text components
            title = article.get("title", "")
            description = article.get("description", "")
            content = article.get("content", "")
            
            # Combine text for analysis
            full_text = f"{title}. {description}. {content}"
            
            # Extract entities
            entities = self.entity_recognition.detect_entities(full_text)
            
            # Extract sentiment
            sentiment = self.entity_recognition.extract_sentiment(full_text)
            
            # Extract price mentions
            price_mentions = self.entity_recognition.extract_price_mentions(full_text)
            
            # Extract key events
            key_events = self._extract_key_events(full_text)
            
            # Create structured information
            info = {
                "article_id": article.get("url", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "published_at": article.get("publishedAt", ""),
                "title": title,
                "entities": entities,
                "sentiment": sentiment,
                "price_mentions": price_mentions,
                "key_events": key_events,
                "cryptocurrencies": entities.get("cryptocurrencies", []),
                "timestamp": datetime.now()
            }
            
            # Store in database
            self._store_extracted_info(info, "news")
            
            logger.debug(f"Extracted information from article: {title}")
            return info
        
        except Exception as e:
            logger.error(f"Error extracting information from news: {e}")
            return {"error": str(e)}
    
    def extract_from_social_media(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured information from a social media post
        
        Args:
            post: Social media post dictionary
            
        Returns:
            Extracted information dictionary
        """
        try:
            # Get post text
            text = post.get("text", "")
            if not text:
                return {"error": "No text content"}
            
            # Extract entities
            entities = self.entity_recognition.detect_entities(text)
            
            # Extract sentiment
            sentiment = self.entity_recognition.extract_sentiment(text)
            
            # Extract price mentions
            price_mentions = self.entity_recognition.extract_price_mentions(text)
            
            # Create structured information
            info = {
                "post_id": post.get("id", ""),
                "platform": post.get("platform", "Unknown"),
                "author": post.get("author", "Unknown"),
                "created_at": post.get("created_at", ""),
                "text": text,
                "entities": entities,
                "sentiment": sentiment,
                "price_mentions": price_mentions,
                "cryptocurrencies": entities.get("cryptocurrencies", []),
                "timestamp": datetime.now()
            }
            
            # Store in database
            self._store_extracted_info(info, "social")
            
            logger.debug(f"Extracted information from social media post")
            return info
        
        except Exception as e:
            logger.error(f"Error extracting information from social media: {e}")
            return {"error": str(e)}
    
    def extract_from_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key insights from market data
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Extracted insights dictionary
        """
        try:
            # Extract basic information
            symbol = market_data.get("symbol", "Unknown")
            current_price = market_data.get("current_price", 0)
            daily_change = market_data.get("daily_change_pct", 0)
            
            # Extract indicators
            indicators = market_data.get("indicators", {})
            
            # Generate insights
            insights = self._generate_market_insights(symbol, current_price, daily_change, indicators)
            
            # Create structured information
            info = {
                "symbol": symbol,
                "price": current_price,
                "daily_change": daily_change,
                "indicators": indicators,
                "insights": insights,
                "timestamp": datetime.now()
            }
            
            # Store in database
            self._store_extracted_info(info, "market")
            
            logger.debug(f"Extracted insights from market data for {symbol}")
            return info
        
        except Exception as e:
            logger.error(f"Error extracting information from market data: {e}")
            return {"error": str(e)}
    
    def _extract_key_events(self, text: str) -> List[Dict[str, Any]]:
        """Extract key events from text"""
        try:
            events = []
            
            # Event keywords to look for
            event_keywords = {
                "launch": ["launch", "release", "introduce", "unveil", "debut"],
                "partnership": ["partner", "collaborate", "alliance", "team up", "join forces"],
                "acquisition": ["acquire", "buy", "purchase", "takeover", "merge"],
                "regulation": ["regulate", "ban", "approve", "compliance", "legal", "illegal", "law"],
                "development": ["develop", "upgrade", "update", "improve", "enhance"],
                "adoption": ["adopt", "use", "integrate", "implement", "accept"],
                "hack": ["hack", "breach", "vulnerability", "attack", "compromise", "exploit"]
            }
            
            # Check for event keywords in the text
            for event_type, keywords in event_keywords.items():
                for keyword in keywords:
                    if keyword in text.lower():
                        # Find the sentence containing the keyword
                        sentences = re.split(r'[.!?]', text)
                        for sentence in sentences:
                            if keyword in sentence.lower():
                                # Extract cryptocurrencies in this sentence
                                coins = self.entity_recognition.extract_coin_mentions(sentence)
                                
                                event = {
                                    "type": event_type,
                                    "keyword": keyword,
                                    "description": sentence.strip(),
                                    "cryptocurrencies": coins,
                                    "confidence": "medium"
                                }
                                
                                # Avoid duplicates
                                if not any(e["description"] == sentence.strip() for e in events):
                                    events.append(event)
            
            return events
        
        except Exception as e:
            logger.error(f"Error extracting key events: {e}")
            return []
    
    def _generate_market_insights(self, symbol: str, price: float, 
                               daily_change: float, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from market data"""
        try:
            insights = []
            
            # Price movement insight
            if daily_change > 5:
                insights.append({
                    "type": "price_movement",
                    "insight": f"Strong bullish movement with {daily_change:.2f}% gain",
                    "significance": "high"
                })
            elif daily_change > 2:
                insights.append({
                    "type": "price_movement",
                    "insight": f"Bullish movement with {daily_change:.2f}% gain",
                    "significance": "medium"
                })
            elif daily_change < -5:
                insights.append({
                    "type": "price_movement",
                    "insight": f"Strong bearish movement with {daily_change:.2f}% loss",
                    "significance": "high"
                })
            elif daily_change < -2:
                insights.append({
                    "type": "price_movement",
                    "insight": f"Bearish movement with {daily_change:.2f}% loss",
                    "significance": "medium"
                })
            
            # RSI insights
            for timeframe, data in indicators.items():
                if "rsi" in data:
                    rsi = data["rsi"]
                    if rsi > 70:
                        insights.append({
                            "type": "technical_indicator",
                            "indicator": "RSI",
                            "timeframe": timeframe,
                            "insight": f"Overbought conditions with RSI at {rsi:.2f}",
                            "significance": "high" if rsi > 80 else "medium"
                        })
                    elif rsi < 30:
                        insights.append({
                            "type": "technical_indicator",
                            "indicator": "RSI",
                            "timeframe": timeframe,
                            "insight": f"Oversold conditions with RSI at {rsi:.2f}",
                            "significance": "high" if rsi < 20 else "medium"
                        })
                
                # Moving average insights
                if "moving_averages" in data:
                    mas = data["moving_averages"]
                    current_price = mas.get("current_price", price)
                    
                    # Check for golden cross (MA 50 crosses above MA 200)
                    if "ma_50" in mas and "ma_200" in mas:
                        ma_50 = mas["ma_50"]
                        ma_200 = mas["ma_200"]
                        
                        if ma_50 > ma_200 and ma_50 / ma_200 < 1.01:  # Within 1% of crossing
                            insights.append({
                                "type": "technical_indicator",
                                "indicator": "Moving Average",
                                "timeframe": timeframe,
                                "insight": "Potential golden cross forming (MA 50 crossing above MA 200)",
                                "significance": "high"
                            })
                        elif ma_50 < ma_200 and ma_50 / ma_200 > 0.99:  # Within 1% of crossing
                            insights.append({
                                "type": "technical_indicator",
                                "indicator": "Moving Average",
                                "timeframe": timeframe,
                                "insight": "Potential death cross forming (MA 50 crossing below MA 200)",
                                "significance": "high"
                            })
                    
                    # Price relative to MA
                    if "ma_20" in mas:
                        ma_20 = mas["ma_20"]
                        if current_price > ma_20 * 1.1:
                            insights.append({
                                "type": "technical_indicator",
                                "indicator": "Moving Average",
                                "timeframe": timeframe,
                                "insight": f"Price significantly above MA 20 (${ma_20:.2f})",
                                "significance": "medium"
                            })
                        elif current_price < ma_20 * 0.9:
                            insights.append({
                                "type": "technical_indicator",
                                "indicator": "Moving Average",
                                "timeframe": timeframe,
                                "insight": f"Price significantly below MA 20 (${ma_20:.2f})",
                                "significance": "medium"
                            })
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating market insights: {e}")
            return []
    
    def _store_extracted_info(self, info: Dict[str, Any], source_type: str) -> bool:
        """Store extracted information in the database"""
        try:
            # Create document for MongoDB
            document = {
                "source_type": source_type,
                "info": info,
                "processed_at": datetime.now()
            }
            
            # Add source-specific fields
            if source_type == "news":
                document["article_id"] = info.get("article_id", "")
                document["title"] = info.get("title", "")
            elif source_type == "social":
                document["post_id"] = info.get("post_id", "")
                document["platform"] = info.get("platform", "")
            elif source_type == "market":
                document["symbol"] = info.get("symbol", "")
            
            # Add cryptocurrency tags for easier querying
            document["cryptocurrencies"] = info.get("cryptocurrencies", [])
            
            # Store in MongoDB
            self.db.insert_one("extracted_information", document)
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing extracted information: {e}")
            return False
    
    def search_extracted_info(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for extracted information
        
        Args:
            query: MongoDB query
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        try:
            # Get results from MongoDB
            results = self.db.find_many("extracted_information", query, 
                                       sort=[("processed_at", -1)], limit=limit)
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching extracted information: {e}")
            return []
    
    def get_recent_insights(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get recent insights for a cryptocurrency
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days to look back
            
        Returns:
            Dictionary with combined insights from news and market data
        """
        try:
            # Calculate date range
            from_date = datetime.now() - timedelta(days=days)
            
            # Query for news insights
            news_query = {
                "cryptocurrencies": symbol,
                "source_type": "news",
                "processed_at": {"$gte": from_date}
            }
            
            news_results = self.db.find_many("extracted_information", news_query, 
                                          sort=[("processed_at", -1)], limit=20)
            
            # Query for market insights
            market_query = {
                "symbol": symbol,
                "source_type": "market",
                "processed_at": {"$gte": from_date}
            }
            
            market_results = self.db.find_many("extracted_information", market_query, 
                                            sort=[("processed_at", -1)], limit=10)
            
            # Extract key insights
            news_insights = []
            for result in news_results:
                info = result.get("info", {})
                
                # Include key events and sentiment
                events = info.get("key_events", [])
                sentiment = info.get("sentiment", "neutral")
                
                if events:
                    for event in events:
                        news_insights.append({
                            "type": "news_event",
                            "event_type": event.get("type", ""),
                            "description": event.get("description", ""),
                            "source": info.get("source", ""),
                            "timestamp": info.get("published_at", ""),
                            "sentiment": sentiment
                        })
            
            market_insights = []
            for result in market_results:
                info = result.get("info", {})
                insights = info.get("insights", [])
                
                for insight in insights:
                    market_insights.append({
                        "type": "market_indicator",
                        "insight_type": insight.get("type", ""),
                        "description": insight.get("insight", ""),
                        "significance": insight.get("significance", "medium"),
                        "timestamp": info.get("timestamp", "")
                    })
            
            # Combine results
            combined_insights = {
                "symbol": symbol,
                "time_range": f"{days} days",
                "news_insights": news_insights,
                "market_insights": market_insights,
                "total_insights": len(news_insights) + len(market_insights),
                "timestamp": datetime.now()
            }
            
            return combined_insights
        
        except Exception as e:
            logger.error(f"Error getting recent insights for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now()
            }

# Singleton instance
information_extraction = InformationExtraction()

# Helper function to get the singleton instance
def get_information_extraction():
    return information_extraction

# Example usage
if __name__ == "__main__":
    extractor = get_information_extraction()
    
    # Sample article
    sample_article = {
        "title": "Bitcoin Surges to $45,000 as Institutional Adoption Grows",
        "description": "BTC price jumped 15% following news of major financial institutions entering the crypto market.",
        "content": "The price of Bitcoin (BTC) has reached $45,000, representing a 15% increase over the past 24 hours. This surge comes as several major Wall Street firms announced new cryptocurrency investment products.",
        "url": "https://example.com/bitcoin-news",
        "source": {"name": "Crypto News"},
        "publishedAt": "2023-04-15T14:30:00Z"
    }
    
    info = extractor.extract_from_news(sample_article)
    print(f"Extracted information: {info}")