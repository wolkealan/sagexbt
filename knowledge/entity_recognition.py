import re
from typing import Dict, List, Any, Optional, Set, Tuple
import nltk
from nltk.tokenize import word_tokenize
from datetime import datetime

from config.config import TradingConfig
from utils.logger import get_logger

logger = get_logger("entity_recognition")

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class EntityRecognition:
    """Recognizes and extracts relevant financial entities from text data"""
    
    def __init__(self):
        self.supported_coins = TradingConfig.SUPPORTED_COINS
        self.coin_aliases = self._initialize_coin_aliases()
        self.financial_terms = self._initialize_financial_terms()
        self.entity_patterns = self._initialize_entity_patterns()
    
    def _initialize_coin_aliases(self) -> Dict[str, List[str]]:
        """Initialize cryptocurrency aliases/full names"""
        return {
            "BTC": ["Bitcoin", "BTC", "XBT", "BTCUSD"],
            "ETH": ["Ethereum", "ETH", "Ether", "ETHUSD"],
            "BNB": ["Binance Coin", "BNB", "BNBUSD"],
            "SOL": ["Solana", "SOL", "SOLUSD"],
            "XRP": ["Ripple", "XRP", "XRPUSD"],
            "ADA": ["Cardano", "ADA", "ADAUSD"],
            "DOGE": ["Dogecoin", "DOGE", "DOGEUSD"],
            "SHIB": ["Shiba Inu", "SHIB", "SHIBUSD"],
            "DOT": ["Polkadot", "DOT", "DOTUSD"],
            "MATIC": ["Polygon", "MATIC", "MATICUSD"],
            "AVAX": ["Avalanche", "AVAX", "AVAXUSD"],
            "LINK": ["Chainlink", "LINK", "LINKUSD"],
            "UNI": ["Uniswap", "UNI", "UNIUSD"],
            "LTC": ["Litecoin", "LTC", "LTCUSD"],
            "ATOM": ["Cosmos", "ATOM", "ATOMUSD"]
        }
    
    def _initialize_financial_terms(self) -> Dict[str, List[str]]:
        """Initialize common financial and trading terms for recognition"""
        return {
            "bullish": ["bull", "bullish", "uptrend", "rally", "surge", "skyrocket", "moon", "outperform"],
            "bearish": ["bear", "bearish", "downtrend", "decline", "crash", "dump", "plummet", "underperform"],
            "volatility": ["volatile", "volatility", "unstable", "fluctuation", "swing", "erratic"],
            "support": ["support", "floor", "bottom", "accumulation", "buying pressure"],
            "resistance": ["resistance", "ceiling", "top", "distribution", "selling pressure"],
            "trading_volume": ["volume", "liquidity", "trading volume", "market activity", "turnover"],
            "market_cap": ["market cap", "market capitalization", "valuation", "value"],
            "regulation": ["regulation", "regulatory", "compliance", "SEC", "CFTC", "ban", "legal", "tax"],
            "adoption": ["adoption", "institutional", "mainstream", "integration", "partnership", "enterprise"],
            "technology": ["blockchain", "protocol", "upgrade", "fork", "layer", "scalability", "security"]
        }
    
    def _initialize_entity_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for entity recognition"""
        return {
            "price": re.compile(r'\$\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?\s*(?:dollars|USD)'),
            "percentage": re.compile(r'\d+(?:\.\d+)?%'),
            "date": re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}|\d{4}-\d{2}-\d{2}'),
            "timeframe": re.compile(r'\d+\s*(?:minute|min|hour|hr|day|week|month|year)s?'),
            "person": re.compile(r'(?:[A-Z][a-z]+\s)+(?:[A-Z][a-z]+)'),  # Simplified person name
        }
    
    def detect_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Detect financial entities in text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of detected entities by category
        """
        try:
            if not text:
                return {}
            
            # Initialize results
            entities = {
                "cryptocurrencies": [],
                "sentiment": [],
                "financial_terms": [],
                "price_mentions": [],
                "percentage_changes": [],
                "dates": [],
                "timeframes": [],
                "people": []
            }
            
            # Tokenize text
            tokens = word_tokenize(text)
            
            # Detect cryptocurrencies and their aliases
            for coin, aliases in self.coin_aliases.items():
                for alias in aliases:
                    if alias.lower() in text.lower() or alias in text:
                        if coin not in entities["cryptocurrencies"]:
                            entities["cryptocurrencies"].append(coin)
            
            # Detect sentiment and financial terms
            text_lower = text.lower()
            
            # Check for bullish/bearish sentiment
            for term in self.financial_terms["bullish"]:
                if term.lower() in text_lower:
                    entities["sentiment"].append("bullish")
                    entities["financial_terms"].append(term)
            
            for term in self.financial_terms["bearish"]:
                if term.lower() in text_lower:
                    entities["sentiment"].append("bearish")
                    entities["financial_terms"].append(term)
            
            # Check for other financial terms
            for category, terms in self.financial_terms.items():
                if category not in ["bullish", "bearish"]:  # Already processed these
                    for term in terms:
                        if term.lower() in text_lower:
                            entities["financial_terms"].append(term)
            
            # Use regex patterns to find structured entities
            for entity_type, pattern in self.entity_patterns.items():
                matches = pattern.findall(text)
                if matches:
                    # Map entity type to the appropriate result category
                    if entity_type == "price":
                        entities["price_mentions"] = matches
                    elif entity_type == "percentage":
                        entities["percentage_changes"] = matches
                    elif entity_type == "date":
                        entities["dates"] = matches
                    elif entity_type == "timeframe":
                        entities["timeframes"] = matches
                    elif entity_type == "person":
                        entities["people"] = matches
            
            # Remove duplicates
            for category in entities:
                entities[category] = list(set(entities[category]))
            
            # Remove empty categories
            entities = {k: v for k, v in entities.items() if v}
            
            logger.debug(f"Detected {sum(len(v) for v in entities.values())} entities in text")
            return entities
        
        except Exception as e:
            logger.error(f"Error detecting entities: {e}")
            return {}
    
    def extract_coin_mentions(self, text: str) -> List[str]:
        """Extract cryptocurrency mentions from text"""
        try:
            entities = self.detect_entities(text)
            return entities.get("cryptocurrencies", [])
        except Exception as e:
            logger.error(f"Error extracting coin mentions: {e}")
            return []
    
    def extract_sentiment(self, text: str) -> str:
        """Extract overall sentiment from text"""
        try:
            entities = self.detect_entities(text)
            sentiment = entities.get("sentiment", [])
            
            if "bullish" in sentiment and "bearish" in sentiment:
                return "mixed"
            elif "bullish" in sentiment:
                return "bullish"
            elif "bearish" in sentiment:
                return "bearish"
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Error extracting sentiment: {e}")
            return "neutral"
    
    def extract_price_mentions(self, text: str, coin: str = None) -> List[Dict[str, Any]]:
        """Extract price mentions with associated cryptocurrency"""
        try:
            entities = self.detect_entities(text)
            price_mentions = entities.get("price_mentions", [])
            
            if not price_mentions:
                return []
            
            result = []
            coins = entities.get("cryptocurrencies", [])
            
            # If a specific coin is provided, only look for that one
            if coin:
                if coin in coins:
                    for price in price_mentions:
                        result.append({
                            "coin": coin,
                            "price": price,
                            "confidence": "high" if len(coins) == 1 else "medium"
                        })
            else:
                # Try to associate prices with coins
                if len(coins) == 1 and len(price_mentions) >= 1:
                    # If there's only one coin mentioned, associate all prices with it
                    for price in price_mentions:
                        result.append({
                            "coin": coins[0],
                            "price": price,
                            "confidence": "high"
                        })
                elif len(coins) > 1:
                    # Multiple coins, try to associate prices based on proximity
                    # This is a simplified approach; in a real system, you'd use NLP dependency parsing
                    sentences = re.split(r'[.!?]', text)
                    for sentence in sentences:
                        sentence_coins = [c for c in coins if c in sentence.upper() or 
                                         any(alias.lower() in sentence.lower() for alias in self.coin_aliases.get(c, []))]
                        sentence_prices = self.entity_patterns["price"].findall(sentence)
                        
                        if len(sentence_coins) == 1 and sentence_prices:
                            # One coin in this sentence, associate all prices
                            for price in sentence_prices:
                                result.append({
                                    "coin": sentence_coins[0],
                                    "price": price,
                                    "confidence": "medium"
                                })
                
                # Add any unassociated prices with low confidence
                all_processed_prices = [item["price"] for item in result]
                unprocessed_prices = [p for p in price_mentions if p not in all_processed_prices]
                
                if unprocessed_prices and coins:
                    for price in unprocessed_prices:
                        result.append({
                            "coin": coins[0],  # Assign to first coin with low confidence
                            "price": price,
                            "confidence": "low"
                        })
            
            return result
        except Exception as e:
            logger.error(f"Error extracting price mentions: {e}")
            return []
    
    def summarize_entities(self, text: str) -> Dict[str, Any]:
        """Provide a summary of detected entities in the text"""
        try:
            entities = self.detect_entities(text)
            
            # Count entities by category
            entity_counts = {category: len(items) for category, items in entities.items()}
            
            # Get overall sentiment
            sentiment = self.extract_sentiment(text)
            
            # Identify main cryptocurrency focus
            main_crypto = None
            cryptos = entities.get("cryptocurrencies", [])
            if cryptos:
                # Simple approach: first mentioned is main focus
                # Could be improved with frequency analysis
                main_crypto = cryptos[0]
            
            # Extract price mentions
            price_mentions = self.extract_price_mentions(text)
            
            # Create summary
            summary = {
                "main_cryptocurrency": main_crypto,
                "all_cryptocurrencies": cryptos,
                "sentiment": sentiment,
                "entity_counts": entity_counts,
                "price_mentions": price_mentions,
                "financial_terms": entities.get("financial_terms", []),
                "timestamp": datetime.now()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing entities: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now()
            }

# Singleton instance
entity_recognition = EntityRecognition()

# Helper function to get the singleton instance
def get_entity_recognition():
    return entity_recognition

# Example usage
if __name__ == "__main__":
    recognizer = get_entity_recognition()
    sample_text = "Bitcoin surged to $45,000 yesterday, showing a 15% increase. Ethereum also performed well, reaching $3,200. Analysts expect BTC to test the resistance at $48,000 in the coming weeks."
    entities = recognizer.detect_entities(sample_text)
    print(f"Detected entities: {entities}")
    
    sentiment = recognizer.extract_sentiment(sample_text)
    print(f"Sentiment: {sentiment}")
    
    summary = recognizer.summarize_entities(sample_text)
    print(f"Entity summary: {summary}")