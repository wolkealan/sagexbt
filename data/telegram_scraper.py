import asyncio
import logging
from telethon import TelegramClient, events
from datetime import datetime
import pytz
import re
import os
import sys
from typing import Dict, List, Any, Optional
import hashlib

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.database import get_database
from config.config import TradingConfig, TelegramConfig, DatabaseConfig

# Create dedicated loggers
logger = get_logger('telegram_scraper')
news_logger = get_logger('news_provider')

class EnhancedTelegramScraper:
    """
    Enhanced Telegram scraper that collects messages from channels and categorizes
    them into different types (crypto news, geopolitical events, etc.)
    """
    
    def __init__(self, api_id, api_hash, phone_number, session_name='telegram_scraper'):
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone_number = phone_number
        self.session_name = session_name
        self.db = get_database()
        
        # Use channels from TelegramConfig
        self.channels = TelegramConfig.CHANNELS
        self.client = None
        
        # Define collection names
        if not DatabaseConfig.TELEGRAM_NEWS_COLLECTION:
            logger.warning("TELEGRAM_NEWS_COLLECTION not defined in DatabaseConfig, using default")
            self.telegram_news_collection = "telegram_news"
        else:
            self.telegram_news_collection = DatabaseConfig.TELEGRAM_NEWS_COLLECTION
            
        self.news_collection = DatabaseConfig.NEWS_COLLECTION
        
        # Keywords for categorization
        self.geopolitical_keywords = [
            "war", "conflict", "sanctions", "election", "protest", 
            "treaty", "agreement", "diplomatic", "relations",
            "terrorism", "military", "coup", "trade war", "tariffs",
            "nuclear", "missile", "UN", "NATO", "EU", "government",
            "president", "minister", "policy", "border", "crisis"
        ]
        
        self.crypto_keywords = [
            "bitcoin", "ethereum", "crypto", "blockchain", "token",
            "altcoin", "mining", "wallet", "exchange", "defi",
            "nft", "web3", "coin", "btc", "eth", "sol", "binance",
            "market", "trading", "price", "cryptocurrency", "ico",
            "xrp", "ripple", "cardano", "ada", "dogecoin", "doge"
        ]
        
        # Cache the supported coins (to avoid using the property repeatedly)
        self.supported_coins = self._get_supported_coins()
    
    def _get_supported_coins(self):
        """Safely get the supported coins, with a fallback if there are issues"""
        try:
            if hasattr(TradingConfig, 'SUPPORTED_COINS_CACHE') and TradingConfig.SUPPORTED_COINS_CACHE:
                return TradingConfig.SUPPORTED_COINS_CACHE
            elif hasattr(TradingConfig, 'DEFAULT_SUPPORTED_COINS'):
                return TradingConfig.DEFAULT_SUPPORTED_COINS
            else:
                # Fallback to a minimal list if nothing else is available
                return ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "SHIB", "DOT", "MATIC"]
        except Exception as e:
            logger.error(f"Error getting supported coins: {e}")
            return ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "SHIB", "DOT", "MATIC"]
    
    async def initialize(self):
        """Initialize the Telegram client"""
        try:
            # Create the Telegram client with explicit user authentication
            def auth_callback():
                return input('Enter the verification code you just received: ')
            
            self.client = TelegramClient(
                self.session_name, 
                self.api_id, 
                self.api_hash
            )
            
            # Start the client with more explicit authentication
            await self.client.start(
                phone=self.phone_number, 
                code_callback=auth_callback
            )
            
            logger.info("Telegram client started with user account")
            
            # Verify we're using a user account by checking user information
            me = await self.client.get_me()
            logger.info(f"Logged in as: {me.first_name} {me.last_name or ''}")
            
            # Register event handler for new messages
            self.client.add_event_handler(
                self.handle_new_message, 
                events.NewMessage(chats=self.channels)
            )
            
            logger.info(f"Monitoring channels in REAL-TIME: {', '.join(self.channels)}")
        
        except Exception as e:
            logger.error(f"Error initializing Telegram client: {e}")
            raise
    
    def _clean_message_text(self, text):
        """
        Clean message text by removing 'Tweet from X' prefix more aggressively
        Uses multiple methods to ensure we catch all variations
        """
        if not text:
            return ""
        
        # Split into lines
        lines = text.split('\n')
        if not lines:
            return ""
        
        # Multiple detection methods for "Tweet from" patterns
        first_line = lines[0].lower()
        
        # Method 1: Simple text detection
        contains_tweet = "tweet" in first_line or "tweets" in first_line
        contains_from = "from" in first_line
        
        # Method 2: Check for emoji followed by "Tweet from" pattern
        emoji_followed_by_tweet = re.search(r'[📝📊🐦📰📈💬🗣️📣🔊🗞️].*(?:tweet|tweets).*from', first_line, re.IGNORECASE)
        
        # Method 3: Check for formatted text with asterisks
        formatted_tweet_pattern = re.search(r'\*\*?tweet\*\*?.*\*\*?from\*\*?', first_line, re.IGNORECASE)
        
        if (contains_tweet and contains_from) or emoji_followed_by_tweet or formatted_tweet_pattern:
            # Skip the first line if it matches any of our patterns
            return '\n'.join(lines[1:]).strip()
        
        return text


    async def handle_new_message(self, event):
        """Handle new messages from monitored channels in real-time with aggressive cleaning"""
        try:
            # Skip messages without text
            if not event.message.text:
                return
            
            # Get chat entity
            chat = await event.get_chat()
            
            # Clean the message text by removing "Tweet from X" prefix
            original_text = event.message.text
            
            # Log the original text for debugging
            logger.debug(f"Original text: {original_text[:100]}")
            
            # Clean text using our improved method
            cleaned_text = self._clean_message_text(original_text)
            
            # Log the cleaned text for debugging
            logger.debug(f"Cleaned text: {cleaned_text[:100]}")
            
            # Create document structure
            document = {
                'channel_id': chat.id,
                'channel_name': getattr(chat, 'title', chat.username),
                'username': chat.username,
                'message_id': event.message.id,
                'text': cleaned_text,  # Use the cleaned text
                'date': event.message.date.isoformat(),
                'processed_date': datetime.now().isoformat(),
                'has_media': event.message.media is not None,
                'source': {
                    'id': f'telegram_{chat.username}',
                    'name': f'Telegram: {getattr(chat, "title", chat.username)}'
                },
                'source_type': 'telegram'
            }
            
            # Extract title from cleaned message
            document['title'] = self._extract_title(cleaned_text)
            
            # Make the document compatible with NewsAPI format for LLM
            document['description'] = cleaned_text  # Use cleaned text for description
            document['url'] = f"https://t.me/{chat.username}/{event.message.id}" if chat.username else ""
            document['publishedAt'] = event.message.date.isoformat()
            
            # Extract and add relevant crypto mentions
            document['coins_mentioned'] = self._extract_crypto_mentions(cleaned_text)
            
            # Store in original telegram_news collection
            await self._store_telegram_message(document)
            
            # Also store in news_data collection with categorization
            await self._store_news_document(document)
            
            # Log the processed message
            logger.info(f"Processed real-time message from {document['channel_name']}: {document['title'][:50]}...")
            
        except Exception as e:
            logger.error(f"Error processing real-time message: {e}")
            # Log the full error traceback for easier debugging
            import traceback
            logger.error(traceback.format_exc())
    
    def _extract_title(self, text):
        """Extract a title from the message text"""
        if not text:
            return "No content"
        
        # Remove markdown and emoji from the start
        text = text.lstrip('*🚨🔥🇺🇸⚡️')
        text = text.replace('**', '')
        
        # Split into lines and get the first meaningful line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return "Empty message"
        
        first_line = lines[0]
        
        # Truncate if too long
        if len(first_line) > 100:
            first_line = first_line[:97] + "..."
        
        return first_line
    
    def _extract_crypto_mentions(self, text):
        """Extract cryptocurrency mentions from text"""
        if not text:
            return []
            
        mentioned_coins = []
        
        # Check for supported coins by symbol
        for coin in self.supported_coins:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(coin) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                mentioned_coins.append(coin)
        
        # Check for coin names (using a mapping)
        coin_names = {
            "bitcoin": "BTC",
            "ethereum": "ETH",
            "solana": "SOL",
            "ripple": "XRP",
            "cardano": "ADA",
            "dogecoin": "DOGE",
            "polkadot": "DOT",
            "shiba": "SHIB",
            "avalanche": "AVAX",
            "chainlink": "LINK",
            "polygon": "MATIC",
            "binance": "BNB",
            "litecoin": "LTC",
            "uniswap": "UNI",
            "stellar": "XLM",
            "cosmos": "ATOM"
        }
        
        for name, symbol in coin_names.items():
            if symbol not in mentioned_coins:  # Skip if already found by symbol
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, text, re.IGNORECASE) and symbol in self.supported_coins:
                    mentioned_coins.append(symbol)
        
        return mentioned_coins
    
    def _is_geopolitical_content(self, text):
        """Determine if the text is about geopolitical events"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.geopolitical_keywords)
    
    def _is_crypto_content(self, text):
        """Determine if the text is about cryptocurrency"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crypto_keywords)
    
    def _determine_content_type(self, text):
        """Determine the type of content in the message"""
        is_geo = self._is_geopolitical_content(text)
        is_crypto = self._is_crypto_content(text)
        
        if is_geo and is_crypto:
            return "mixed"
        elif is_geo:
            return "geopolitical"
        elif is_crypto:
            return "crypto"
        else:
            return "general"
    
    def _determine_region(self, text):
        """Determine the region mentioned in the text"""
        text_lower = text.lower()
        
        regions = {
            "North America": ["united states", "us", "usa", "america", "canada", "mexico"],
            "Europe": ["europe", "eu", "european union", "uk", "britain", "germany", "france", "italy", "spain"],
            "Asia": ["asia", "china", "japan", "india", "south korea", "north korea"],
            "Middle East": ["middle east", "iran", "iraq", "saudi arabia", "israel", "syria", "turkey", "egypt"],
            "Africa": ["africa", "nigeria", "egypt", "south africa", "ethiopia", "kenya"],
            "Latin America": ["latin america", "brazil", "argentina", "colombia", "venezuela", "chile"],
            "Russia": ["russia", "russian", "moscow", "putin"],
            "Global": ["global", "world", "international", "un", "united nations", "g20", "g7"]
        }
        
        for region, keywords in regions.items():
            if any(keyword in text_lower for keyword in keywords):
                return region
        
        return "Unknown"
    
    def _determine_event_type(self, text):
        """Determine the type of geopolitical event"""
        text_lower = text.lower()
        
        # Check for conflict
        if any(word in text_lower for word in ["war", "conflict", "attack", "invasion", "battle", "fighting"]):
            return "conflict"
        
        # Check for diplomacy
        if any(word in text_lower for word in ["treaty", "agreement", "diplomatic", "relations", "talks", "peace"]):
            return "diplomacy"
        
        # Check for elections
        if any(word in text_lower for word in ["election", "vote", "ballot", "campaign", "candidate", "president"]):
            return "election"
        
        # Check for trade
        if any(word in text_lower for word in ["trade", "tariff", "economic", "import", "export", "commerce"]):
            return "trade"
        
        # Check for regulation
        if any(word in text_lower for word in ["regulation", "law", "policy", "ban", "restrict", "compliance"]):
            return "regulation"
        
        # Check for terrorism
        if any(word in text_lower for word in ["terrorism", "terrorist", "attack", "bomb", "extremist"]):
            return "terrorism"
        
        # Default to "other"
        return "other"
    
    def _determine_impact_level(self, text):
        """Determine the impact level of a geopolitical event"""
        text_lower = text.lower()
        
        # High impact keywords
        high_impact = ["global crisis", "world war", "nuclear", "catastrophic", "major conflict", 
                      "international crisis", "economic collapse", "massive impact"]
        
        # Medium impact keywords
        medium_impact = ["sanctions", "trade war", "conflict", "significant", "important", 
                        "election", "diplomatic crisis", "economic impact"]
        
        # Check for high impact
        if any(keyword in text_lower for keyword in high_impact):
            return "high"
        
        # Check for medium impact
        if any(keyword in text_lower for keyword in medium_impact):
            return "medium"
        
        # Default to "low"
        return "low"
    
    async def _store_telegram_message(self, document):
        """Store message in the telegram_news collection"""
        try:
            # Create unique ID from channel and message ID
            query = {
                "channel_id": document['channel_id'],
                "message_id": document['message_id']
            }
            
            # Insert or update (upsert)
            result = self.db.update_one(
                self.telegram_news_collection,
                query,
                {"$set": document},
                upsert=True
            )
            
            logger.debug(f"Stored/updated message in {self.telegram_news_collection}")
            return True
        except Exception as e:
            logger.error(f"Error storing message in {self.telegram_news_collection}: {e}")
            return False
    
    # In the _store_news_document method, modify to ensure unique entries:
    async def _store_news_document(self, document):
        """Store message directly in the news_data collection"""
        try:
            # Determine content type
            text = document.get('text', '')
            content_type = self._determine_content_type(text)
            
            # Create a simplified article document with a timestamp to ensure uniqueness
            current_time = datetime.now().isoformat()
            article = {
                'title': document.get('title', ''),
                'description': document.get('text', ''),
                'telegram_message_id': document.get('message_id', ''),
                'channel_name': document.get('channel_name', ''),
                'publishedAt': document.get('date', current_time),
                'processed_at': current_time,
                'source_type': 'telegram',
                'content_type': content_type,
                'coins_mentioned': document.get('coins_mentioned', [])
            }
            
            # Add geopolitical metadata if applicable
            if content_type in ["geopolitical", "mixed"]:
                article['region'] = self._determine_region(text)
                article['event_type'] = self._determine_event_type(text)
                article['impact_level'] = self._determine_impact_level(text)
            
            # Insert directly without worrying about duplication
            result = self.db.insert_one(self.news_collection, article)
            
            if result:
                news_logger.info(f"Stored {content_type} message in news_data")
                return True
            else:
                news_logger.warning(f"Failed to store {content_type} message in news_data")
                return False
                
        except Exception as e:
            logger.error(f"Error storing message in {self.news_collection}: {e}")
            return False
    
    async def run(self):
        """Run the scraper"""
        await self.initialize()
        
        try:
            logger.info("🔴 LIVE: Now monitoring Telegram channels in real-time")
            await self.client.run_until_disconnected()
        except KeyboardInterrupt:
            logger.info("Stopping Telegram scraper due to keyboard interrupt")
        except Exception as e:
            logger.error(f"Telegram scraper error: {e}")
        finally:
            await self.client.disconnect()
            logger.info("Telegram client disconnected")

# Main function to run the enhanced scraper
async def run_enhanced_telegram_scraper(api_id, api_hash, phone_number):
    """Main function to run the enhanced telegram scraper"""
    try:
        scraper = EnhancedTelegramScraper(api_id, api_hash, phone_number)
        await scraper.run()
    except Exception as e:
        logger.error(f"Error running enhanced telegram scraper: {e}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Run the scraper
    asyncio.run(run_enhanced_telegram_scraper(
        TelegramConfig.API_ID, 
        TelegramConfig.API_HASH, 
        TelegramConfig.PHONE_NUMBER
    ))