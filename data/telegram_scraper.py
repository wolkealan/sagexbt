import asyncio
import logging
from telethon import TelegramClient, events
from datetime import datetime
import pytz
import re
import os
import sys
from typing import Dict, List, Any, Optional

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.database import get_database
from config.config import TradingConfig, TelegramConfig, DatabaseConfig

# Create a dedicated logger for the telegram scraper
logger = get_logger('telegram_scraper')

class TelegramNewsScraper:
    """Collects news from Telegram channels in real-time for crypto analysis"""
    
    def __init__(self, api_id, api_hash, phone_number, session_name='telegram_scraper'):
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone_number = phone_number
        self.session_name = session_name
        self.db = get_database()
        
        # Use channels from TelegramConfig
        self.channels = TelegramConfig.CHANNELS
        self.client = None
        
        # Create collection if it doesn't exist
        if not DatabaseConfig.TELEGRAM_NEWS_COLLECTION:
            logger.warning("TELEGRAM_NEWS_COLLECTION not defined in DatabaseConfig, using default")
            self.collection_name = "telegram_news"
        else:
            self.collection_name = DatabaseConfig.TELEGRAM_NEWS_COLLECTION
    
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
            
            # Register event handler for new messages ONLY
            self.client.add_event_handler(
                self.handle_new_message, 
                events.NewMessage(chats=self.channels)
            )
            
            logger.info(f"Monitoring channels in REAL-TIME: {', '.join(self.channels)}")
        
        except Exception as e:
            logger.error(f"Error initializing Telegram client: {e}")
            raise
    
    async def handle_new_message(self, event):
        """Handle new messages from monitored channels in real-time"""
        try:
            # Skip messages without text
            if not event.message.text:
                return
            
            # Get chat entity
            chat = await event.get_chat()
            
            # Create document structure
            document = {
                'channel_id': chat.id,
                'channel_name': getattr(chat, 'title', chat.username),
                'username': chat.username,
                'message_id': event.message.id,
                'text': event.message.text,
                'date': event.message.date.isoformat(),
                'processed_date': datetime.now().isoformat(),
                'has_media': event.message.media is not None,
                'source': {
                    'id': f'telegram_{chat.username}',
                    'name': f'Telegram: {getattr(chat, "title", chat.username)}'
                },
                'source_type': 'telegram'
            }
            
            # Extract and add relevant crypto mentions
            document['coins_mentioned'] = self._extract_crypto_mentions(event.message.text)
            
            # Extract title from message
            document['title'] = self._extract_title(event.message.text)
            
            # Make the document compatible with NewsAPI format for LLM
            document['description'] = event.message.text
            document['url'] = f"https://t.me/{chat.username}/{event.message.id}" if chat.username else ""
            document['publishedAt'] = event.message.date.isoformat()
            
            # Store in MongoDB
            await self._store_message(document)
            
            # Log the processed message
            logger.info(f"Processed real-time message from {document['channel_name']}: {document['title'][:50]}...")
            
        except Exception as e:
            logger.error(f"Error processing real-time message: {e}")
    
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
        for coin in TradingConfig.SUPPORTED_COINS:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(coin) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                mentioned_coins.append(coin)
        
        # Check for coin names (using a mapping you should define in TradingConfig)
        coin_names = {
            "bitcoin": "BTC",
            "ethereum": "ETH",
            "solana": "SOL",
            # Add more mappings as needed
        }
        
        for name, symbol in coin_names.items():
            if symbol not in mentioned_coins:  # Skip if already found by symbol
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, text, re.IGNORECASE) and symbol in TradingConfig.SUPPORTED_COINS:
                    mentioned_coins.append(symbol)
        
        return mentioned_coins
    
    async def _store_message(self, document):
        """Store message in MongoDB"""
        try:
            # Create unique ID from channel and message ID
            query = {
                "channel_id": document['channel_id'],
                "message_id": document['message_id']
            }
            
            # Insert or update (upsert)
            result = await self.db.update_one(
                self.collection_name,
                query,
                {"$set": document},
                upsert=True
            )
            
            if result.upserted_id:
                logger.info(f"Stored new Telegram message with ID {document['message_id']}")
            else:
                logger.debug(f"Updated existing Telegram message with ID {document['message_id']}")
                
        except Exception as e:
            logger.error(f"Error storing message in MongoDB: {e}")
    
    async def run(self):
        """Run the scraper"""
        await self.initialize()
        
        try:
            logger.info("🔴 LIVE: Now monitoring Telegram channels in real-time")
            # Keep the client running to receive real-time updates
            await self.client.run_until_disconnected()
        except KeyboardInterrupt:
            logger.info("Stopping Telegram scraper due to keyboard interrupt")
        except Exception as e:
            logger.error(f"Telegram scraper error: {e}")
        finally:
            await self.client.disconnect()
            logger.info("Telegram client disconnected")

# Main function to run the scraper
async def run_telegram_scraper(api_id, api_hash, phone_number):
    scraper = TelegramNewsScraper(api_id, api_hash, phone_number)
    await scraper.run()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Run the scraper
    asyncio.run(run_telegram_scraper(
        TelegramConfig.API_ID, 
        TelegramConfig.API_HASH, 
        TelegramConfig.PHONE_NUMBER
    ))