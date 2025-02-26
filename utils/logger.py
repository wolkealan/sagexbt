import logging
import sys
import os
from datetime import datetime
from pathlib import Path

from config.config import AppConfig, BASE_DIR

# Create logs directory if it doesn't exist
logs_dir = Path(BASE_DIR) / "logs"
logs_dir.mkdir(exist_ok=True)

# Generate log filename with current date
log_filename = f"crypto_advisor_{datetime.now().strftime('%Y-%m-%d')}.log"
log_filepath = logs_dir / log_filename

# Configure logging
def setup_logger(name, level=AppConfig.LOG_LEVEL):
    """Set up and return a logger with the specified name and level"""
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Main application logger
def get_logger(module_name=None):
    """Get a logger for the specified module"""
    name = module_name if module_name else "crypto_advisor"
    return setup_logger(name)

# Special loggers for specific subsystems
def get_market_logger():
    return setup_logger("crypto_advisor.market")

def get_news_logger():
    return setup_logger("crypto_advisor.news")

def get_llm_logger():
    return setup_logger("crypto_advisor.llm")

def get_api_logger():
    return setup_logger("crypto_advisor.api")