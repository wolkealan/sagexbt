import requests
import json
import time
from typing import Dict, List, Any, Optional
import os

from config.config import APIConfig, AppConfig
from utils.logger import get_llm_logger

logger = get_llm_logger()

class DeepSeekLLM:
    """Interface for interacting with DeepSeek LLM API"""
    
    def __init__(self):
        self.api_key = APIConfig.DEEPSEEK_API_KEY
        self.api_base = APIConfig.DEEPSEEK_API_BASE
        self.model = AppConfig.LLM_MODEL
        self.temperature = AppConfig.LLM_TEMPERATURE
        self.max_tokens = AppConfig.LLM_MAX_TOKENS
    
    def chat_completion(self, messages: List[Dict[str, str]],
                        temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Send a chat completion request to the DeepSeek API
        
        Args:
            messages: List of message objects with 'role' and 'content'
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            API response as a dictionary
        """
        if not self.api_key:
            logger.error("DeepSeek API key not provided")
            raise ValueError("DeepSeek API key not provided")
        
        # Use provided parameters or fall back to defaults
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens
        }
        
        try:
            # Attempt to use OpenAI-compatible endpoint first (DeepSeek may support this)
            endpoint = f"{self.api_base}/v1/chat/completions"
            logger.debug(f"Sending request to: {endpoint}")
            
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload
            )
            
            # Check if request was successful
            response.raise_for_status()
            result = response.json()
            
            logger.info("Successfully received response from DeepSeek API")
            return result
            
        except requests.exceptions.RequestException as e:
            # If OpenAI-compatible endpoint fails, try DeepSeek's native endpoint format if different
            try:
                # This is a fallback in case DeepSeek uses a different API structure
                endpoint = f"{self.api_base}/api/v1/generate"
                logger.debug(f"First attempt failed, trying alternate endpoint: {endpoint}")
                
                # Adjust payload for potential different format
                alt_payload = {
                    "model": self.model,
                    "prompt": messages[-1]['content'],  # Use the last message content as prompt
                    "temperature": temp,
                    "max_tokens": tokens
                }
                
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=alt_payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info("Successfully received response from alternate DeepSeek API endpoint")
                return result
                
            except requests.exceptions.RequestException as e2:
                logger.error(f"Error communicating with DeepSeek API: {str(e2)}")
                raise RuntimeError(f"Failed to communicate with DeepSeek API: {str(e2)}")
    
    def generate_recommendation(self, 
                              coin: str, 
                              market_data: Dict[str, Any],
                              news_data: Dict[str, Any],
                              market_context: Dict[str, Any],
                              action_type: str = "spot") -> Dict[str, Any]:
        """
        Generate trading recommendation for a coin
        
        Args:
            coin: Cryptocurrency symbol
            market_data: Technical market data for the coin
            news_data: News and sentiment data for the coin
            market_context: Overall market context and sentiment
            action_type: 'spot' or 'futures' trading
            
        Returns:
            Recommendation with explanation
        """
        try:
            # Prepare data for the prompt
            current_price = market_data.get('current_price', 'Unknown')
            daily_change = market_data.get('daily_change_pct', 'Unknown')
            
            # Get indicators from market data
            rsi_1d = market_data.get('indicators', {}).get('1d', {}).get('rsi', 'Unknown')
            
            # Get sentiment from news data
            news_sentiment = news_data.get('sentiment', {}).get('sentiment', 'neutral')
            sentiment_score = news_data.get('sentiment', {}).get('sentiment_score', 0)
            
            # Build the prompt
            system_prompt = """You are a cryptocurrency trading advisor specialized in providing recommendations based on technical analysis, news sentiment, and market conditions.
Your task is to analyze the provided data and give a clear recommendation for the specified cryptocurrency.
Your recommendation should consider:
1. Technical indicators (RSI, moving averages, etc.)
2. Recent news sentiment
3. Overall market context
4. Current geopolitical and regulatory sentiment

For each recommendation:
- Provide a clear BUY, SELL, or HOLD recommendation
- Include a confidence level (Low, Medium, High)
- Explain your reasoning in a concise manner
- Mention key factors influencing your decision
- If recommending futures trading, specify long or short position
- Include relevant risk warnings
"""
            
            user_prompt = f"""Please analyze the following data and provide a trading recommendation for {coin}:

MARKET DATA:
- Current price: {current_price} USD
- 24h change: {daily_change}%
- RSI (1 day): {rsi_1d}
{self._format_indicators(market_data.get('indicators', {}))}

NEWS SENTIMENT:
- Overall sentiment: {news_sentiment} ({sentiment_score})
- Recent headlines: {self._format_headlines(news_data)}

MARKET CONTEXT:
- General market sentiment: {market_context.get('market', {}).get('sentiment', {}).get('sentiment', 'neutral')}
- Geopolitical sentiment: {market_context.get('geopolitical', {}).get('sentiment', {}).get('sentiment', 'neutral')}
- Regulatory sentiment: {market_context.get('regulatory', {}).get('sentiment', {}).get('sentiment', 'neutral')}

Please provide a {action_type.upper()} trading recommendation (BUY/SELL/HOLD) with explanation.
"""
            
            # Prepare messages for the API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Get response from LLM
            response = self.chat_completion(messages, temperature=0.2)
            
            # Extract the recommendation from the response
            if 'choices' in response and len(response['choices']) > 0:
                recommendation_text = response['choices'][0]['message']['content']
                
                # Parse the recommendation to extract the action
                action = self._parse_recommendation_action(recommendation_text)
                confidence = self._parse_recommendation_confidence(recommendation_text)
                
                recommendation = {
                    'coin': coin,
                    'action': action,
                    'confidence': confidence,
                    'action_type': action_type,
                    'explanation': recommendation_text,
                    'timestamp': time.time()
                }
                
                logger.info(f"Generated {action_type} recommendation for {coin}: {action} ({confidence} confidence)")
                return recommendation
                
            else:
                logger.error(f"Unexpected response format from LLM: {response}")
                return {
                    'coin': coin,
                    'action': 'HOLD',
                    'confidence': 'Low',
                    'action_type': action_type,
                    'explanation': "Unable to generate recommendation due to API response issue.",
                    'timestamp': time.time()
                }
                
        except Exception as e:
            logger.error(f"Error generating recommendation for {coin}: {e}")
            return {
                'coin': coin,
                'action': 'HOLD',
                'confidence': 'Low',
                'action_type': action_type,
                'explanation': f"Error generating recommendation: {str(e)}",
                'timestamp': time.time()
            }
    
    def _format_indicators(self, indicators: Dict[str, Any]) -> str:
        """Format technical indicators for the prompt"""
        result = []
        
        for timeframe, data in indicators.items():
            result.append(f"- Timeframe {timeframe}:")
            
            # Add RSI if available
            if 'rsi' in data:
                rsi = data['rsi']
                rsi_interpretation = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
                result.append(f"  * RSI: {rsi:.1f} ({rsi_interpretation})")
            
            # Add moving averages if available
            if 'moving_averages' in data and data['moving_averages']:
                mas = data['moving_averages']
                current_price = mas.get('current_price', 0)
                
                for ma_type, ma_value in mas.items():
                    if ma_type != 'current_price' and ma_value:
                        position = "above" if current_price > ma_value else "below"
                        result.append(f"  * {ma_type.upper()}: {ma_value:.2f} (price is {position})")
        
        return "\n".join(result)
    
    def _format_headlines(self, news_data: Dict[str, Any]) -> str:
        """Format recent headlines for the prompt"""
        articles = news_data.get('recent_articles', [])
        if not articles:
            return "No recent headlines available"
        
        headlines = []
        for article in articles[:3]:  # Use top 3 articles
            source = article.get('source', 'Unknown')
            title = article.get('title', 'No title')
            headlines.append(f"  * {title} ({source})")
        
        return "\n" + "\n".join(headlines)
    
    def _parse_recommendation_action(self, text: str) -> str:
        """Extract the recommended action (BUY/SELL/HOLD) from the response"""
        text_upper = text.upper()
        
        if "BUY" in text_upper or "LONG" in text_upper:
            return "BUY"
        elif "SELL" in text_upper or "SHORT" in text_upper:
            return "SELL"
        else:
            return "HOLD"
    
    def _parse_recommendation_confidence(self, text: str) -> str:
        """Extract the confidence level from the response"""
        text_lower = text.lower()
        
        if "high confidence" in text_lower or "strong" in text_lower:
            return "High"
        elif "low confidence" in text_lower or "weak" in text_lower or "uncertain" in text_lower:
            return "Low"
        else:
            return "Medium"

# Singleton instance
deepseek_llm = DeepSeekLLM()

# Helper function to get the singleton instance
def get_deepseek_llm():
    return deepseek_llm

# Example usage
if __name__ == "__main__":
    llm = get_deepseek_llm()
    # Test with a simple message
    response = llm.chat_completion([
        {"role": "user", "content": "What are the key factors to consider when trading cryptocurrencies?"}
    ])
    print(f"Response: {response}")