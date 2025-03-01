import requests
import json
import time
from typing import Dict, List, Any, Optional
import os
import math
import numpy as np
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
            
    # 3. Add helper method for formatting patterns in the DeepSeekLLM class

    def _format_patterns(self, pattern_data: Dict[str, Any]) -> str:
        """Format technical pattern data for the prompt - with robust handling of NaN values"""

        
        if not pattern_data or not isinstance(pattern_data, dict):
            return "No pattern data available"
        
        # Helper function to check if a value is NaN
        def is_not_nan(value):
            if isinstance(value, (float, np.float64, np.float32)):
                return not (math.isnan(value) or np.isnan(value))
            return True
        
        # Helper function to safely get a numeric value
        def safe_format(value, format_str="%.2f"):
            if not is_not_nan(value):
                return "N/A"
            try:
                return format_str % float(value)
            except (ValueError, TypeError):
                return "N/A"
        
        result = []
        
        # Format trend data
        if "trend" in pattern_data and isinstance(pattern_data["trend"], dict) and "error" not in pattern_data["trend"]:
            trend = pattern_data["trend"]
            if "overall" in trend:
                overall = trend.get("overall", "neutral")
                result.append(f"- Overall Trend: {overall.replace('_', ' ').title()}")
            if "short_term" in trend:
                result.append(f"  * Short-term trend: {trend['short_term']}")
            if "medium_term" in trend:
                result.append(f"  * Medium-term trend: {trend['medium_term']}")
            if "long_term" in trend:
                result.append(f"  * Long-term trend: {trend['long_term']}")
            if "special_event" in trend:
                result.append(f"  * Special event: {trend['special_event']}")
        
        # Format support/resistance data
        if "support_resistance" in pattern_data and isinstance(pattern_data["support_resistance"], dict) and "error" not in pattern_data["support_resistance"]:
            sr = pattern_data["support_resistance"]
            if "support" in sr and sr["support"]:
                supports = sr["support"]
                result.append("- Support Levels:")
                for level in supports:
                    if isinstance(level, dict) and "level" in level and is_not_nan(level['level']):
                        try:
                            strength = level.get('strength', 'N/A')
                            strength_str = safe_format(strength) if strength != 'N/A' else 'N/A'
                            result.append(f"  * ${float(level['level']):.2f} (strength: {strength_str})")
                        except (ValueError, TypeError):
                            pass
            if "resistance" in sr and sr["resistance"]:
                resistances = sr["resistance"]
                result.append("- Resistance Levels:")
                for level in resistances:
                    if isinstance(level, dict) and "level" in level and is_not_nan(level['level']):
                        try:
                            strength = level.get('strength', 'N/A')
                            strength_str = safe_format(strength) if strength != 'N/A' else 'N/A'
                            result.append(f"  * ${float(level['level']):.2f} (strength: {strength_str})")
                        except (ValueError, TypeError):
                            pass
        
        # Format candlestick patterns
        if "candlestick" in pattern_data and pattern_data["candlestick"] and "error" not in pattern_data["candlestick"]:
            candles = pattern_data["candlestick"]
            if candles and isinstance(candles, dict):
                result.append("- Candlestick Patterns:")
                for name, details in candles.items():
                    if isinstance(details, dict) and "significance" in details and "description" in details:
                        result.append(f"  * {details.get('type', name).replace('_', ' ').title()} ({details['significance']}): {details['description']}")
        
        # Format chart patterns
        if "chart_patterns" in pattern_data and pattern_data["chart_patterns"] and "error" not in pattern_data["chart_patterns"]:
            charts = pattern_data["chart_patterns"]
            if charts and isinstance(charts, dict):
                result.append("- Chart Patterns:")
                for name, details in charts.items():
                    if isinstance(details, dict) and "significance" in details and "description" in details:
                        pattern_type = details.get('type', name).replace('_', ' ').title()
                        if "level" in details and is_not_nan(details['level']):
                            level_str = safe_format(details['level'])
                            result.append(f"  * {pattern_type} ({details['significance']}): {details['description']} - Level: ${level_str}")
                        elif "neckline" in details and is_not_nan(details['neckline']):
                            neckline_str = safe_format(details['neckline'])
                            result.append(f"  * {pattern_type} ({details['significance']}): {details['description']} - Neckline: ${neckline_str}")
                        else:
                            result.append(f"  * {pattern_type} ({details['significance']}): {details['description']}")
        
        return "\n".join(result) if result else "No significant technical patterns detected"
    
    def generate_recommendation(self, 
                  coin: str, 
                  market_data: Dict[str, Any],
                  news_data: Dict[str, Any],
                  market_context: Dict[str, Any],
                  pattern_data: Dict[str, Any] = None,  # Add this parameter
                  action_type: str = "spot") -> Dict[str, Any]:
        try:
            # Prepare data for the prompt
            current_price = market_data.get('current_price', 'Unknown')
            daily_change = market_data.get('daily_change_pct', 'Unknown')
            
            # Get indicators from market data
            rsi_1d = market_data.get('indicators', {}).get('1d', {}).get('rsi', 'Unknown')
            
            # Improved news sentiment extraction
            news_sentiment = 'neutral'
            sentiment_score = 0
            headlines = "No recent headlines available"
            
            # More robust news sentiment extraction
            if news_data and isinstance(news_data, dict):
                news_sentiment = news_data.get('sentiment', {}).get('sentiment', 'neutral')
                sentiment_score = news_data.get('sentiment', {}).get('sentiment_score', 0)
                
                # Extract headlines
                recent_articles = news_data.get('recent_articles', [])
                if recent_articles:
                    headlines = "\n".join([
                        f"  * {article.get('title', 'No title')} ({article.get('source', {}).get('name', 'Unknown')})" 
                        for article in recent_articles[:3]
                    ])
            
            # Format pattern recognition data
            pattern_analysis = self._format_patterns(pattern_data) if pattern_data else "No pattern data available"
        
            # Build the prompt
            system_prompt = """You are a cryptocurrency trading advisor specialized in providing recommendations based on technical analysis, news sentiment, and market conditions.
Your task is to analyze the provided data and give a clear recommendation for the specified cryptocurrency.

IMPORTANT: 
1. Always include the current price in your recommendation near the beginning of your analysis.
2. ALWAYS mention the specific technical patterns identified (support/resistance levels, chart patterns, trend direction)
3. If support and resistance levels are provided, ALWAYS include them in your recommendation 

Your recommendation should consider:
1. Technical indicators (RSI, moving averages, etc.)
2. Technical chart patterns and trend analysis (ALWAYS mention these explicitly)
3. Recent news sentiment
4. Overall market context
5. Support and resistance levels (ALWAYS mention these explicitly if available)

For each recommendation:
- Provide a clear BUY, SELL, or HOLD recommendation
- Include a confidence level (Low, Medium, High)
- Explicitly mention the current price in your analysis
- Explain your reasoning in a concise manner
- Mention key factors influencing your decision
- If recommending futures trading, specify long or short position
- Include relevant risk warnings
- Reference important support/resistance levels when available
"""
        
            user_prompt = f"""Please analyze the following data and provide a trading recommendation for {coin}:

MARKET DATA:
- Current price: ${current_price} USD
- 24h change: {daily_change}%
- RSI (1 day): {rsi_1d}
{self._format_indicators(market_data.get('indicators', {}))}

TECHNICAL PATTERNS:
{pattern_analysis}

NEWS SENTIMENT:
- Overall sentiment: {news_sentiment} ({sentiment_score})
- Recent headlines: 
{headlines}

MARKET CONTEXT:
- General market sentiment: {market_context.get('market', {}).get('sentiment', {}).get('sentiment', 'neutral')}
- Geopolitical sentiment: {market_context.get('geopolitical', {}).get('sentiment', {}).get('sentiment', 'neutral')}
- Regulatory sentiment: {market_context.get('regulatory', {}).get('sentiment', {}).get('sentiment', 'neutral')}

Please provide a {action_type.upper()} trading recommendation (BUY/SELL/HOLD) with explanation.
Make sure to include the current price (${current_price} USD) in your analysis.
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
                
                # Create the recommendation structure
                recommendation = {
                    'coin': coin,
                    'action': action,
                    'confidence': confidence,
                    'action_type': action_type,
                    'explanation': recommendation_text,
                    'timestamp': time.time(),
                    'context': {
                        'market_data': {
                            'price': current_price,
                            'daily_change': daily_change,
                            'rsi_1d': rsi_1d
                        },
                        'news_sentiment': news_sentiment,
                        'sentiment_score': sentiment_score,
                        'patterns': pattern_data  # Add pattern data to the context
                    }
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
                    'timestamp': time.time(),
                    'context': {
                        'market_data': {
                            'price': current_price,
                            'daily_change': daily_change,
                            'rsi_1d': rsi_1d
                        }
                    }
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
            
            # Add MACD if available
            if 'macd' in data and data['macd']:
                macd = data['macd']
                macd_line = macd.get('macd_line', 0)
                signal_line = macd.get('signal_line', 0)
                histogram = macd.get('macd_histogram', 0)
                
                signal = "bullish" if macd_line > signal_line else "bearish"
                trend_strength = "strengthening" if histogram > 0 else "weakening"
                
                result.append(f"  * MACD: Line {macd_line:.4f}, Signal {signal_line:.4f}, Histogram {histogram:.4f}")
                result.append(f"    ({signal} signal, trend {trend_strength})")
            
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