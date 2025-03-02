import requests
import json
import time
from typing import Dict, List, Any, Optional
import os
import math
import numpy as np
from config.config import APIConfig, AppConfig
from utils.logger import get_llm_logger
from datetime import datetime, timezone, timedelta
logger = get_llm_logger()

class GrokLLM:
    """Interface for interacting with Grok LLM API"""
    
    def __init__(self):
        self.api_key = APIConfig.GROK_API_KEY
        self.api_base = APIConfig.GROK_API_BASE
        self.model = AppConfig.LLM_MODEL
        self.temperature = AppConfig.LLM_TEMPERATURE
        self.max_tokens = AppConfig.LLM_MAX_TOKENS
    
    def chat_completion(self, messages: List[Dict[str, str]],
                        temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Send a chat completion request to the Grok API
        
        Args:
            messages: List of message objects with 'role' and 'content'
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            API response as a dictionary
        """
        if not self.api_key:
            logger.error("Grok API key not provided")
            raise ValueError("Grok API key not provided")
        
        # Use provided parameters or fall back to defaults
        temp = temperature if temperature is not None else self.temperature
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "stream": False
        }
        
        # Add max_tokens if provided
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        try:
            endpoint = f"{self.api_base}/v1/chat/completions"
            logger.debug(f"Sending request to: {endpoint} with model: {self.model}")
            logger.debug(f"Sending request to: {endpoint}")
            
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                verify=False  # Temporarily disable SSL verification
            )
            
            # Check if request was successful
            response.raise_for_status()
            result = response.json()
            
            logger.info("Successfully received response from Grok API")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Grok API: {str(e)}")
            raise RuntimeError(f"Failed to communicate with Grok API: {str(e)}")
    
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
    
    def _calculate_leverage_recommendation(self, 
                                      risk_tolerance: str,
                                      market_data: Dict[str, Any],
                                      pattern_data: Dict[str, Any],
                                      confidence: str) -> Dict[str, Any]:
        """
        Calculate appropriate leverage based on multiple factors
        
        Args:
            risk_tolerance: User's risk tolerance (low, medium, high)
            market_data: Market data including volatility
            pattern_data: Technical patterns data
            confidence: Recommendation confidence level
            
        Returns:
            Dict with leverage recommendation and explanation
        """
        # Base leverage ranges by risk tolerance
        leverage_ranges = {
            "low": {"min": 1, "max": 2, "description": "very conservative"},
            "medium": {"min": 2, "max": 5, "description": "moderate"},
            "high": {"min": 5, "max": 20, "description": "aggressive"}
        }
        
        # Get base range based on risk tolerance (default to medium if not specified)
        base_range = leverage_ranges.get(risk_tolerance.lower(), leverage_ranges["medium"])
        
        # Adjust for market volatility
        volatility_factor = 1.0
        if "volatility" in market_data:
            vol = market_data.get("volatility", 0)
            if vol > 5:  # High volatility
                volatility_factor = 0.5
            elif vol > 3:  # Medium volatility
                volatility_factor = 0.7
        
        # Adjust for confidence level
        confidence_factor = 1.0
        if confidence == "Low":
            confidence_factor = 0.6
        elif confidence == "High":
            confidence_factor = 1.2
        
        # Adjust for proximity to support/resistance
        proximity_factor = 1.0
        if pattern_data and "support_resistance" in pattern_data:
            sr_data = pattern_data.get("support_resistance", {})
            if "proximity_to_level" in sr_data:
                proximity = sr_data.get("proximity_to_level", 0)
                if proximity < 0.05:  # Very close to a level (within 5%)
                    proximity_factor = 0.7  # Reduce leverage near key levels
        
        # Calculate range
        min_leverage = max(1, base_range["min"] * volatility_factor * confidence_factor * proximity_factor)
        max_leverage = max(min_leverage, base_range["max"] * volatility_factor * confidence_factor * proximity_factor)
        
        # Round to clean numbers
        min_leverage = round(min_leverage)
        max_leverage = round(max_leverage)
        
        # Generate explanation
        factors = []
        factors.append(f"{risk_tolerance} risk tolerance suggesting {base_range['description']} leverage")
        
        if volatility_factor < 1:
            factors.append("elevated market volatility (suggesting lower leverage)")
        
        if confidence_factor < 1:
            factors.append("lower confidence in the trade direction")
        elif confidence_factor > 1:
            factors.append("higher confidence in the trade direction")
        
        if proximity_factor < 1:
            factors.append("proximity to key support/resistance levels")
        
        explanation = f"Based on {', '.join(factors)}"
        
        return {
            "min_leverage": min_leverage,
            "max_leverage": max_leverage,
            "explanation": explanation
        }
    
    def generate_recommendation(self, 
          coin: str, 
          market_data: Dict[str, Any],
          news_data: Dict[str, Any],
          market_context: Dict[str, Any],
          pattern_data: Dict[str, Any] = None,
          action_type: str = "spot",
          risk_tolerance: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Prepare data for the prompt
            current_price = market_data.get('current_price', 'Unknown')
            daily_change = market_data.get('daily_change_pct', 'Unknown')
            
            # Get indicators from market data
            rsi_1d = market_data.get('indicators', {}).get('1d', {}).get('rsi', 'Unknown')
            
            # Format pattern recognition data
            pattern_analysis = self._format_patterns(pattern_data) if pattern_data else "No pattern data available"
            
            # Get current UTC time and determine if it's a weekend
            now_utc = datetime.now(timezone.utc)
            weekday = now_utc.weekday()  # 0-4 are Monday to Friday, 5-6 are weekend
            is_weekend = weekday >= 5  # 5 = Saturday, 6 = Sunday
            
            # Current hour and minute in UTC for comparison
            current_hour_utc = now_utc.hour
            current_minute_utc = now_utc.minute
            current_time_decimal = current_hour_utc + (current_minute_utc / 60)
            
            # Market hours in UTC (regardless of weekend)
            us_market_start = 14.5  # 14:30 UTC (2:30 PM UTC)
            us_market_end = 4.0    # 21:00 UTC (9:00 PM UTC)
            
            eu_market_start = 8.0   # 8:00 UTC (8:00 AM UTC)
            eu_market_end = 22.5    # 16:30 UTC (4:30 PM UTC)
            
            china_market_start = 1.5  # 1:30 UTC (1:30 AM UTC)
            china_market_end = 11.5    # 7:30 UTC (7:30 AM UTC)
            
            # Determine which market hours we're in (regardless of weekend)
            in_us_market_hours = us_market_start <= current_time_decimal < us_market_end
            in_eu_market_hours = eu_market_start <= current_time_decimal < eu_market_end
            in_china_market_hours = china_market_start <= current_time_decimal < china_market_end
            
            # Create market hours context
            market_hours_context = {
                "current_utc_time": now_utc.strftime("%H:%M UTC %d-%b-%Y"),
                "is_weekend": is_weekend,
                "current_market_timezone": []
            }
            
            if in_us_market_hours:
                market_hours_context["current_market_timezone"].append("US")
            if in_eu_market_hours:
                market_hours_context["current_market_timezone"].append("European")
            if in_china_market_hours:
                market_hours_context["current_market_timezone"].append("Asian")
                
            if not market_hours_context["current_market_timezone"]:
                market_hours_context["market_timezone_description"] = "Outside of major market hours"
            else:
                market_hours_context["market_timezone_description"] = f"During {', '.join(market_hours_context['current_market_timezone'])} market hours"
            
            # Create description of market conditions based on both timezone and weekend status
            if is_weekend:
                market_hours_context["market_conditions"] = "It's currently the weekend. While crypto markets operate 24/7, weekend trading typically has lower volume and potentially higher volatility, as traditional financial markets are closed and institutional participation is reduced."
            else:
                if in_us_market_hours:
                    market_hours_context["market_conditions"] = "During US market hours. When US traditional markets are open on weekdays, crypto markets typically see highest volume and liquidity as US institutional investors are active."
                elif in_eu_market_hours:
                    market_hours_context["market_conditions"] = "During European market hours. When European traditional markets are open on weekdays, crypto markets typically see moderate to high volume and liquidity."
                elif in_china_market_hours:
                    market_hours_context["market_conditions"] = "During Asian market hours. When Asian traditional markets are open on weekdays, crypto markets typically see variable volume with potentially strong movements for Asian-focused projects."
                else:
                    market_hours_context["market_conditions"] = "Outside major traditional market hours. While crypto markets operate 24/7, this period typically sees lower volume and potentially wider spreads."
            
            # Build the prompt with enhanced ICT OTE guidance
            system_prompt = """You are a cryptocurrency trading advisor specialized in providing recommendations based on technical analysis, news sentiment, and market conditions.
Your task is to analyze the provided data and give a clear recommendation for the specified cryptocurrency.

IMPORTANT: 
1. Always include the current price in your recommendation near the beginning of your analysis.
2. ALWAYS mention the specific technical patterns identified (support/resistance levels, chart patterns, trend direction)
3. If support and resistance levels are provided, ALWAYS include them in your recommendation
4. The primary timeframe is 1-hour (1h) - focus on this timeframe for your main analysis
5. Shorter timeframes (30m, 15m, 5m) are useful for immediate entry/exit points
6. Longer timeframes (4h, 1d) provide context for the overall trend direction

ICT OPTIMAL TRADE ENTRY (OTE) STRATEGY:
- Pay special attention to ICT OTE setups when present - these are high-probability trade setups
- OTE setups identify precise entry points at the 62-79% Fibonacci retracement level
- For BULLISH OTE: Price breaks above previous day's high, then retraces to 62-79% Fib zone
- For BEARISH OTE: Price breaks below previous day's low, then retraces to 62-79% Fib zone
- If an OTE setup is identified as "IN OTE ZONE: YES", it's a prime entry opportunity

OTE TRADE MANAGEMENT:
- Entry should be near the 62% Fibonacci retracement level
- Stop loss should be placed just below/above the lowest/highest candle of the retracement
- Take profit targets should be at Fibonacci extensions: -0.5, -1.0, and -2.0
- After first profit target: Move stop to breakeven
- After second profit target: Trail stop below/above recent structure
- Always maintain a small position for the final target

TIME-BASED CONSIDERATIONS:
- Crypto markets trade 24/7, but volume and liquidity vary by time of day:
  * US market hours (9:30 AM - 4:00 PM ET / 14:30 - 21:00 UTC): Usually highest volume
  * European market hours (8:00 AM - 4:30 PM CET / 7:00 - 15:30 UTC): Moderate to high volume
  * Asian market hours (9:00 AM - 3:00 PM JST / 0:00 - 6:00 UTC): Variable volume
- IMPORTANT: Identify which specific market hours are active at the time of analysis
- SPECIFICALLY state whether we are currently in US, European, or Asian market hours
- Note how the current market hours may affect trading conditions for this specific recommendation
- Weekend trading tends to have lower volume and potentially higher volatility
- For geopolitical analysis:
  * ONLY include very recent (0-48 hours old) events that are still developing
  * Focus on breaking news and fresh developments that haven't been fully priced in
  * Explicitly state if there are no significant fresh geopolitical developments
  * Older news (3+ days) has likely already been priced into the market and should be excluded
- For upcoming events analysis:
  * Prioritize events within the next 72 hours which will have immediate impact
  * Include relevant events up to 7 days out that may influence trading decisions
  * Focus on US Federal Reserve announcements, SEC decisions, major economic data releases
  * Consider proximity to these events when making risk management recommendations
- Adjust trading recommendations based on proximity to these events (e.g., lower position sizes, wider stops)

LEVERAGE RECOMMENDATION:
- ALWAYS include a specific leverage recommendation when the user asks about leverage or mentions risk tolerance
- Format it as "Recommended Leverage: X-Y×" on its own line for clear visibility
- Base the leverage recommendation on the following factors:
  * Risk tolerance: 
    - Low risk: 1-2× leverage
    - Medium risk: 2-5× leverage
    - High risk: 5-10× leverage
  * Adjust these ranges down for:
    - High market volatility
    - Bearish or uncertain market conditions
    - Proximity to major support/resistance levels
    - Low confidence in the trade direction
    - Weekend trading or low liquidity periods
- Explain the reasoning behind your leverage recommendation in a separate paragraph

Your recommendation should consider:
1. Technical indicators from the 1-hour timeframe (RSI, MACD, etc.)
2. Technical chart patterns from the 1-hour timeframe
3. Support/resistance levels from the 1-hour timeframe (PRIMARY FOCUS)
4. ICT OTE setups from the 5-minute timeframe (HIGH PRIORITY WHEN PRESENT)
5. Recent news sentiment
6. Overall market context
7. Current market hours and upcoming significant events

For each recommendation:
- Provide a clear BUY, SELL, or HOLD recommendation
- Include a confidence level (Low, Medium, High)
- Explicitly mention the current price in your analysis
- Explain your reasoning in a concise manner
- Mention key factors influencing your decision
- If an OTE setup is identified, provide specific entry, stop-loss, and take-profit levels
- Include relevant risk warnings
- Reference important support/resistance levels from the 1-hour timeframe
- Comment on any relevant time-based factors (weekends, market hours, upcoming events)
- If futures trading or leverage is mentioned:
  * Provide a specific leverage recommendation (1x-125x) based on:
    - User's risk tolerance (low: 1-2x, medium: 3-5x, high: 5-20x)
    - Market volatility (higher volatility = lower leverage)
    - Current trend strength and confidence level
    - Support/resistance proximity
  * Include clear risk warnings about liquidation
  * Explain the specific reasoning for the leverage recommendation

LATEST NEWS SECTION:
Always include a dedicated "LATEST NEWS" section at the beginning of your response, with the most recent 2-3 headlines or significant developments about the specific cryptocurrency being analyzed. Format this as "As of {datetime.now().strftime('%Y-%m-%d')}, here's the latest news about [coin]:" followed by bullet points of recent developments. Focus on news from the past week that could impact price movement.

Also leverage your direct access to Twitter data to include any relevant social sentiment around this cryptocurrency.
"""
    
            user_prompt = f"""Please analyze the following data and provide a trading recommendation for {coin}:

MARKET DATA:
- Current price: ${current_price} USD
- 24h change: {daily_change}%
- RSI (1 day): {rsi_1d}
{self._format_indicators(market_data.get('indicators', {}))}

TECHNICAL PATTERNS:
{pattern_analysis}

GEOPOLITICAL AND MARKET-WIDE CONTEXT:
- ONLY include very recent (past 24-48 hours) geopolitical events that have NOT YET been fully priced into the market
- Focus on DEVELOPING situations, NEW announcements, or BREAKING news that could affect crypto in the coming days
- Ignore older events (3+ days) that have already impacted the market, as these are likely already priced in
- If there are no significant fresh geopolitical developments in the past 48 hours, clearly state this

UPCOMING MARKET-WIDE EVENTS:
- Identify imminent events (next 1-7 days) that could impact ALL cryptocurrencies (not just {coin})
- Focus on SEC decisions, presidential actions, Federal Reserve announcements, congressional hearings
- Include any upcoming elections, regulatory deadlines, or major economic data releases
- Prioritize events that are scheduled within the next 72 hours as these will have the most immediate impact

CURRENT MARKET HOURS:
- Current time: {market_hours_context['current_utc_time']}
- {"Weekend: Yes" if is_weekend else "Weekday: Yes"}
- Market timezone: {market_hours_context['market_timezone_description']}
- Market conditions: {market_hours_context['market_conditions']}
- Consider how these current market conditions might impact trading decisions for {coin}

USER PROFILE:
- Risk tolerance: {risk_tolerance if risk_tolerance else "not specified"}

Please provide a {action_type.upper()} trading recommendation (BUY/SELL/HOLD) with explanation.
Make sure to include the current price (${current_price} USD) in your analysis.
If an ICT OTE setup is present in the data, prioritize this in your recommendation.

IMPORTANT: Begin your response with a dedicated "LATEST NEWS" section showing the most recent developments for {coin}, using your access to real-time news sources and Twitter data. For geopolitical events, ONLY include genuinely fresh developments that haven't fully impacted markets yet.
Also include any relevant Twitter sentiment about {coin} that might impact the price.
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
grok_llm = GrokLLM()

# Helper function to get the singleton instance
def get_grok_llm():
    return grok_llm

# Example usage
if __name__ == "__main__":
    llm = get_grok_llm()
    # Test with a simple message
    response = llm.chat_completion([
        {"role": "user", "content": "What are the key factors to consider when trading cryptocurrencies?"}
    ])
    print(f"Response: {response}")