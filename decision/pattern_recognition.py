import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from config.config import TradingConfig
from utils.logger import get_logger
from data.market_data import get_market_data_provider

logger = get_logger("pattern_recognition")

class PatternRecognition:
    """Identifies technical patterns in cryptocurrency price data"""
    
    def __init__(self):
        self.market_data = get_market_data_provider()
    
    def identify_patterns(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """
        Identify various technical patterns in the price data
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Time period for the analysis
            
        Returns:
            Dictionary with identified patterns and their details
        """
        try:
            # Get historical data
            df = self.market_data.get_historical_data(symbol, timeframe)
            if df.empty:
                logger.warning(f"No data available for pattern recognition for {symbol}")
                return {"error": "No data available"}
            
            # Find patterns
            patterns = {}
            
            # Check for trend patterns
            patterns["trend"] = self._identify_trend(df)
            
            # Check for support/resistance levels
            patterns["support_resistance"] = self._identify_support_resistance(df)
            
            # Check for candlestick patterns
            patterns["candlestick"] = self._identify_candlestick_patterns(df)
            
            # Check for chart patterns
            patterns["chart_patterns"] = self._identify_chart_patterns(df)
            
            logger.info(f"Identified {sum(len(v) for v in patterns.values() if isinstance(v, dict))} patterns for {symbol}")
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying patterns for {symbol}: {e}")
            return {"error": str(e)}
    
    def _identify_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify the current price trend"""
        try:
            # Calculate short and long term moving averages
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()
            df['ma_200'] = df['close'].rolling(window=200).mean()
            
            # Get the most recent values
            current_price = df['close'].iloc[-1]
            ma_20 = df['ma_20'].iloc[-1]
            ma_50 = df['ma_50'].iloc[-1]
            ma_200 = df['ma_200'].iloc[-1]
            
            # Calculate price changes
            price_change_1d = df['close'].pct_change(1).iloc[-1] * 100
            price_change_7d = df['close'].pct_change(7).iloc[-1] * 100
            price_change_30d = df['close'].pct_change(30).iloc[-1] * 100
            
            # Determine trend based on moving averages and price action
            trend = {
                "short_term": "neutral",
                "medium_term": "neutral",
                "long_term": "neutral",
                "strength": 0
            }
            
            # Short-term trend (based on 20-day MA)
            if current_price > ma_20 and price_change_7d > 0:
                trend["short_term"] = "bullish"
                trend["strength"] += 1
            elif current_price < ma_20 and price_change_7d < 0:
                trend["short_term"] = "bearish"
                trend["strength"] -= 1
            
            # Medium-term trend (based on 50-day MA)
            if current_price > ma_50 and ma_20 > ma_50:
                trend["medium_term"] = "bullish"
                trend["strength"] += 1
            elif current_price < ma_50 and ma_20 < ma_50:
                trend["medium_term"] = "bearish"
                trend["strength"] -= 1
            
            # Long-term trend (based on 200-day MA)
            if current_price > ma_200 and ma_50 > ma_200:
                trend["long_term"] = "bullish"
                trend["strength"] += 1
            elif current_price < ma_200 and ma_50 < ma_200:
                trend["long_term"] = "bearish"
                trend["strength"] -= 1
            
            # Additional trend data
            trend["current_price"] = current_price
            trend["ma_20"] = ma_20
            trend["ma_50"] = ma_50
            trend["ma_200"] = ma_200
            trend["price_change_1d"] = price_change_1d
            trend["price_change_7d"] = price_change_7d
            trend["price_change_30d"] = price_change_30d
            
            # Special cases - Golden Cross and Death Cross
            if ma_50 > ma_200 and df['ma_50'].shift(1).iloc[-1] <= df['ma_200'].shift(1).iloc[-1]:
                trend["special_event"] = "Golden Cross"
                trend["strength"] += 2
            elif ma_50 < ma_200 and df['ma_50'].shift(1).iloc[-1] >= df['ma_200'].shift(1).iloc[-1]:
                trend["special_event"] = "Death Cross"
                trend["strength"] -= 2
            
            # Overall trend determination
            if trend["strength"] >= 2:
                trend["overall"] = "strong_bullish"
            elif trend["strength"] == 1:
                trend["overall"] = "bullish"
            elif trend["strength"] == 0:
                trend["overall"] = "neutral"
            elif trend["strength"] == -1:
                trend["overall"] = "bearish"
            else:
                trend["overall"] = "strong_bearish"
            
            return trend
            
        except Exception as e:
            logger.error(f"Error identifying trend: {e}")
            return {"error": str(e)}
    
    def _identify_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify support and resistance levels"""
        try:
            # Get price data
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            current_price = closes[-1]
            
            # Function to find local extrema
            def find_extrema(data, window=5):
                extrema = []
                for i in range(window, len(data) - window):
                    if all(data[i] >= data[i-window:i]) and all(data[i] >= data[i+1:i+window+1]):
                        extrema.append((i, data[i]))
                    elif all(data[i] <= data[i-window:i]) and all(data[i] <= data[i+1:i+window+1]):
                        extrema.append((i, data[i]))
                return extrema
            
            # Find extrema in high and low values
            high_extrema = find_extrema(highs)
            low_extrema = find_extrema(lows)
            
            # Group nearby levels
            def group_levels(extrema, threshold=0.02):
                if not extrema:
                    return []
                
                # Sort by price level
                sorted_extrema = sorted(extrema, key=lambda x: x[1])
                grouped = []
                current_group = [sorted_extrema[0]]
                
                for i in range(1, len(sorted_extrema)):
                    if sorted_extrema[i][1] / current_group[0][1] - 1 < threshold:
                        current_group.append(sorted_extrema[i])
                    else:
                        # Calculate average price for the group
                        avg_price = sum(level[1] for level in current_group) / len(current_group)
                        grouped.append((len(current_group), avg_price))
                        current_group = [sorted_extrema[i]]
                
                # Add the last group
                if current_group:
                    avg_price = sum(level[1] for level in current_group) / len(current_group)
                    grouped.append((len(current_group), avg_price))
                
                return grouped
            
            # Group the levels
            support_levels = group_levels([x for x in low_extrema if x[1] < current_price])
            resistance_levels = group_levels([x for x in high_extrema if x[1] > current_price])
            
            # Sort by strength (number of touches) and proximity to current price
            support_levels.sort(key=lambda x: (-x[0], abs(current_price - x[1])))
            resistance_levels.sort(key=lambda x: (-x[0], abs(current_price - x[1])))
            
            # Format the result
            result = {
                "support": [{"level": level, "strength": count} for count, level in support_levels[:3]],
                "resistance": [{"level": level, "strength": count} for count, level in resistance_levels[:3]],
                "current_price": current_price
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error identifying support/resistance: {e}")
            return {"error": str(e)}
    
    def _identify_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify common candlestick patterns"""
        try:
            # Get the most recent candlesticks
            recent_df = df.tail(5).copy()
            if len(recent_df) < 5:
                return {"error": "Not enough data for candlestick analysis"}
            
            patterns = {}
            
            # Calculate candle properties
            recent_df['body_size'] = abs(recent_df['close'] - recent_df['open'])
            recent_df['upper_shadow'] = recent_df['high'] - recent_df[['open', 'close']].max(axis=1)
            recent_df['lower_shadow'] = recent_df[['open', 'close']].min(axis=1) - recent_df['low']
            recent_df['bullish'] = recent_df['close'] > recent_df['open']
            
            # Get the most recent 3 candles
            c1 = recent_df.iloc[-3]  # Two days ago
            c2 = recent_df.iloc[-2]  # Yesterday
            c3 = recent_df.iloc[-1]  # Today
            
            # Simple average for body size reference
            avg_body = recent_df['body_size'].mean()
            
            # Check for doji (very small body)
            if c3['body_size'] < 0.1 * avg_body:
                if c3['upper_shadow'] > 2 * c3['body_size'] and c3['lower_shadow'] > 2 * c3['body_size']:
                    patterns["doji"] = {
                        "type": "doji",
                        "significance": "neutral",
                        "description": "Small body with upper and lower shadows, indicating indecision"
                    }
            
            # Hammer pattern (bullish reversal)
            if c3['lower_shadow'] > 2 * c3['body_size'] and c3['upper_shadow'] < 0.2 * c3['body_size']:
                if not c3['bullish']:  # Traditional hammer
                    patterns["hammer"] = {
                        "type": "hammer",
                        "significance": "bullish",
                        "description": "Bearish candle with long lower shadow, indicating potential reversal"
                    }
                else:  # Inverted hammer
                    patterns["inverted_hammer"] = {
                        "type": "inverted_hammer",
                        "significance": "bullish",
                        "description": "Bullish candle with long lower shadow, suggesting buying pressure"
                    }
            
            # Shooting star (bearish reversal)
            if c3['upper_shadow'] > 2 * c3['body_size'] and c3['lower_shadow'] < 0.2 * c3['body_size']:
                if c3['bullish']:
                    patterns["shooting_star"] = {
                        "type": "shooting_star",
                        "significance": "bearish",
                        "description": "Bullish candle with long upper shadow, indicating potential reversal"
                    }
            
            # Engulfing patterns
            if c3['bullish'] and not c2['bullish']:  # Today bullish, yesterday bearish
                if c3['open'] < c2['close'] and c3['close'] > c2['open']:
                    patterns["bullish_engulfing"] = {
                        "type": "bullish_engulfing",
                        "significance": "bullish",
                        "description": "Bullish candle completely engulfs previous bearish candle, strong reversal signal"
                    }
            elif not c3['bullish'] and c2['bullish']:  # Today bearish, yesterday bullish
                if c3['open'] > c2['close'] and c3['close'] < c2['open']:
                    patterns["bearish_engulfing"] = {
                        "type": "bearish_engulfing",
                        "significance": "bearish",
                        "description": "Bearish candle completely engulfs previous bullish candle, strong reversal signal"
                    }
            
            # Morning star (bullish reversal)
            if not c1['bullish'] and c3['bullish']:  # First bearish, last bullish
                if c2['body_size'] < 0.3 * avg_body:  # Middle small body
                    if c1['close'] > c2['open'] and c2['close'] < c3['open'] and c3['close'] > c1['close']:
                        patterns["morning_star"] = {
                            "type": "morning_star",
                            "significance": "bullish",
                            "description": "Three-candle pattern showing a potential trend reversal from bearish to bullish"
                        }
            
            # Evening star (bearish reversal)
            if c1['bullish'] and not c3['bullish']:  # First bullish, last bearish
                if c2['body_size'] < 0.3 * avg_body:  # Middle small body
                    if c1['close'] < c2['open'] and c2['close'] > c3['open'] and c3['close'] < c1['close']:
                        patterns["evening_star"] = {
                            "type": "evening_star",
                            "significance": "bearish",
                            "description": "Three-candle pattern showing a potential trend reversal from bullish to bearish"
                        }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying candlestick patterns: {e}")
            return {"error": str(e)}
    
    def _identify_chart_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify chart patterns such as head and shoulders, double tops, etc."""
        try:
            # Need more data for reliable chart pattern detection
            if len(df) < 30:
                return {"error": "Not enough data for chart pattern analysis"}
            
            patterns = {}
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Double Top pattern
            double_top = self._check_double_top(highs, closes)
            if double_top:
                patterns["double_top"] = {
                    "type": "double_top",
                    "significance": "bearish",
                    "level": double_top,
                    "description": "Price reaches a high, pulls back, then reaches a similar high before declining"
                }
            
            # Double Bottom pattern
            double_bottom = self._check_double_bottom(lows, closes)
            if double_bottom:
                patterns["double_bottom"] = {
                    "type": "double_bottom",
                    "significance": "bullish",
                    "level": double_bottom,
                    "description": "Price reaches a low, rebounds, then reaches a similar low before rising"
                }
            
            # Head and Shoulders pattern (more complex, simplified version here)
            head_shoulders = self._check_head_and_shoulders(highs, lows, closes)
            if head_shoulders:
                patterns["head_and_shoulders"] = {
                    "type": "head_and_shoulders",
                    "significance": "bearish",
                    "neckline": head_shoulders,
                    "description": "Three peaks with the middle one highest, indicating potential trend reversal"
                }
            
            # Inverse Head and Shoulders pattern
            inv_head_shoulders = self._check_inverse_head_and_shoulders(highs, lows, closes)
            if inv_head_shoulders:
                patterns["inverse_head_and_shoulders"] = {
                    "type": "inverse_head_and_shoulders",
                    "significance": "bullish",
                    "neckline": inv_head_shoulders,
                    "description": "Three troughs with the middle one lowest, indicating potential trend reversal"
                }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying chart patterns: {e}")
            return {"error": str(e)}
    
    def _check_double_top(self, highs: np.ndarray, closes: np.ndarray) -> Optional[float]:
        """Check for double top pattern"""
        try:
            # Look for two peaks with similar heights
            window = 5
            threshold = 0.03  # 3% difference allowed between peaks
            
            # Find peaks
            peaks = []
            for i in range(window, len(highs) - window):
                if all(highs[i] >= highs[i-window:i]) and all(highs[i] >= highs[i+1:i+window+1]):
                    peaks.append((i, highs[i]))
            
            # Need at least 2 peaks
            if len(peaks) < 2:
                return None
            
            # Check the last two peaks
            if len(peaks) >= 2:
                peak1_idx, peak1_val = peaks[-2]
                peak2_idx, peak2_val = peaks[-1]
                
                # Check if peaks are similar in height
                if abs(peak1_val / peak2_val - 1) < threshold:
                    # Check if there's a valley between them
                    valley_idx = np.argmin(closes[peak1_idx:peak2_idx]) + peak1_idx
                    valley_val = closes[valley_idx]
                    
                    # Valley should be significantly lower than peaks
                    if valley_val < 0.9 * min(peak1_val, peak2_val):
                        # Check if current price is below the valley
                        if closes[-1] < valley_val:
                            return float(peak1_val)  # Return the resistance level
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking for double top: {e}")
            return None
    
    def _check_double_bottom(self, lows: np.ndarray, closes: np.ndarray) -> Optional[float]:
        """Check for double bottom pattern"""
        try:
            # Look for two troughs with similar depths
            window = 5
            threshold = 0.03  # 3% difference allowed between troughs
            
            # Find troughs
            troughs = []
            for i in range(window, len(lows) - window):
                if all(lows[i] <= lows[i-window:i]) and all(lows[i] <= lows[i+1:i+window+1]):
                    troughs.append((i, lows[i]))
            
            # Need at least 2 troughs
            if len(troughs) < 2:
                return None
            
            # Check the last two troughs
            if len(troughs) >= 2:
                trough1_idx, trough1_val = troughs[-2]
                trough2_idx, trough2_val = troughs[-1]
                
                # Check if troughs are similar in depth
                if abs(trough1_val / trough2_val - 1) < threshold:
                    # Check if there's a peak between them
                    peak_idx = np.argmax(closes[trough1_idx:trough2_idx]) + trough1_idx
                    peak_val = closes[peak_idx]
                    
                    # Peak should be significantly higher than troughs
                    if peak_val > 1.1 * max(trough1_val, trough2_val):
                        # Check if current price is above the peak
                        if closes[-1] > peak_val:
                            return float(trough1_val)  # Return the support level
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking for double bottom: {e}")
            return None
    
    def _check_head_and_shoulders(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Optional[float]:
        """Simplified check for head and shoulders pattern"""
        try:
            # Simplified implementation - in a real system this would be more sophisticated
            window = 5
            
            # Find peaks
            peaks = []
            for i in range(window, len(highs) - window):
                if all(highs[i] >= highs[i-window:i]) and all(highs[i] >= highs[i+1:i+window+1]):
                    peaks.append((i, highs[i]))
            
            # Need at least 3 peaks
            if len(peaks) < 3:
                return None
            
            # Check the last three peaks
            if len(peaks) >= 3:
                # Get the last three peaks
                left_shoulder_idx, left_shoulder_val = peaks[-3]
                head_idx, head_val = peaks[-2]
                right_shoulder_idx, right_shoulder_val = peaks[-1]
                
                # Head should be higher than shoulders
                if head_val > left_shoulder_val and head_val > right_shoulder_val:
                    # Shoulders should be at similar heights
                    if abs(left_shoulder_val / right_shoulder_val - 1) < 0.1:
                        # Calculate neckline (support level connecting troughs between shoulders)
                        left_trough_idx = np.argmin(lows[left_shoulder_idx:head_idx]) + left_shoulder_idx
                        right_trough_idx = np.argmin(lows[head_idx:right_shoulder_idx]) + head_idx
                        
                        left_trough_val = lows[left_trough_idx]
                        right_trough_val = lows[right_trough_idx]
                        
                        # Calculate neckline level (simplified as average)
                        neckline = (left_trough_val + right_trough_val) / 2
                        
                        # Check if price has broken below the neckline
                        if closes[-1] < neckline:
                            return float(neckline)
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking for head and shoulders: {e}")
            return None
    
    def _check_inverse_head_and_shoulders(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Optional[float]:
        """Simplified check for inverse head and shoulders pattern"""
        try:
            # Simplified implementation - in a real system this would be more sophisticated
            window = 5
            
            # Find troughs
            troughs = []
            for i in range(window, len(lows) - window):
                if all(lows[i] <= lows[i-window:i]) and all(lows[i] <= lows[i+1:i+window+1]):
                    troughs.append((i, lows[i]))
            
            # Need at least 3 troughs
            if len(troughs) < 3:
                return None
            
            # Check the last three troughs
            if len(troughs) >= 3:
                # Get the last three troughs
                left_shoulder_idx, left_shoulder_val = troughs[-3]
                head_idx, head_val = troughs[-2]
                right_shoulder_idx, right_shoulder_val = troughs[-1]
                
                # Head should be lower than shoulders
                if head_val < left_shoulder_val and head_val < right_shoulder_val:
                    # Shoulders should be at similar depths
                    if abs(left_shoulder_val / right_shoulder_val - 1) < 0.1:
                        # Calculate neckline (resistance level connecting peaks between shoulders)
                        left_peak_idx = np.argmax(highs[left_shoulder_idx:head_idx]) + left_shoulder_idx
                        right_peak_idx = np.argmax(highs[head_idx:right_shoulder_idx]) + head_idx
                        
                        left_peak_val = highs[left_peak_idx]
                        right_peak_val = highs[right_peak_idx]
                        
                        # Calculate neckline level (simplified as average)
                        neckline = (left_peak_val + right_peak_val) / 2
                        
                        # Check if price has broken above the neckline
                        if closes[-1] > neckline:
                            return float(neckline)
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking for inverse head and shoulders: {e}")
            return None

# Singleton instance
pattern_recognition = PatternRecognition()

# Helper function to get the singleton instance
def get_pattern_recognition():
    return pattern_recognition

# Example usage
if __name__ == "__main__":
    recognizer = get_pattern_recognition()
    # Identify patterns for Bitcoin
    patterns = recognizer.identify_patterns("BTC", "1d")
    print(f"BTC Patterns: {patterns}")