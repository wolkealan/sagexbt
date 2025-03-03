from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

from config.config import TradingConfig
from utils.logger import get_logger
from data.market_data import get_market_data_provider

logger = get_logger("ote_analyzer")

class OTEAnalyzer:
    """
    Analyzer for ICT Optimal Trade Entry (OTE) setups
    
    This class implements the ICT (Inner Circle Trader) methodology for identifying
    Optimal Trade Entry (OTE) setups on cryptocurrency charts. It focuses on finding
    high-probability entry points using Fibonacci retracement levels.
    """
    
    def __init__(self):
        self.market_data = get_market_data_provider()
        self.min_swing_pct = 0.5  # Minimum percentage swing required (0.5%)
        self.lookup_days = 3      # Number of days to look back for analysis
    
    def identify_ote_setup(self, symbol: str) -> Dict[str, Any]:
        """
        Identify potential ICT Optimal Trade Entry (OTE) setups
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Dictionary with OTE setup details
        """
        try:
            # Get data for different timeframes
            daily_df = self.market_data.get_historical_data(symbol, '1d', days=7)  # Get daily data for context
            df_5m = self.market_data.get_historical_data(symbol, '5m', days=self.lookup_days)  # 5-minute data for OTE identification
            df_1h = self.market_data.get_historical_data(symbol, '1h', days=self.lookup_days)  # 1-hour data for context
            
            if daily_df.empty or df_5m.empty:
                logger.warning(f"Insufficient data for OTE analysis for {symbol}")
                return {"error": "Insufficient data for OTE analysis"}
            
            # Extract previous day's high and low
            prev_day_high = daily_df['high'].iloc[-2]
            prev_day_low = daily_df['low'].iloc[-2]
            current_price = df_5m['close'].iloc[-1]
            
            # Initialize result dictionary
            ote_setup = {
                "symbol": symbol,
                "current_price": current_price,
                "prev_day_high": prev_day_high,
                "prev_day_low": prev_day_low,
                "bullish_setup": None,
                "bearish_setup": None,
                "timeframe": "5m",
                "has_valid_setup": False,  # Add a flag to make detection easier
                "setup_type": "none"       # Add setup type for easy reference
            }
            
            # Check for bullish setup
            bullish_breakout = self._check_bullish_breakout(df_5m, prev_day_high)
            if bullish_breakout:
                # Calculate Fibonacci levels for bullish setup
                swing_low, swing_high = self._find_recent_swing_points(df_5m, is_bullish=True)
                if swing_low is not None and swing_high is not None:
                    # Validate swing magnitude
                    swing_pct = (swing_high - swing_low) / swing_low * 100
                    if swing_pct >= self.min_swing_pct:
                        fib_levels = self._calculate_fibonacci_levels(swing_low, swing_high, is_bullish=True)
                        
                        # Check if price is in the OTE zone (62-79% retracement)
                        ote_zone_min = fib_levels["retracement_62"]
                        ote_zone_max = fib_levels["retracement_79"]
                        in_ote_zone = ote_zone_min <= current_price <= ote_zone_max
                        
                        # Find retracement low for stop loss calculation
                        retracement_low = self._find_retracement_low(df_5m, bullish_breakout['breakout_index'])
                        
                        # Calculate stop loss - below the lowest candle of the retracement or below swing low
                        stop_loss = min(retracement_low * 0.995, swing_low * 0.995)
                        
                        # Create bullish setup details
                        ote_setup["bullish_setup"] = {
                            "valid": True,
                            "in_ote_zone": in_ote_zone,
                            "swing_low": swing_low,
                            "swing_high": swing_high,
                            "retracement_low": retracement_low,
                            "breakout_index": bullish_breakout['breakout_index'],
                            "breakout_time": bullish_breakout['breakout_time'],
                            "fibonacci_levels": fib_levels,
                            "entry": fib_levels["retracement_62"],
                            "stop_loss": stop_loss,
                            "take_profit_1": fib_levels["extension_0_5"],
                            "take_profit_2": fib_levels["extension_1_0"],
                            "take_profit_3": min(fib_levels["extension_2_0"], current_price * 1.1),  # 10% or 2.0 extension
                            "risk_reward": self._calculate_risk_reward(
                                entry=fib_levels["retracement_62"],
                                stop=stop_loss,
                                target=fib_levels["extension_1_0"]
                            )
                        }
                        
                        # Update main ote_setup properties
                        ote_setup["has_valid_setup"] = True
                        ote_setup["setup_type"] = "bullish"
            
            # Check for bearish setup
            bearish_breakout = self._check_bearish_breakout(df_5m, prev_day_low)
            if bearish_breakout:
                # Calculate Fibonacci levels for bearish setup
                swing_high, swing_low = self._find_recent_swing_points(df_5m, is_bullish=False)
                if swing_high is not None and swing_low is not None:
                    # Validate swing magnitude
                    swing_pct = (swing_high - swing_low) / swing_low * 100
                    if swing_pct >= self.min_swing_pct:
                        fib_levels = self._calculate_fibonacci_levels(swing_high, swing_low, is_bullish=False)
                        
                        # Check if price is in the OTE zone (62-79% retracement)
                        ote_zone_min = fib_levels["retracement_79"]
                        ote_zone_max = fib_levels["retracement_62"]
                        in_ote_zone = ote_zone_min <= current_price <= ote_zone_max  # Reversed for bearish
                        
                        # Find retracement high for stop loss calculation
                        retracement_high = self._find_retracement_high(df_5m, bearish_breakout['breakout_index'])
                        
                        # Calculate stop loss - above the highest candle of the retracement or above swing high
                        stop_loss = max(retracement_high * 1.005, swing_high * 1.005)
                        
                        # Create bearish setup details
                        ote_setup["bearish_setup"] = {
                            "valid": True,
                            "in_ote_zone": in_ote_zone,
                            "swing_high": swing_high,
                            "swing_low": swing_low,
                            "retracement_high": retracement_high,
                            "breakout_index": bearish_breakout['breakout_index'],
                            "breakout_time": bearish_breakout['breakout_time'],
                            "fibonacci_levels": fib_levels,
                            "entry": fib_levels["retracement_62"],
                            "stop_loss": stop_loss,
                            "take_profit_1": fib_levels["extension_0_5"],
                            "take_profit_2": fib_levels["extension_1_0"],
                            "take_profit_3": max(fib_levels["extension_2_0"], current_price * 0.9),  # 10% or 2.0 extension
                            "risk_reward": self._calculate_risk_reward(
                                entry=fib_levels["retracement_62"],
                                stop=stop_loss,
                                target=fib_levels["extension_1_0"]
                            )
                        }
                        
                        # Update main ote_setup properties
                        ote_setup["has_valid_setup"] = True
                        ote_setup["setup_type"] = "bearish"
            
            # Even if we don't find a valid setup, add sensible default OTE values
            # for entry/exit points based on Fibonacci retracements from recent high/low
            if not ote_setup["has_valid_setup"]:
                self._add_generic_fib_levels(ote_setup, df_1h)
            
            logger.info(f"OTE analysis completed for {symbol}")
            return ote_setup
            
        except Exception as e:
            logger.error(f"Error in OTE analysis for {symbol}: {e}")
            return {"error": f"Error in OTE analysis: {str(e)}"}
    
    def _add_generic_fib_levels(self, ote_setup: Dict[str, Any], df: pd.DataFrame) -> None:
        """Add generic Fibonacci levels even when no valid OTE setup is found"""
        try:
            # Get high and low from recent price action
            high = df['high'].max()
            low = df['low'].min()
            current_price = df['close'].iloc[-1]
            
            # Determine if we're in an uptrend or downtrend
            ma_20 = df['close'].rolling(window=20).mean().iloc[-1]
            ma_50 = df['close'].rolling(window=50).mean().iloc[-1]
            
            # Crude trend determination
            is_uptrend = current_price > ma_20 > ma_50
            
            # Add potential entry points based on Fibonacci retracements
            if is_uptrend:
                # In uptrend, suggest buying on pullbacks
                retracement_50 = high - (high - low) * 0.5
                retracement_61 = high - (high - low) * 0.618
                retracement_78 = high - (high - low) * 0.786
                
                ote_setup["suggested_entry"] = retracement_61
                ote_setup["suggested_stop"] = low * 0.98  # 2% below low
                ote_setup["suggested_take_profit"] = high * 1.1  # 10% above high
                
                ote_setup["generic_levels"] = {
                    "trend": "uptrend",
                    "recent_high": high,
                    "recent_low": low,
                    "retracement_50": retracement_50,
                    "retracement_61": retracement_61,
                    "retracement_78": retracement_78,
                    "extension_1.0": high + (high - low),
                    "extension_1.618": high + (high - low) * 1.618
                }
            else:
                # In downtrend, suggest selling on pullbacks
                retracement_50 = low + (high - low) * 0.5
                retracement_61 = low + (high - low) * 0.618
                retracement_78 = low + (high - low) * 0.786
                
                ote_setup["suggested_entry"] = retracement_61
                ote_setup["suggested_stop"] = high * 1.02  # 2% above high
                ote_setup["suggested_take_profit"] = low * 0.9  # 10% below low
                
                ote_setup["generic_levels"] = {
                    "trend": "downtrend",
                    "recent_high": high,
                    "recent_low": low,
                    "retracement_50": retracement_50,
                    "retracement_61": retracement_61,
                    "retracement_78": retracement_78,
                    "extension_1.0": low - (high - low),
                    "extension_1.618": low - (high - low) * 1.618
                }
        except Exception as e:
            logger.warning(f"Error adding generic Fibonacci levels: {e}")
    
    def _check_bullish_breakout(self, df: pd.DataFrame, prev_day_high: float) -> Optional[Dict[str, Any]]:
        """
        Check if price has broken above the previous day's high
        
        Args:
            df: DataFrame with price data
            prev_day_high: Previous day's high price
            
        Returns:
            Dictionary with breakout details if breakout occurred, None otherwise
        """
        try:
            # Find the first candle that breaks above the previous day's high
            breakout_mask = df['high'] > prev_day_high
            
            if breakout_mask.any():
                # Get index of first breakout
                breakout_index = breakout_mask.idxmax()
                breakout_candle = df.loc[breakout_index]
                
                return {
                    "breakout_index": breakout_index,
                    "breakout_price": breakout_candle['high'],
                    "breakout_time": breakout_candle.name if hasattr(breakout_candle, 'name') else None
                }
            
            return None
        except Exception as e:
            logger.error(f"Error checking bullish breakout: {e}")
            return None
    
    def _check_bearish_breakout(self, df: pd.DataFrame, prev_day_low: float) -> Optional[Dict[str, Any]]:
        """
        Check if price has broken below the previous day's low
        
        Args:
            df: DataFrame with price data
            prev_day_low: Previous day's low price
            
        Returns:
            Dictionary with breakout details if breakout occurred, None otherwise
        """
        try:
            # Find the first candle that breaks below the previous day's low
            breakout_mask = df['low'] < prev_day_low
            
            if breakout_mask.any():
                # Get index of first breakout
                breakout_index = breakout_mask.idxmax()
                breakout_candle = df.loc[breakout_index]
                
                return {
                    "breakout_index": breakout_index,
                    "breakout_price": breakout_candle['low'],
                    "breakout_time": breakout_candle.name if hasattr(breakout_candle, 'name') else None
                }
            
            return None
        except Exception as e:
            logger.error(f"Error checking bearish breakout: {e}")
            return None
    
    def _find_recent_swing_points(self, df: pd.DataFrame, is_bullish: bool = True, window: int = 10) -> Tuple[float, float]:
        """
        Find recent swing points for Fibonacci retracement
        """
        try:
            # Use more recent data for swing point detection
            recent_df = df.tail(min(len(df), 100))  # Last 100 candles, adjust as needed
            
            if is_bullish:
                # For bullish setup, find recent swing low and subsequent high
                swing_low_idx = recent_df['low'].idxmin()
                swing_low = recent_df.loc[swing_low_idx, 'low']
                
                # Find high after the swing low
                subsequent_df = df.loc[swing_low_idx:]
                if len(subsequent_df) > 1:
                    swing_high_idx = subsequent_df['high'].idxmax()
                    swing_high = subsequent_df.loc[swing_high_idx, 'high']
                    return swing_low, swing_high
            else:
                # For bearish setup, find recent swing high and subsequent low
                swing_high_idx = recent_df['high'].idxmax()
                swing_high = recent_df.loc[swing_high_idx, 'high']
                
                # Find low after the swing high
                subsequent_df = df.loc[swing_high_idx:]
                if len(subsequent_df) > 1:
                    swing_low_idx = subsequent_df['low'].idxmin()
                    swing_low = subsequent_df.loc[swing_low_idx, 'low']
                    return swing_high, swing_low
        
        except Exception as e:
            logger.error(f"Error finding swing points: {e}")
        
        return None, None
    
    def _find_retracement_low(self, df: pd.DataFrame, breakout_index) -> float:
        """Find the lowest low after breakout for stop loss calculation"""
        try:
            # Get data after the breakout
            post_breakout_df = df.loc[breakout_index:]
            if not post_breakout_df.empty:
                return post_breakout_df['low'].min()
        except Exception as e:
            logger.error(f"Error finding retracement low: {e}")
        
        return float('inf')  # Return a high value if error
    
    def _find_retracement_high(self, df: pd.DataFrame, breakout_index) -> float:
        """Find the highest high after breakout for stop loss calculation"""
        try:
            # Get data after the breakout
            post_breakout_df = df.loc[breakout_index:]
            if not post_breakout_df.empty:
                return post_breakout_df['high'].max()
        except Exception as e:
            logger.error(f"Error finding retracement high: {e}")
        
        return 0  # Return a low value if error
    
    def _calculate_fibonacci_levels(self, start_price: float, end_price: float, is_bullish: bool = True) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement and extension levels
        """
        try:
            price_diff = abs(end_price - start_price)
            
            if is_bullish:
                # Bullish: from swing low to swing high
                retracement_50 = end_price - (price_diff * 0.50)
                retracement_62 = end_price - (price_diff * 0.618)
                retracement_79 = end_price - (price_diff * 0.786)
                
                # Extensions beyond the swing high
                extension_0_5 = end_price + (price_diff * 0.5)
                extension_1_0 = end_price + (price_diff * 1.0)
                extension_2_0 = end_price + (price_diff * 2.0)
            else:
                # Bearish: from swing high to swing low
                retracement_50 = end_price + (price_diff * 0.50)
                retracement_62 = end_price + (price_diff * 0.618)
                retracement_79 = end_price + (price_diff * 0.786)
                
                # Extensions beyond the swing low
                extension_0_5 = end_price - (price_diff * 0.5)
                extension_1_0 = end_price - (price_diff * 1.0)
                extension_2_0 = end_price - (price_diff * 2.0)
            
            return {
                "retracement_50": retracement_50,
                "retracement_62": retracement_62,
                "retracement_79": retracement_79,
                "extension_0_5": extension_0_5,
                "extension_1_0": extension_1_0,
                "extension_2_0": extension_2_0
            }
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return {}
    
    def _calculate_risk_reward(self, entry: float, stop: float, target: float) -> float:
        """
        Calculate risk to reward ratio
        """
        try:
            risk = abs(entry - stop)
            reward = abs(target - entry)
            
            if risk > 0:
                return round(reward / risk, 2)
            return 0
        except Exception as e:
            logger.error(f"Error calculating risk/reward: {e}")
            return 0

# Singleton instance
ote_analyzer = OTEAnalyzer()

# Helper function to get the singleton instance
def get_ote_analyzer():
    return ote_analyzer

# Example usage
if __name__ == "__main__":
    analyzer = get_ote_analyzer()
    # Test OTE analysis for BTC
    ote_setup = analyzer.identify_ote_setup("BTC")
    print(f"BTC OTE Setup: {ote_setup}")