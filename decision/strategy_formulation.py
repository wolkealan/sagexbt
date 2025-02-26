import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from config.config import TradingConfig
from utils.logger import get_logger
from data.market_data import get_market_data_provider
from data.news_provider import get_news_provider
from decision.pattern_recognition import get_pattern_recognition
from decision.risk_assessment import get_risk_assessment

logger = get_logger("strategy_formulation")

class StrategyFormulation:
    """Formulates trading strategies based on market data, patterns, and risk assessment"""
    
    def __init__(self):
        self.market_data = get_market_data_provider()
        self.news_provider = get_news_provider()
        self.pattern_recognition = get_pattern_recognition()
        self.risk_assessment = get_risk_assessment()
    
    def formulate_strategy(self, symbol: str, action_type: str = "spot", 
                         risk_tolerance: str = "medium") -> Dict[str, Any]:
        """
        Formulate a trading strategy for a cryptocurrency
        
        Args:
            symbol: Cryptocurrency symbol
            action_type: 'spot' or 'futures' trading
            risk_tolerance: User's risk tolerance level ('low', 'medium', 'high')
            
        Returns:
            Trading strategy with entry/exit points and risk management
        """
        try:
            # Get market data
            market_summary = self.market_data.get_market_summary(symbol)
            current_price = market_summary.get("current_price", 0)
            
            # Get technical patterns
            patterns = self.pattern_recognition.identify_patterns(symbol)
            
            # Get risk assessment
            risk_data = self.risk_assessment.assess_risk(symbol, action_type, risk_tolerance)
            
            # Generate entry strategy
            entry_strategy = self._generate_entry_strategy(symbol, current_price, patterns, risk_data)
            
            # Generate exit strategy
            exit_strategy = self._generate_exit_strategy(symbol, current_price, patterns, risk_data, entry_strategy)
            
            # Generate position sizing recommendation
            position_sizing = self._generate_position_sizing(symbol, risk_data)
            
            # Generate time horizon recommendation
            time_horizon = self._generate_time_horizon(symbol, patterns, risk_data)
            
            # Create complete strategy
            strategy = {
                "symbol": symbol,
                "action_type": action_type,
                "risk_tolerance": risk_tolerance,
                "current_price": current_price,
                "entry_strategy": entry_strategy,
                "exit_strategy": exit_strategy,
                "position_sizing": position_sizing,
                "time_horizon": time_horizon,
                "risk_level": risk_data.get("risk_level", "medium"),
                "timestamp": datetime.now()
            }
            
            logger.info(f"Generated trading strategy for {symbol}")
            return strategy
        
        except Exception as e:
            logger.error(f"Error formulating strategy for {symbol}: {e}")
            return {
                "symbol": symbol,
                "action_type": action_type,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    def _generate_entry_strategy(self, symbol: str, current_price: float,
                              patterns: Dict[str, Any], risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate entry strategy based on technical patterns and risk assessment"""
        try:
            entry_strategy = {"strategy": "market", "reasoning": []}
            
            # Default to market entry
            entry_price = current_price
            entry_type = "market"
            reasoning = []
            
            # Check trend direction
            trend = patterns.get("trend", {})
            trend_overall = trend.get("overall", "neutral")
            
            # Check support/resistance levels
            support_resistance = patterns.get("support_resistance", {})
            support_levels = support_resistance.get("support", [])
            resistance_levels = support_resistance.get("resistance", [])
            
            # Determine if we should wait for a better entry
            risk_level = risk_data.get("risk_level", "medium")
            
            if trend_overall in ["strong_bearish", "bearish"]:
                # In a bearish trend, consider waiting for support
                if support_levels and len(support_levels) > 0:
                    # Use the strongest support level
                    strongest_support = support_levels[0].get("level", current_price * 0.9)
                    
                    # If current price is significantly above support, suggest limit order
                    if current_price > strongest_support * 1.05:
                        entry_price = strongest_support
                        entry_type = "limit"
                        reasoning.append(f"Bearish trend detected. Consider waiting for price to reach support at {strongest_support:.2f}")
                else:
                    # No clear support, suggest scaling in
                    entry_type = "scaled"
                    reasoning.append("Bearish trend with no clear support. Consider scaling in to reduce timing risk")
            
            elif trend_overall in ["strong_bullish", "bullish"]:
                # In a bullish trend, check if price is at resistance
                if resistance_levels and len(resistance_levels) > 0:
                    strongest_resistance = resistance_levels[0].get("level", current_price * 1.1)
                    
                    # If price is approaching resistance, wait for breakout confirmation
                    if current_price > strongest_resistance * 0.95:
                        entry_price = strongest_resistance * 1.02  # Slight breakout
                        entry_type = "stop"
                        reasoning.append(f"Bullish trend with price near resistance. Consider entering on breakout above {strongest_resistance:.2f}")
                    else:
                        # Not near resistance, can enter at market
                        reasoning.append("Bullish trend with price below resistance. Market entry acceptable")
                else:
                    reasoning.append("Bullish trend detected. Market entry acceptable")
            
            else:  # Neutral trend
                # In neutral trend, look for other signals
                candlestick_patterns = patterns.get("candlestick", {})
                if candlestick_patterns:
                    # Check if there are bullish patterns
                    bullish_patterns = [p for p, data in candlestick_patterns.items() 
                                       if data.get("significance") == "bullish"]
                    bearish_patterns = [p for p, data in candlestick_patterns.items() 
                                       if data.get("significance") == "bearish"]
                    
                    if bullish_patterns:
                        reasoning.append(f"Bullish candlestick patterns detected ({', '.join(bullish_patterns)}). Market entry acceptable")
                    elif bearish_patterns:
                        if support_levels and len(support_levels) > 0:
                            entry_price = support_levels[0].get("level", current_price * 0.9)
                            entry_type = "limit"
                            reasoning.append(f"Bearish candlestick patterns detected. Consider waiting for price to reach support at {entry_price:.2f}")
                        else:
                            reasoning.append("Bearish candlestick patterns with no clear support. Consider deferring entry")
                    else:
                        reasoning.append("No clear directional candlestick patterns. Market entry acceptable")
                else:
                    reasoning.append("Neutral trend with no clear signals. Consider scaling in or waiting for clearer signals")
                    entry_type = "scaled"
            
            # Adjust based on risk level
            if risk_level in ["high", "very_high"] and entry_type == "market":
                # For high risk, prefer limit orders or scaling in
                if support_levels and len(support_levels) > 0:
                    entry_price = support_levels[0].get("level", current_price * 0.95)
                    entry_type = "limit"
                    reasoning.append(f"High risk detected. Consider using limit order at {entry_price:.2f} instead of market entry")
                else:
                    entry_type = "scaled"
                    reasoning.append("High risk detected. Consider scaling in over time instead of market entry")
            
            # Build the entry strategy
            strategy = {
                "type": entry_type,
                "price": entry_price if entry_type != "scaled" else None,
                "reasoning": reasoning
            }
            
            if entry_type == "scaled":
                # Define scaling strategy
                strategy["scale_levels"] = [
                    {"price": current_price, "percentage": 0.3},
                    {"price": current_price * 0.95, "percentage": 0.3},
                    {"price": current_price * 0.9, "percentage": 0.4}
                ]
            
            return strategy
        
        except Exception as e:
            logger.error(f"Error generating entry strategy for {symbol}: {e}")
            return {"type": "market", "price": current_price, "reasoning": [f"Error: {str(e)}"]}
    
    def _generate_exit_strategy(self, symbol: str, current_price: float,
                              patterns: Dict[str, Any], risk_data: Dict[str, Any],
                              entry_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Generate exit strategy with profit targets and stop-loss levels"""
        try:
            # Get entry price
            entry_type = entry_strategy.get("type", "market")
            entry_price = entry_strategy.get("price", current_price)
            
            if entry_type == "scaled":
                # For scaled entries, use current price as reference
                entry_price = current_price
            
            # Default exit strategy
            exit_strategy = {
                "take_profit": [],
                "stop_loss": None,
                "trailing_stop": None,
                "time_stop": None,
                "reasoning": []
            }
            
            # Get risk level
            risk_level = risk_data.get("risk_level", "medium")
            
            # Check support/resistance levels
            support_resistance = patterns.get("support_resistance", {})
            support_levels = support_resistance.get("support", [])
            resistance_levels = support_resistance.get("resistance", [])
            
            # Set stop loss based on risk level and support levels
            if support_levels and len(support_levels) > 0:
                # Use nearest support level below entry price
                valid_supports = [s.get("level") for s in support_levels if s.get("level") < entry_price]
                if valid_supports:
                    stop_loss = max(valid_supports)
                    exit_strategy["stop_loss"] = stop_loss
                    exit_strategy["reasoning"].append(f"Stop loss set at nearest support level: {stop_loss:.2f}")
                else:
                    # No valid support below, use percentage-based stop loss
                    stop_percentage = 0.05  # Default 5%
                    if risk_level == "low":
                        stop_percentage = 0.03
                    elif risk_level == "high":
                        stop_percentage = 0.07
                    elif risk_level == "very_high":
                        stop_percentage = 0.1
                    
                    stop_loss = entry_price * (1 - stop_percentage)
                    exit_strategy["stop_loss"] = stop_loss
                    exit_strategy["reasoning"].append(f"Stop loss set at {stop_percentage*100:.1f}% below entry: {stop_loss:.2f}")
            else:
                # No support levels, use percentage-based stop loss
                stop_percentage = 0.05  # Default 5%
                if risk_level == "low":
                    stop_percentage = 0.03
                elif risk_level == "high":
                    stop_percentage = 0.07
                elif risk_level == "very_high":
                    stop_percentage = 0.1
                
                stop_loss = entry_price * (1 - stop_percentage)
                exit_strategy["stop_loss"] = stop_loss
                exit_strategy["reasoning"].append(f"Stop loss set at {stop_percentage*100:.1f}% below entry: {stop_loss:.2f}")
            
            # Set take profit targets based on resistance levels and risk/reward ratio
            if resistance_levels and len(resistance_levels) > 0:
                # Use resistance levels above entry price
                valid_resistances = [r.get("level") for r in resistance_levels if r.get("level") > entry_price]
                
                if valid_resistances:
                    # Calculate risk
                    risk = entry_price - exit_strategy["stop_loss"]
                    
                    # Set multiple take profit targets
                    for i, resistance in enumerate(valid_resistances[:3]):  # Use up to 3 resistance levels
                        reward = resistance - entry_price
                        risk_reward = reward / risk if risk > 0 else 1
                        
                        target = {
                            "price": resistance,
                            "percentage": 0.33 if i < 2 else 0.34,  # Distribute position
                            "risk_reward": risk_reward
                        }
                        
                        exit_strategy["take_profit"].append(target)
                        exit_strategy["reasoning"].append(f"Take profit {i+1} set at resistance: {resistance:.2f} (R/R: {risk_reward:.2f})")
                else:
                    # No valid resistance above, use percentage-based targets
                    target1 = entry_price * 1.1  # 10% gain
                    target2 = entry_price * 1.2  # 20% gain
                    target3 = entry_price * 1.5  # 50% gain
                    
                    exit_strategy["take_profit"] = [
                        {"price": target1, "percentage": 0.33, "risk_reward": 2},
                        {"price": target2, "percentage": 0.33, "risk_reward": 4},
                        {"price": target3, "percentage": 0.34, "risk_reward": 10}
                    ]
                    
                    exit_strategy["reasoning"].append(f"Take profit targets set at 10%, 20%, and 50% above entry")
            else:
                # No resistance levels, use percentage-based targets
                target1 = entry_price * 1.1  # 10% gain
                target2 = entry_price * 1.2  # 20% gain
                target3 = entry_price * 1.5  # 50% gain
                
                exit_strategy["take_profit"] = [
                    {"price": target1, "percentage": 0.33, "risk_reward": 2},
                    {"price": target2, "percentage": 0.33, "risk_reward": 4},
                    {"price": target3, "percentage": 0.34, "risk_reward": 10}
                ]
                
                exit_strategy["reasoning"].append(f"Take profit targets set at 10%, 20%, and 50% above entry")
            
            # Set trailing stop if appropriate
            trend = patterns.get("trend", {})
            trend_overall = trend.get("overall", "neutral")
            
            if trend_overall in ["strong_bullish", "bullish"]:
                # In strong trends, use trailing stop
                trail_percentage = 0.1  # 10% trailing stop
                exit_strategy["trailing_stop"] = {
                    "activation_price": exit_strategy["take_profit"][0]["price"],
                    "trail_percentage": trail_percentage
                }
                
                exit_strategy["reasoning"].append(f"Trailing stop of {trail_percentage*100:.1f}% activated after first target hit to capture extended move")
            
            # Set time-based stop if needed
            time_frames = {
                "short_term": 7,  # 7 days
                "medium_term": 30,  # 30 days
                "long_term": 90   # 90 days
            }
            
            exit_strategy["time_stop"] = {
                "days": time_frames["medium_term"],
                "action": "reevaluate"
            }
            
            exit_strategy["reasoning"].append(f"Reevaluate position after {time_frames['medium_term']} days if targets not reached")
            
            return exit_strategy
        
        except Exception as e:
            logger.error(f"Error generating exit strategy for {symbol}: {e}")
            # Default conservative exit strategy
            return {
                "take_profit": [{"price": current_price * 1.1, "percentage": 1.0, "risk_reward": 2}],
                "stop_loss": current_price * 0.95,
                "reasoning": [f"Error: {str(e)}", "Using default conservative exit strategy"]
            }
    
    def _generate_position_sizing(self, symbol: str, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate position sizing recommendations"""
        try:
            # Get risk level
            risk_level = risk_data.get("risk_level", "medium")
            risk_score = risk_data.get("risk_score", 50)
            
            # Default position sizing based on risk level
            base_percentage = 0.05  # 5% of portfolio as base
            
            if risk_level == "very_low":
                base_percentage = 0.1  # 10% for very low risk
            elif risk_level == "low":
                base_percentage = 0.075  # 7.5% for low risk
            elif risk_level == "high":
                base_percentage = 0.03  # 3% for high risk
            elif risk_level == "very_high":
                base_percentage = 0.01  # 1% for very high risk
            
            # Adjust based on risk tolerance
            risk_tolerance = risk_data.get("user_risk_tolerance", "medium")
            
            if risk_tolerance == "low":
                base_percentage *= 0.75
            elif risk_tolerance == "high":
                base_percentage *= 1.25
            
            # Cap the percentage
            base_percentage = min(base_percentage, 0.15)  # Maximum 15% of portfolio
            
            # Create position sizing recommendation
            sizing = {
                "portfolio_percentage": base_percentage,
                "max_risk_per_trade": 0.01,  # 1% max risk per trade
                "recommended_approach": "scaled" if risk_level in ["high", "very_high"] else "full",
                "reasoning": [
                    f"Position size of {base_percentage*100:.1f}% recommended based on {risk_level} risk level",
                    f"Maximum risk per trade should not exceed 1% of total portfolio"
                ]
            }
            
            return sizing
        
        except Exception as e:
            logger.error(f"Error generating position sizing for {symbol}: {e}")
            return {
                "portfolio_percentage": 0.02,  # Conservative 2%
                "max_risk_per_trade": 0.01,
                "recommended_approach": "scaled",
                "reasoning": [f"Error: {str(e)}", "Using conservative position sizing"]
            }
    
    def _generate_time_horizon(self, symbol: str, patterns: Dict[str, Any], 
                             risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate time horizon recommendation"""
        try:
            # Default time horizon
            horizon = "medium"  # Medium-term trade
            days = 30
            
            # Get trend information
            trend = patterns.get("trend", {})
            trend_overall = trend.get("overall", "neutral")
            
            # Get risk level
            risk_level = risk_data.get("risk_level", "medium")
            
            # Adjust based on trend and risk
            if trend_overall in ["strong_bullish", "bullish"]:
                if risk_level in ["low", "very_low"]:
                    horizon = "long"
                    days = 90
                else:
                    horizon = "medium"
                    days = 30
            elif trend_overall in ["strong_bearish", "bearish"]:
                horizon = "short"
                days = 14
            else:  # Neutral trend
                if risk_level in ["high", "very_high"]:
                    horizon = "short"
                    days = 14
                else:
                    horizon = "medium"
                    days = 30
            
            # Create time horizon recommendation
            time_horizon = {
                "horizon": horizon,
                "days": days,
                "reasoning": [
                    f"{horizon.capitalize()}-term trade ({days} days) recommended based on {trend_overall} trend and {risk_level} risk"
                ]
            }
            
            return time_horizon
        
        except Exception as e:
            logger.error(f"Error generating time horizon for {symbol}: {e}")
            return {
                "horizon": "medium",
                "days": 30,
                "reasoning": [f"Error: {str(e)}", "Using default medium-term horizon"]
            }

# Singleton instance
strategy_formulation = StrategyFormulation()

# Helper function to get the singleton instance
def get_strategy_formulation():
    return strategy_formulation

# Example usage
if __name__ == "__main__":
    strategy = get_strategy_formulation()
    # Formulate strategy for Bitcoin
    btc_strategy = strategy.formulate_strategy("BTC", "spot", "medium")
    print(f"BTC Strategy: {btc_strategy}")