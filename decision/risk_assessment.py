import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from config.config import TradingConfig
from utils.logger import get_logger
from data.market_data import get_market_data_provider
from data.news_provider import get_news_provider
from decision.pattern_recognition import get_pattern_recognition

logger = get_logger("risk_assessment")

class RiskAssessment:
    """Assesses trading risks for cryptocurrency investments"""
    
    def __init__(self):
        self.market_data = get_market_data_provider()
        self.news_provider = get_news_provider()
        self.pattern_recognition = get_pattern_recognition()
    
    def assess_risk(self, symbol: str, action_type: str = "spot", 
                   risk_tolerance: str = "medium") -> Dict[str, Any]:
        """
        Assess the risk level for a cryptocurrency trade
        
        Args:
            symbol: Cryptocurrency symbol
            action_type: 'spot' or 'futures' trading
            risk_tolerance: User's risk tolerance level ('low', 'medium', 'high')
            
        Returns:
            Risk assessment data
        """
        try:
            # Get market data
            market_summary = self.market_data.get_market_summary(symbol)
            
            # Get volatility data
            volatility = self._calculate_volatility(symbol)
            
            # Get market sentiment
            market_context = self.news_provider.get_market_context()
            
            # Get technical patterns
            patterns = self.pattern_recognition.identify_patterns(symbol)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(symbol, action_type, volatility, market_context, patterns)
            
            # Adjust risk metrics based on user's risk tolerance
            adjusted_metrics = self._adjust_for_risk_tolerance(risk_metrics, risk_tolerance)
            
            # Generate risk recommendations
            recommendations = self._generate_risk_recommendations(symbol, adjusted_metrics, action_type)
            
            # Create complete risk assessment
            assessment = {
                "symbol": symbol,
                "action_type": action_type,
                "risk_tolerance": risk_tolerance,
                "risk_score": adjusted_metrics["risk_score"],
                "risk_level": adjusted_metrics["risk_level"],
                "volatility": volatility,
                "risk_metrics": risk_metrics,
                "recommendations": recommendations,
                "timestamp": datetime.now()
            }
            
            logger.info(f"Generated risk assessment for {symbol}: {adjusted_metrics['risk_level']} risk")
            return assessment
        
        except Exception as e:
            logger.error(f"Error assessing risk for {symbol}: {e}")
            return {
                "symbol": symbol,
                "action_type": action_type,
                "risk_tolerance": risk_tolerance,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    def _calculate_volatility(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Calculate volatility metrics for the symbol"""
        try:
            # Get historical data
            df = self.market_data.get_historical_data(symbol, timeframe="1d")
            if df.empty:
                logger.warning(f"No data available for volatility calculation for {symbol}")
                return {"daily_volatility": 0, "status": "low"}
            
            # Ensure we have enough data
            if len(df) < days:
                days = len(df)
            
            # Calculate daily returns
            df = df.tail(days)
            df['returns'] = df['close'].pct_change() * 100
            
            # Calculate metrics
            daily_volatility = df['returns'].std()
            max_daily_drop = df['returns'].min()
            max_daily_gain = df['returns'].max()
            
            # Annualized volatility (traditional finance metric)
            annualized_volatility = daily_volatility * np.sqrt(365)
            
            # Determine volatility status
            status = "medium"
            if daily_volatility > 5:  # Highly volatile (arbitrary threshold, adjust as needed)
                status = "high"
            elif daily_volatility < 2:  # Low volatility
                status = "low"
            
            volatility = {
                "daily_volatility": round(daily_volatility, 2),
                "annualized_volatility": round(annualized_volatility, 2),
                "max_daily_drop": round(max_daily_drop, 2),
                "max_daily_gain": round(max_daily_gain, 2),
                "days": days,
                "status": status
            }
            
            logger.debug(f"Calculated volatility for {symbol}: {daily_volatility:.2f}%")
            return volatility
        
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return {"daily_volatility": 0, "status": "unknown", "error": str(e)}
    
    def _calculate_risk_metrics(self, symbol: str, action_type: str,
                              volatility: Dict[str, Any], 
                              market_context: Dict[str, Any],
                              patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics based on different factors"""
        try:
            risk_metrics = {}
            risk_score = 50  # Base risk score (neutral)
            
            # 1. Adjust for volatility (0-25 points)
            if volatility["status"] == "high":
                risk_score += 25
                risk_metrics["volatility_risk"] = "high"
            elif volatility["status"] == "medium":
                risk_score += 12.5
                risk_metrics["volatility_risk"] = "medium"
            else:
                risk_metrics["volatility_risk"] = "low"
            
            # 2. Adjust for market sentiment (0-25 points)
            # Get overall market sentiment
            overall_sentiment = market_context.get("overall_sentiment", 0)
            if overall_sentiment < -0.3:  # Negative sentiment
                risk_score += 25
                risk_metrics["sentiment_risk"] = "high"
            elif overall_sentiment > 0.3:  # Positive sentiment
                risk_metrics["sentiment_risk"] = "low"
            else:
                risk_score += 12.5
                risk_metrics["sentiment_risk"] = "medium"
            
            # 3. Adjust for technical patterns (0-25 points)
            pattern_risk = self._evaluate_pattern_risk(patterns)
            risk_score += pattern_risk * 25
            if pattern_risk > 0.7:
                risk_metrics["pattern_risk"] = "high"
            elif pattern_risk > 0.3:
                risk_metrics["pattern_risk"] = "medium"
            else:
                risk_metrics["pattern_risk"] = "low"
            
            # 4. Adjust for trading type (0-25 points)
            if action_type == "futures":
                risk_score += 25
                risk_metrics["instrument_risk"] = "high"
            else:
                risk_metrics["instrument_risk"] = "low"
            
            # Cap the risk score at 100
            risk_score = min(risk_score, 100)
            
            # Determine overall risk level
            risk_level = "medium"
            if risk_score >= 75:
                risk_level = "very_high"
            elif risk_score >= 60:
                risk_level = "high"
            elif risk_score >= 40:
                risk_level = "medium"
            elif risk_score >= 25:
                risk_level = "low"
            else:
                risk_level = "very_low"
            
            # Complete risk metrics
            risk_metrics["risk_score"] = risk_score
            risk_metrics["risk_level"] = risk_level
            
            logger.debug(f"Calculated risk metrics for {symbol}: score {risk_score}, level {risk_level}")
            return risk_metrics
        
        except Exception as e:
            logger.error(f"Error calculating risk metrics for {symbol}: {e}")
            return {"risk_score": 50, "risk_level": "medium", "error": str(e)}
    
    def _evaluate_pattern_risk(self, patterns: Dict[str, Any]) -> float:
        """Evaluate risk based on technical patterns"""
        try:
            pattern_risk = 0.5  # Default neutral risk
            
            # Get trend information
            trend = patterns.get("trend", {})
            trend_overall = trend.get("overall", "neutral")
            
            # Adjust risk based on trend
            if trend_overall in ["strong_bearish", "bearish"]:
                pattern_risk += 0.2
            elif trend_overall in ["strong_bullish", "bullish"]:
                pattern_risk -= 0.2
            
            # Check for specific high-risk patterns
            chart_patterns = patterns.get("chart_patterns", {})
            if "double_top" in chart_patterns or "head_and_shoulders" in chart_patterns:
                pattern_risk += 0.2
            elif "double_bottom" in chart_patterns or "inverse_head_and_shoulders" in chart_patterns:
                pattern_risk -= 0.2
            
            # Check for candlestick patterns
            candlestick_patterns = patterns.get("candlestick", {})
            for pattern_name, pattern_data in candlestick_patterns.items():
                significance = pattern_data.get("significance", "neutral")
                if significance == "bearish":
                    pattern_risk += 0.1
                elif significance == "bullish":
                    pattern_risk -= 0.1
            
            # Ensure risk is between 0 and 1
            pattern_risk = max(0, min(pattern_risk, 1))
            
            return pattern_risk
        
        except Exception as e:
            logger.error(f"Error evaluating pattern risk: {e}")
            return 0.5  # Default to medium risk
    
    def _adjust_for_risk_tolerance(self, risk_metrics: Dict[str, Any], risk_tolerance: str) -> Dict[str, Any]:
        """Adjust risk metrics based on user's risk tolerance"""
        # Clone the metrics to avoid modifying the original
        adjusted_metrics = risk_metrics.copy()
        
        # Get base risk score
        risk_score = risk_metrics.get("risk_score", 50)
        
        # Adjust based on risk tolerance
        if risk_tolerance == "low":
            # Increase perceived risk for risk-averse users
            risk_score = risk_score * 1.2
        elif risk_tolerance == "high":
            # Decrease perceived risk for risk-seeking users
            risk_score = risk_score * 0.8
        
        # Cap at 100
        risk_score = min(risk_score, 100)
        
        # Recalculate risk level based on adjusted score
        risk_level = "medium"
        if risk_score >= 75:
            risk_level = "very_high"
        elif risk_score >= 60:
            risk_level = "high"
        elif risk_score >= 40:
            risk_level = "medium"
        elif risk_score >= 25:
            risk_level = "low"
        else:
            risk_level = "very_low"
        
        # Update the metrics
        adjusted_metrics["risk_score"] = risk_score
        adjusted_metrics["risk_level"] = risk_level
        adjusted_metrics["user_risk_tolerance"] = risk_tolerance
        
        return adjusted_metrics
    
    def _generate_risk_recommendations(self, symbol: str, risk_metrics: Dict[str, Any], 
                                      action_type: str) -> List[str]:
        """Generate risk management recommendations based on the assessment"""
        recommendations = []
        risk_level = risk_metrics.get("risk_level", "medium")
        
        # Common recommendations for all risk levels
        recommendations.append("Only invest what you can afford to lose.")
        recommendations.append("Consider using stop-loss orders to limit potential losses.")
        
        # Add specific recommendations based on risk level
        if risk_level in ["high", "very_high"]:
            recommendations.append(f"Heightened risk detected for {symbol}. Consider reducing position size.")
            if action_type == "futures":
                recommendations.append("Leveraged trading significantly increases risk. Consider using spot trading instead.")
                recommendations.append("If proceeding with futures, use no more than 2x leverage.")
            recommendations.append("Implement a strict risk management plan with predefined exit points.")
            
        elif risk_level == "medium":
            recommendations.append(f"Moderate risk detected for {symbol}. Use standard position sizing.")
            if action_type == "futures":
                recommendations.append("Consider limiting leverage to reduce risk exposure.")
            recommendations.append("Balance your portfolio with less volatile assets.")
            
        else:  # Low or very low risk
            recommendations.append(f"Lower risk detected for {symbol}, but all crypto investments carry some risk.")
            recommendations.append("Consider dollar-cost averaging to reduce timing risk.")
            
        # Add volatility-specific recommendations
        volatility_risk = risk_metrics.get("volatility_risk", "medium")
        if volatility_risk == "high":
            recommendations.append(f"{symbol} shows high volatility. Prepare for significant price swings.")
        
        # Add sentiment-specific recommendations
        sentiment_risk = risk_metrics.get("sentiment_risk", "medium")
        if sentiment_risk == "high":
            recommendations.append("Market sentiment is negative. Consider waiting for sentiment improvement.")
        
        return recommendations

# Singleton instance
risk_assessment = RiskAssessment()

# Helper function to get the singleton instance
def get_risk_assessment():
    return risk_assessment

# Example usage
if __name__ == "__main__":
    assessment = get_risk_assessment()
    # Assess risk for Bitcoin
    risk_data = assessment.assess_risk("BTC", "spot", "medium")
    print(f"BTC Risk Assessment: {risk_data}")