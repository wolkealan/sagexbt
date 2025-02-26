from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from config.config import DatabaseConfig
from utils.logger import get_logger
from utils.database import get_database
from knowledge.information_extraction import get_information_extraction

logger = get_logger("event_correlation")

class EventCorrelation:
    """Correlates market events with price movements and identifies patterns"""
    
    def __init__(self):
        self.db = get_database()
        self.information_extraction = get_information_extraction()
    
    def correlate_events_with_price(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Correlate news and market events with price movements
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days to analyze
            
        Returns:
            Dictionary with correlation analysis
        """
        try:
            # Calculate date range
            from_date = datetime.now() - timedelta(days=days)
            
            # Get price data
            price_data = self._get_price_data(symbol, from_date)
            if not price_data or "error" in price_data:
                return {"error": f"No price data available for {symbol}"}
            
            # Get news events
            news_events = self._get_news_events(symbol, from_date)
            
            # Get market events
            market_events = self._get_market_events(symbol, from_date)
            
            # Analyze price reactions to news events
            news_correlations = self._analyze_price_reactions(symbol, news_events, price_data)
            
            # Analyze price conformance to technical indicators
            indicator_correlations = self._analyze_indicator_conformance(symbol, market_events, price_data)
            
            # Identify most impactful event types
            impact_analysis = self._analyze_event_impact(news_correlations, indicator_correlations)
            
            # Create correlation analysis report
            correlation_analysis = {
                "symbol": symbol,
                "time_range": f"{days} days",
                "news_correlations": news_correlations,
                "indicator_correlations": indicator_correlations,
                "impact_analysis": impact_analysis,
                "price_summary": {
                    "start_price": price_data[0]["price"] if price_data else None,
                    "end_price": price_data[-1]["price"] if price_data else None,
                    "total_change_pct": self._calculate_change(
                        price_data[0]["price"] if price_data else 0,
                        price_data[-1]["price"] if price_data else 0
                    ),
                    "volatility": self._calculate_volatility(price_data)
                },
                "timestamp": datetime.now()
            }
            
            logger.info(f"Generated event correlation analysis for {symbol}")
            return correlation_analysis
        
        except Exception as e:
            logger.error(f"Error correlating events with price for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    def identify_recurring_patterns(self, symbol: str, months: int = 6) -> Dict[str, Any]:
        """
        Identify recurring patterns between events and price movements
        
        Args:
            symbol: Cryptocurrency symbol
            months: Number of months to analyze
            
        Returns:
            Dictionary with recurring pattern analysis
        """
        try:
            # Calculate date range (more extended period for pattern recognition)
            from_date = datetime.now() - timedelta(days=30*months)
            
            # Get price data
            price_data = self._get_price_data(symbol, from_date)
            if not price_data or "error" in price_data:
                return {"error": f"No price data available for {symbol}"}
            
            # Get news events
            news_events = self._get_news_events(symbol, from_date)
            
            # Get market events
            market_events = self._get_market_events(symbol, from_date)
            
            # Identify recurring news event patterns
            news_patterns = self._identify_news_patterns(symbol, news_events, price_data)
            
            # Identify recurring technical patterns
            technical_patterns = self._identify_technical_patterns(symbol, market_events, price_data)
            
            # Create pattern analysis report
            pattern_analysis = {
                "symbol": symbol,
                "time_range": f"{months} months",
                "news_patterns": news_patterns,
                "technical_patterns": technical_patterns,
                "confidence_level": self._calculate_confidence_level(news_patterns, technical_patterns),
                "timestamp": datetime.now()
            }
            
            logger.info(f"Generated recurring pattern analysis for {symbol}")
            return pattern_analysis
        
        except Exception as e:
            logger.error(f"Error identifying recurring patterns for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    def correlate_multiple_coins(self, symbols: List[str], days: int = 30) -> Dict[str, Any]:
        """
        Correlate events and price movements across multiple cryptocurrencies
        
        Args:
            symbols: List of cryptocurrency symbols
            days: Number of days to analyze
            
        Returns:
            Dictionary with cross-coin correlation analysis
        """
        try:
            if not symbols or len(symbols) < 2:
                return {"error": "Need at least two symbols for cross-correlation"}
            
            coin_data = {}
            
            # Get data for each coin
            for symbol in symbols:
                correlation = self.correlate_events_with_price(symbol, days)
                coin_data[symbol] = correlation
            
            # Calculate cross-correlations
            cross_correlations = self._calculate_cross_correlations(coin_data)
            
            # Identify shared impactful events
            shared_events = self._identify_shared_events(coin_data)
            
            # Create cross-coin analysis report
            cross_analysis = {
                "symbols": symbols,
                "time_range": f"{days} days",
                "cross_correlations": cross_correlations,
                "shared_events": shared_events,
                "market_leader": self._identify_market_leader(coin_data, cross_correlations),
                "timestamp": datetime.now()
            }
            
            logger.info(f"Generated cross-coin correlation analysis for {symbols}")
            return cross_analysis
        
        except Exception as e:
            logger.error(f"Error correlating multiple coins: {e}")
            return {
                "symbols": symbols,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    def _get_price_data(self, symbol: str, from_date: datetime) -> List[Dict[str, Any]]:
        """Get historical price data from database"""
        try:
            # Query for market data
            query = {
                "key": {"$regex": f"^binance_{symbol}_1d_ohlcv"},
                "timestamp": {"$gte": from_date}
            }
            
            results = self.db.find_many(DatabaseConfig.MARKET_DATA_COLLECTION, query, 
                                       sort=[("timestamp", 1)])
            
            # Extract and format price data
            price_data = []
            for result in results:
                data = result.get("data", {})
                if not data:
                    continue
                
                # Extract OHLCV data
                ohlcv_data = data.get("data", [])
                
                for ohlcv in ohlcv_data:
                    if len(ohlcv) >= 5:  # Ensure we have enough data points
                        timestamp = ohlcv[0]  # Timestamp is first element
                        close_price = ohlcv[4]  # Close price is fifth element
                        
                        # Convert timestamp to datetime if it's in milliseconds
                        if isinstance(timestamp, int) and timestamp > 1000000000000:
                            timestamp = datetime.fromtimestamp(timestamp / 1000)
                        elif isinstance(timestamp, int):
                            timestamp = datetime.fromtimestamp(timestamp)
                        
                        price_data.append({
                            "timestamp": timestamp,
                            "price": close_price
                        })
            
            # Sort by timestamp
            price_data.sort(key=lambda x: x["timestamp"])
            
            logger.debug(f"Retrieved {len(price_data)} price data points for {symbol}")
            return price_data
        
        except Exception as e:
            logger.error(f"Error getting price data for {symbol}: {e}")
            return []
    
    def _get_news_events(self, symbol: str, from_date: datetime) -> List[Dict[str, Any]]:
        """Get news events from database"""
        try:
            # Query for news events
            query = {
                "cryptocurrencies": symbol,
                "source_type": "news",
                "processed_at": {"$gte": from_date}
            }
            
            results = self.db.find_many("extracted_information", query, 
                                      sort=[("processed_at", 1)])
            
            # Extract and format news events
            news_events = []
            for result in results:
                info = result.get("info", {})
                
                # Get publication timestamp
                published_at = info.get("published_at", "")
                if not published_at:
                    continue
                
                # Convert to datetime if it's a string
                if isinstance(published_at, str):
                    try:
                        published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    except:
                        continue
                
                # Process each key event from the news
                key_events = info.get("key_events", [])
                sentiment = info.get("sentiment", "neutral")
                
                if key_events:
                    for event in key_events:
                        news_events.append({
                            "timestamp": published_at,
                            "type": "news",
                            "event_type": event.get("type", "general"),
                            "description": event.get("description", ""),
                            "sentiment": sentiment,
                            "source": info.get("source", "Unknown"),
                            "cryptocurrencies": info.get("cryptocurrencies", [symbol])
                        })
                else:
                    # If no specific events, add the general news with sentiment
                    news_events.append({
                        "timestamp": published_at,
                        "type": "news",
                        "event_type": "general",
                        "description": info.get("title", ""),
                        "sentiment": sentiment,
                        "source": info.get("source", "Unknown"),
                        "cryptocurrencies": info.get("cryptocurrencies", [symbol])
                    })
            
            logger.debug(f"Retrieved {len(news_events)} news events for {symbol}")
            return news_events
        
        except Exception as e:
            logger.error(f"Error getting news events for {symbol}: {e}")
            return []
    
    def _get_market_events(self, symbol: str, from_date: datetime) -> List[Dict[str, Any]]:
        """Get market and technical events from database"""
        try:
            # Query for market insights
            query = {
                "symbol": symbol,
                "source_type": "market",
                "processed_at": {"$gte": from_date}
            }
            
            results = self.db.find_many("extracted_information", query, 
                                      sort=[("processed_at", 1)])
            
            # Extract and format market events
            market_events = []
            for result in results:
                info = result.get("info", {})
                
                # Get timestamp
                timestamp = info.get("timestamp", None)
                if not timestamp:
                    continue
                
                # Process each market insight
                insights = info.get("insights", [])
                
                for insight in insights:
                    market_events.append({
                        "timestamp": timestamp,
                        "type": "market",
                        "event_type": insight.get("type", "general"),
                        "indicator": insight.get("indicator", ""),
                        "timeframe": insight.get("timeframe", "1d"),
                        "description": insight.get("insight", ""),
                        "significance": insight.get("significance", "medium"),
                        "symbol": symbol
                    })
            
            logger.debug(f"Retrieved {len(market_events)} market events for {symbol}")
            return market_events
        
        except Exception as e:
            logger.error(f"Error getting market events for {symbol}: {e}")
            return []
    
    def _analyze_price_reactions(self, symbol: str, news_events: List[Dict[str, Any]], 
                               price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze price reactions to news events"""
        try:
            if not news_events or not price_data:
                return {"no_data": True}
            
            # Group events by type
            event_types = {}
            for event in news_events:
                event_type = event.get("event_type", "general")
                if event_type not in event_types:
                    event_types[event_type] = []
                event_types[event_type].append(event)
            
            # Analyze each event type
            correlations = {}
            for event_type, events in event_types.items():
                # Skip if too few events
                if len(events) < 3:
                    continue
                
                # Track price changes after events
                changes_24h = []
                changes_72h = []
                
                for event in events:
                    # Get event timestamp
                    event_time = event.get("timestamp")
                    
                    # Find price at event time
                    price_at_event = self._find_price_at_time(event_time, price_data)
                    if not price_at_event:
                        continue
                    
                    # Find price 24h after event
                    price_24h_after = self._find_price_at_time(event_time + timedelta(hours=24), price_data)
                    if price_24h_after:
                        change_24h = self._calculate_change(price_at_event, price_24h_after)
                        changes_24h.append(change_24h)
                    
                    # Find price 72h after event
                    price_72h_after = self._find_price_at_time(event_time + timedelta(hours=72), price_data)
                    if price_72h_after:
                        change_72h = self._calculate_change(price_at_event, price_72h_after)
                        changes_72h.append(change_72h)
                
                # Calculate statistics
                if changes_24h:
                    avg_change_24h = sum(changes_24h) / len(changes_24h)
                    median_change_24h = sorted(changes_24h)[len(changes_24h) // 2]
                    positive_count_24h = sum(1 for c in changes_24h if c > 0)
                    negative_count_24h = sum(1 for c in changes_24h if c < 0)
                else:
                    avg_change_24h = 0
                    median_change_24h = 0
                    positive_count_24h = 0
                    negative_count_24h = 0
                
                if changes_72h:
                    avg_change_72h = sum(changes_72h) / len(changes_72h)
                    median_change_72h = sorted(changes_72h)[len(changes_72h) // 2]
                    positive_count_72h = sum(1 for c in changes_72h if c > 0)
                    negative_count_72h = sum(1 for c in changes_72h if c < 0)
                else:
                    avg_change_72h = 0
                    median_change_72h = 0
                    positive_count_72h = 0
                    negative_count_72h = 0
                
                # Determine correlation strength
                correlation_strength = "weak"
                if abs(avg_change_24h) > 5 or abs(avg_change_72h) > 10:
                    correlation_strength = "strong"
                elif abs(avg_change_24h) > 2 or abs(avg_change_72h) > 5:
                    correlation_strength = "moderate"
                
                # Determine correlation direction
                correlation_direction = "neutral"
                if avg_change_24h > 1 and avg_change_72h > 2:
                    correlation_direction = "positive"
                elif avg_change_24h < -1 and avg_change_72h < -2:
                    correlation_direction = "negative"
                
                # Add to correlations
                correlations[event_type] = {
                    "events_count": len(events),
                    "avg_change_24h": avg_change_24h,
                    "median_change_24h": median_change_24h,
                    "positive_count_24h": positive_count_24h,
                    "negative_count_24h": negative_count_24h,
                    "avg_change_72h": avg_change_72h,
                    "median_change_72h": median_change_72h,
                    "positive_count_72h": positive_count_72h,
                    "negative_count_72h": negative_count_72h,
                    "correlation_strength": correlation_strength,
                    "correlation_direction": correlation_direction
                }
            
            return correlations
        
        except Exception as e:
            logger.error(f"Error analyzing price reactions for {symbol}: {e}")
            return {"error": str(e)}
    
    def _analyze_indicator_conformance(self, symbol: str, market_events: List[Dict[str, Any]], 
                                     price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how well price movements conform to technical indicators"""
        try:
            if not market_events or not price_data:
                return {"no_data": True}
            
            # Group events by indicator
            indicators = {}
            for event in market_events:
                indicator = event.get("indicator", "general")
                if indicator not in indicators:
                    indicators[indicator] = []
                indicators[indicator].append(event)
            
            # Analyze each indicator
            conformance = {}
            for indicator_name, events in indicators.items():
                # Skip if too few events
                if len(events) < 3:
                    continue
                
                # Track prediction accuracy
                correct_predictions = 0
                total_predictions = 0
                
                for event in events:
                    # Get event description and significance
                    description = event.get("description", "")
                    significance = event.get("significance", "medium")
                    event_time = event.get("timestamp")
                    
                    # Skip if no clear prediction in description
                    if not any(word in description.lower() for word in ["bullish", "bearish", "overbought", "oversold", "above", "below"]):
                        continue
                    
                    # Determine prediction direction
                    prediction_direction = "neutral"
                    if any(word in description.lower() for word in ["bullish", "oversold", "support"]):
                        prediction_direction = "bullish"
                    elif any(word in description.lower() for word in ["bearish", "overbought", "resistance"]):
                        prediction_direction = "bearish"
                    
                    if prediction_direction == "neutral":
                        continue
                    
                    total_predictions += 1
                    
                    # Check price movement after indicator
                    price_at_event = self._find_price_at_time(event_time, price_data)
                    if not price_at_event:
                        continue
                    
                    # Timeframe depends on indicator significance
                    hours_to_check = 24
                    if significance == "high":
                        hours_to_check = 72
                    elif significance == "low":
                        hours_to_check = 12
                    
                    # Find price after timeframe
                    price_after = self._find_price_at_time(event_time + timedelta(hours=hours_to_check), price_data)
                    if not price_after:
                        continue
                    
                    # Calculate change
                    change = self._calculate_change(price_at_event, price_after)
                    
                    # Check if prediction was correct
                    if (prediction_direction == "bullish" and change > 0) or (prediction_direction == "bearish" and change < 0):
                        correct_predictions += 1
                
                # Calculate accuracy
                accuracy = 0
                if total_predictions > 0:
                    accuracy = correct_predictions / total_predictions * 100
                
                # Determine reliability
                reliability = "low"
                if accuracy >= 70:
                    reliability = "high"
                elif accuracy >= 55:
                    reliability = "medium"
                
                # Add to conformance
                conformance[indicator_name] = {
                    "events_count": len(events),
                    "predictive_events": total_predictions,
                    "correct_predictions": correct_predictions,
                    "accuracy": accuracy,
                    "reliability": reliability
                }
            
            return conformance
        
        except Exception as e:
            logger.error(f"Error analyzing indicator conformance for {symbol}: {e}")
            return {"error": str(e)}
    
    def _analyze_event_impact(self, news_correlations: Dict[str, Any], 
                            indicator_correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Identify most impactful event types across news and indicators"""
        try:
            impact_analysis = {
                "high_impact_news": [],
                "high_impact_indicators": [],
                "combined_impact_score": {}
            }
            
            # Find high-impact news events
            for event_type, data in news_correlations.items():
                if data.get("correlation_strength", "weak") == "strong":
                    impact_analysis["high_impact_news"].append({
                        "event_type": event_type,
                        "direction": data.get("correlation_direction", "neutral"),
                        "avg_change_24h": data.get("avg_change_24h", 0),
                        "events_count": data.get("events_count", 0)
                    })
            
            # Find high-reliability indicators
            for indicator, data in indicator_correlations.items():
                if data.get("reliability", "low") == "high":
                    impact_analysis["high_impact_indicators"].append({
                        "indicator": indicator,
                        "accuracy": data.get("accuracy", 0),
                        "events_count": data.get("events_count", 0)
                    })
            
            # Calculate combined impact score for decision-making
            combined_score = {}
            
            # Add news impact
            for event_type, data in news_correlations.items():
                strength_score = 0
                if data.get("correlation_strength") == "strong":
                    strength_score = 3
                elif data.get("correlation_strength") == "moderate":
                    strength_score = 2
                elif data.get("correlation_strength") == "weak":
                    strength_score = 1
                
                direction_score = 0
                if data.get("correlation_direction") == "positive":
                    direction_score = data.get("avg_change_72h", 0)
                elif data.get("correlation_direction") == "negative":
                    direction_score = -data.get("avg_change_72h", 0)
                
                combined_score[f"news_{event_type}"] = strength_score * direction_score
            
            # Add indicator impact
            for indicator, data in indicator_correlations.items():
                reliability_score = 0
                if data.get("reliability") == "high":
                    reliability_score = 3
                elif data.get("reliability") == "medium":
                    reliability_score = 2
                elif data.get("reliability") == "low":
                    reliability_score = 1
                
                accuracy_score = data.get("accuracy", 50) / 50  # Normalize around 1.0
                
                combined_score[f"indicator_{indicator}"] = reliability_score * accuracy_score
            
            impact_analysis["combined_impact_score"] = combined_score
            
            return impact_analysis
        
        except Exception as e:
            logger.error(f"Error analyzing event impact: {e}")
            return {"error": str(e)}
    
    def _identify_news_patterns(self, symbol: str, news_events: List[Dict[str, Any]], 
                              price_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify recurring patterns in news events and price reactions"""
        try:
            if not news_events or not price_data:
                return []
            
            patterns = []
            
            # Group events by type
            event_types = {}
            for event in news_events:
                event_type = event.get("event_type", "general")
                sentiment = event.get("sentiment", "neutral")
                
                key = f"{event_type}_{sentiment}"
                if key not in event_types:
                    event_types[key] = []
                event_types[key].append(event)
            
            # Look for patterns in each event type
            for key, events in event_types.items():
                # Skip if too few events
                if len(events) < 5:
                    continue
                
                event_type, sentiment = key.split("_")
                
                # Track price movements after events
                price_movements = []
                
                for event in events:
                    event_time = event.get("timestamp")
                    
                    # Track price for 5 days after event
                    movement = []
                    price_at_event = self._find_price_at_time(event_time, price_data)
                    
                    if not price_at_event:
                        continue
                    
                    # Record price changes for each day
                    for days in range(1, 6):
                        price_after = self._find_price_at_time(event_time + timedelta(days=days), price_data)
                        if price_after:
                            change = self._calculate_change(price_at_event, price_after)
                            movement.append(change)
                        else:
                            movement.append(None)
                    
                    price_movements.append(movement)
                
                # Need at least 5 complete movements to identify patterns
                complete_movements = [m for m in price_movements if len(m) == 5 and all(x is not None for x in m)]
                if len(complete_movements) < 5:
                    continue
                
                # Calculate average movement pattern
                avg_movement = [sum(day) / len(complete_movements) for day in zip(*complete_movements)]
                
                # Calculate consistency (standard deviation)
                std_movement = [np.std([m[i] for m in complete_movements]) for i in range(5)]
                
                # Calculate consistency percentage
                same_direction_count = []
                for i in range(5):
                    if avg_movement[i] > 0:
                        count = sum(1 for m in complete_movements if m[i] > 0)
                    else:
                        count = sum(1 for m in complete_movements if m[i] < 0)
                    
                    same_direction_count.append(count / len(complete_movements) * 100)
                
                # Determine if there's a significant pattern
                avg_consistency = sum(same_direction_count) / len(same_direction_count)
                
                if avg_consistency >= 60:  # At least 60% consistent
                    patterns.append({
                        "event_type": event_type,
                        "sentiment": sentiment,
                        "events_count": len(complete_movements),
                        "avg_movement": avg_movement,
                        "std_movement": std_movement,
                        "consistency_pct": same_direction_count,
                        "avg_consistency": avg_consistency,
                        "pattern_strength": "strong" if avg_consistency >= 75 else "moderate",
                        "price_direction": "up" if avg_movement[4] > 0 else "down",
                        "max_change": max(abs(day) for day in avg_movement)
                    })
            
            # Sort by pattern strength
            patterns.sort(key=lambda x: x["avg_consistency"], reverse=True)
            
            return patterns
        
        except Exception as e:
            logger.error(f"Error identifying news patterns for {symbol}: {e}")
            return []
    
    def _identify_technical_patterns(self, symbol: str, market_events: List[Dict[str, Any]], 
                                   price_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify recurring patterns in technical indicators"""
        try:
            if not market_events or not price_data:
                return []
            
            patterns = []
            
            # Group events by indicator and event type
            indicator_types = {}
            for event in market_events:
                indicator = event.get("indicator", "general")
                event_type = event.get("event_type", "general")
                
                key = f"{indicator}_{event_type}"
                if key not in indicator_types:
                    indicator_types[key] = []
                indicator_types[key].append(event)
            
            # Look for patterns in each indicator type
            for key, events in indicator_types.items():
                # Skip if too few events
                if len(events) < 5:
                    continue
                
                indicator, event_type = key.split("_")
                
                # Track price movements after indicator events
                price_movements = []
                
                for event in events:
                    event_time = event.get("timestamp")
                    
                    # Track price for 5 days after event
                    movement = []
                    price_at_event = self._find_price_at_time(event_time, price_data)
                    
                    if not price_at_event:
                        continue
                    
                    # Record price changes for each day
                    for days in range(1, 6):
                        price_after = self._find_price_at_time(event_time + timedelta(days=days), price_data)
                        if price_after:
                            change = self._calculate_change(price_at_event, price_after)
                            movement.append(change)
                        else:
                            movement.append(None)
                    
                    price_movements.append(movement)
                
                # Need at least 5 complete movements to identify patterns
                complete_movements = [m for m in price_movements if len(m) == 5 and all(x is not None for x in m)]
                if len(complete_movements) < 5:
                    continue
                
                # Calculate average movement pattern
                avg_movement = [sum(day) / len(complete_movements) for day in zip(*complete_movements)]
                
                # Calculate consistency (standard deviation)
                std_movement = [np.std([m[i] for m in complete_movements]) for i in range(5)]
                
                # Calculate consistency percentage
                same_direction_count = []
                for i in range(5):
                    if avg_movement[i] > 0:
                        count = sum(1 for m in complete_movements if m[i] > 0)
                    else:
                        count = sum(1 for m in complete_movements if m[i] < 0)
                    
                    same_direction_count.append(count / len(complete_movements) * 100)
                    # Determine if there's a significant pattern
                avg_consistency = sum(same_direction_count) / len(same_direction_count)
                
                if avg_consistency >= 60:  # At least 60% consistent
                    patterns.append({
                        "indicator": indicator,
                        "event_type": event_type,
                        "events_count": len(complete_movements),
                        "avg_movement": avg_movement,
                        "std_movement": std_movement,
                        "consistency_pct": same_direction_count,
                        "avg_consistency": avg_consistency,
                        "pattern_strength": "strong" if avg_consistency >= 75 else "moderate",
                        "price_direction": "up" if avg_movement[4] > 0 else "down",
                        "max_change": max(abs(day) for day in avg_movement)
                    })
            
            # Sort by pattern strength
            patterns.sort(key=lambda x: x["avg_consistency"], reverse=True)
            
            return patterns
        
        except Exception as e:
            logger.error(f"Error identifying technical patterns for {symbol}: {e}")
            return []
    
    def _calculate_cross_correlations(self, coin_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate cross-correlations between different coins"""
        try:
            cross_correlations = {}
            
            symbols = list(coin_data.keys())
            
            # Calculate pairwise correlations
            for i in range(len(symbols)):
                symbol1 = symbols[i]
                
                for j in range(i+1, len(symbols)):
                    symbol2 = symbols[j]
                    
                    # Create pair key
                    pair = f"{symbol1}_{symbol2}"
                    
                    # Get price data
                    price_summary1 = coin_data[symbol1].get("price_summary", {})
                    price_summary2 = coin_data[symbol2].get("price_summary", {})
                    
                    # Skip if missing data
                    if not price_summary1 or not price_summary2:
                        continue
                    
                    # Compare total price changes
                    change1 = price_summary1.get("total_change_pct", 0)
                    change2 = price_summary2.get("total_change_pct", 0)
                    
                    # Check if they moved in the same direction
                    same_direction = (change1 > 0 and change2 > 0) or (change1 < 0 and change2 < 0)
                    
                    # Calculate rough correlation (simplified)
                    if same_direction:
                        correlation = min(abs(change1), abs(change2)) / max(abs(change1), abs(change2))
                    else:
                        correlation = -min(abs(change1), abs(change2)) / max(abs(change1), abs(change2))
                    
                    # Check shared events
                    shared_news_types = self._find_shared_news_types(coin_data[symbol1], coin_data[symbol2])
                    
                    cross_correlations[pair] = {
                        "correlation": correlation,
                        "relationship": "positive" if correlation > 0 else "negative",
                        "strength": abs(correlation),
                        "shared_news_types": shared_news_types,
                        "change_ratio": change1 / change2 if change2 != 0 else 0
                    }
            
            return cross_correlations
        
        except Exception as e:
            logger.error(f"Error calculating cross-correlations: {e}")
            return {}
    
    def _identify_shared_events(self, coin_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify events that affected multiple coins"""
        try:
            shared_events = []
            
            # Extract all events for each coin
            coin_events = {}
            
            for symbol, data in coin_data.items():
                # Get news correlations
                news_correlations = data.get("news_correlations", {})
                
                # Add significant news events
                for event_type, details in news_correlations.items():
                    if details.get("correlation_strength", "weak") in ["strong", "moderate"]:
                        if event_type not in coin_events:
                            coin_events[event_type] = []
                        
                        coin_events[event_type].append({
                            "symbol": symbol,
                            "correlation_direction": details.get("correlation_direction", "neutral"),
                            "avg_change_24h": details.get("avg_change_24h", 0)
                        })
            
            # Find events that affected multiple coins
            for event_type, impacts in coin_events.items():
                if len(impacts) > 1:  # Affected multiple coins
                    # Check if direction was consistent
                    directions = [impact.get("correlation_direction") for impact in impacts]
                    consistent_direction = all(d == directions[0] for d in directions) and directions[0] != "neutral"
                    
                    shared_events.append({
                        "event_type": event_type,
                        "affected_coins": len(impacts),
                        "coin_impacts": impacts,
                        "consistent_direction": consistent_direction,
                        "direction": directions[0] if consistent_direction else "mixed",
                        "significance": "high" if len(impacts) >= 3 else "medium"
                    })
            
            # Sort by significance
            shared_events.sort(key=lambda x: x["affected_coins"], reverse=True)
            
            return shared_events
        
        except Exception as e:
            logger.error(f"Error identifying shared events: {e}")
            return []
    
    def _identify_market_leader(self, coin_data: Dict[str, Dict[str, Any]], 
                             cross_correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Identify which coin appears to lead the market movements"""
        try:
            symbols = list(coin_data.keys())
            
            # Count how many times each coin appears to lead others
            lead_count = {symbol: 0 for symbol in symbols}
            
            for pair, correlation in cross_correlations.items():
                # Skip weak correlations
                if correlation.get("strength", 0) < 0.5:
                    continue
                
                # Split the pair
                symbol1, symbol2 = pair.split("_")
                
                # Get price summary for each coin
                price_summary1 = coin_data[symbol1].get("price_summary", {})
                price_summary2 = coin_data[symbol2].get("price_summary", {})
                
                # Compare volatility to determine leader (higher volatility often follows)
                volatility1 = price_summary1.get("volatility", 0)
                volatility2 = price_summary2.get("volatility", 0)
                
                # Simple heuristic: coin with lower volatility often leads
                if volatility1 < volatility2:
                    lead_count[symbol1] += 1
                elif volatility2 < volatility1:
                    lead_count[symbol2] += 1
            
            # Find the coin with highest lead count
            max_leads = max(lead_count.values()) if lead_count else 0
            leaders = [s for s, count in lead_count.items() if count == max_leads]
            
            # If there's a clear leader
            if leaders and max_leads > 0:
                leader = leaders[0]
                
                return {
                    "symbol": leader,
                    "lead_count": max_leads,
                    "confidence": "high" if max_leads >= len(symbols) - 1 else "medium",
                    "reasoning": "Based on cross-correlation analysis and volatility patterns"
                }
            else:
                return {
                    "symbol": None,
                    "confidence": "low",
                    "reasoning": "No clear market leader identified"
                }
        
        except Exception as e:
            logger.error(f"Error identifying market leader: {e}")
            return {"error": str(e)}
    
    def _find_shared_news_types(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> List[str]:
        """Find news event types that affected both coins"""
        try:
            news_types1 = set(data1.get("news_correlations", {}).keys())
            news_types2 = set(data2.get("news_correlations", {}).keys())
            
            return list(news_types1.intersection(news_types2))
        
        except Exception as e:
            logger.error(f"Error finding shared news types: {e}")
            return []
    
    def _find_price_at_time(self, target_time: datetime, price_data: List[Dict[str, Any]]) -> Optional[float]:
        """Find the price at or closest to a specific time"""
        try:
            if not price_data:
                return None
            
            # Sort by timestamp just to be sure
            sorted_data = sorted(price_data, key=lambda x: x["timestamp"])
            
            # Find the closest data point
            closest_price = None
            min_diff = timedelta(days=365)  # Initialize with a large time difference
            
            for data_point in sorted_data:
                timestamp = data_point["timestamp"]
                diff = abs(timestamp - target_time)
                
                if diff < min_diff:
                    min_diff = diff
                    closest_price = data_point["price"]
                
                # If we found a point after the target time, we can stop searching
                if timestamp > target_time:
                    break
            
            # Only use the price if it's within 24 hours of the target time
            if min_diff <= timedelta(hours=24):
                return closest_price
            else:
                return None
        
        except Exception as e:
            logger.error(f"Error finding price at time: {e}")
            return None
    
    def _calculate_change(self, price_before: float, price_after: float) -> float:
        """Calculate percentage change between two prices"""
        try:
            if price_before <= 0:
                return 0
            
            return (price_after - price_before) / price_before * 100
        
        except Exception as e:
            logger.error(f"Error calculating change: {e}")
            return 0
    
    def _calculate_volatility(self, price_data: List[Dict[str, Any]]) -> float:
        """Calculate price volatility"""
        try:
            if not price_data or len(price_data) < 2:
                return 0
            
            # Extract prices
            prices = [data_point["price"] for data_point in price_data]
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(prices)):
                daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(daily_return)
            
            # Calculate standard deviation of returns
            if returns:
                return np.std(returns) * 100  # Convert to percentage
            else:
                return 0
        
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0
    
    def _calculate_confidence_level(self, news_patterns: List[Dict[str, Any]], 
                                  technical_patterns: List[Dict[str, Any]]) -> str:
        """Calculate overall confidence in the identified patterns"""
        try:
            # No patterns found
            if not news_patterns and not technical_patterns:
                return "very_low"
            
            # Count strong patterns
            strong_news = sum(1 for p in news_patterns if p.get("pattern_strength") == "strong")
            strong_technical = sum(1 for p in technical_patterns if p.get("pattern_strength") == "strong")
            
            # Count moderate patterns
            moderate_news = sum(1 for p in news_patterns if p.get("pattern_strength") == "moderate")
            moderate_technical = sum(1 for p in technical_patterns if p.get("pattern_strength") == "moderate")
            
            # Calculate weighted score
            score = (strong_news * 3 + strong_technical * 3 + moderate_news * 1 + moderate_technical * 1)
            
            # Determine confidence level
            if score >= 10:
                return "very_high"
            elif score >= 6:
                return "high"
            elif score >= 3:
                return "medium"
            elif score >= 1:
                return "low"
            else:
                return "very_low"
        
        except Exception as e:
            logger.error(f"Error calculating confidence level: {e}")
            return "low"

# Singleton instance
event_correlation = EventCorrelation()

# Helper function to get the singleton instance
def get_event_correlation():
    return event_correlation

# Example usage
if __name__ == "__main__":
    correlation = get_event_correlation()
    # Correlate events with price for Bitcoin
    btc_correlation = correlation.correlate_events_with_price("BTC", days=30)
    print(f"BTC Event Correlation: {btc_correlation}")
    # Identify recurring patterns
    btc_patterns = correlation.identify_recurring_patterns("BTC", months=3)
    print(f"BTC Recurring Patterns: {btc_patterns}")