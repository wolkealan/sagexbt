import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from config.config import DatabaseConfig
from utils.logger import get_logger
from utils.database import get_database
from knowledge.event_correlation import get_event_correlation

logger = get_logger("temporal_analysis")

class TemporalAnalysis:
    """Analyzes temporal patterns in cryptocurrency data"""
    
    def __init__(self):
        self.db = get_database()
        self.event_correlation = get_event_correlation()
    
    def analyze_time_series(self, symbol: str, timeframe: str = "1d", 
                          days: int = 365) -> Dict[str, Any]:
        """
        Perform time series analysis on price data
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Time period for analysis ("1d", "4h", etc.)
            days: Number of days of historical data to analyze
            
        Returns:
            Dictionary with time series analysis results
        """
        try:
            # Get historical price data
            price_data = self._get_historical_data(symbol, timeframe, days)
            
            if not price_data or len(price_data) < 30:
                return {"error": f"Not enough data for {symbol}"}
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(price_data)
            df.set_index("timestamp", inplace=True)
            
            # Resample to ensure regular frequency
            df = df.resample("D").last() if timeframe == "1d" else df.resample("H").last()
            
            # Forward fill missing values
            df.fillna(method="ffill", inplace=True)
            
            # Perform trend analysis
            trend_analysis = self._analyze_trend(df)
            
            # Perform seasonality analysis
            seasonality_analysis = self._analyze_seasonality(df)
            
            # Perform cyclical analysis
            cyclical_analysis = self._analyze_cycles(df)
            
            # Perform volatility analysis
            volatility_analysis = self._analyze_volatility(df)
            
            # Create time series analysis report
            analysis = {
                "symbol": symbol,
                "timeframe": timeframe,
                "data_points": len(df),
                "start_date": df.index[0].strftime("%Y-%m-%d"),
                "end_date": df.index[-1].strftime("%Y-%m-%d"),
                "trend_analysis": trend_analysis,
                "seasonality_analysis": seasonality_analysis,
                "cyclical_analysis": cyclical_analysis,
                "volatility_analysis": volatility_analysis,
                "timestamp": datetime.now()
            }
            
            logger.info(f"Completed time series analysis for {symbol}")
            return analysis
        
        except Exception as e:
            logger.error(f"Error performing time series analysis for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    def detect_temporal_patterns(self, symbol: str, pattern_type: str = "all") -> Dict[str, Any]:
        """
        Detect specific temporal patterns in price data
        
        Args:
            symbol: Cryptocurrency symbol
            pattern_type: Type of patterns to detect ("seasonality", "cycles", "volatility", or "all")
            
        Returns:
            Dictionary with detected patterns
        """
        try:
            # Get daily data for 2 years
            df_daily = pd.DataFrame(self._get_historical_data(symbol, "1d", 730))
            
            if df_daily.empty or len(df_daily) < 60:
                return {"error": f"Not enough data for {symbol}"}
            
            df_daily.set_index("timestamp", inplace=True)
            
            # Get hourly data for 60 days
            df_hourly = pd.DataFrame(self._get_historical_data(symbol, "1h", 60))
            
            if df_hourly.empty:
                df_hourly = None
            else:
                df_hourly.set_index("timestamp", inplace=True)
            
            patterns = {}
            
            # Detect patterns based on specified type
            if pattern_type in ["seasonality", "all"]:
                patterns["seasonality"] = self._detect_seasonality_patterns(df_daily, df_hourly)
            
            if pattern_type in ["cycles", "all"]:
                patterns["cycles"] = self._detect_cyclical_patterns(df_daily)
            
            if pattern_type in ["volatility", "all"]:
                patterns["volatility"] = self._detect_volatility_patterns(df_daily, df_hourly)
            
            # Create pattern detection report
            detection_results = {
                "symbol": symbol,
                "pattern_type": pattern_type,
                "patterns_detected": patterns,
                "confidence_level": self._calculate_pattern_confidence(patterns),
                "timestamp": datetime.now()
            }
            
            logger.info(f"Detected temporal patterns for {symbol}")
            return detection_results
        
        except Exception as e:
            logger.error(f"Error detecting temporal patterns for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    def analyze_historical_events(self, symbol: str, event_type: str = None) -> Dict[str, Any]:
        """
        Analyze how historical events correlate with price movements over time
        
        Args:
            symbol: Cryptocurrency symbol
            event_type: Specific event type to analyze (None for all)
            
        Returns:
            Dictionary with historical event analysis
        """
        try:
            # Get historical price data
            price_data = self._get_historical_data(symbol, "1d", 730)  # 2 years
            
            if not price_data or len(price_data) < 30:
                return {"error": f"Not enough data for {symbol}"}
            
            # Get historical news events
            news_events = self._get_historical_news_events(symbol, event_type)
            
            if not news_events:
                return {"error": f"No news events found for {symbol}"}
            
            # Group events by type
            events_by_type = {}
            for event in news_events:
                event_type = event.get("event_type", "general")
                if event_type not in events_by_type:
                    events_by_type[event_type] = []
                events_by_type[event_type].append(event)
            
            # Analyze each event type
            event_analysis = {}
            
            for event_type, events in events_by_type.items():
                if len(events) < 3:  # Need at least 3 events for meaningful analysis
                    continue
                
                # Analyze price impact for this event type
                impact_data = self._analyze_event_price_impact(events, price_data)
                temporal_distribution = self._analyze_event_temporal_distribution(events)
                
                event_analysis[event_type] = {
                    "count": len(events),
                    "price_impact": impact_data,
                    "temporal_distribution": temporal_distribution
                }
            
            # Create historical event analysis report
            analysis = {
                "symbol": symbol,
                "total_events": len(news_events),
                "event_types": list(event_analysis.keys()),
                "event_analysis": event_analysis,
                "most_impactful_event": self._identify_most_impactful_event(event_analysis),
                "timestamp": datetime.now()
            }
            
            logger.info(f"Completed historical event analysis for {symbol}")
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing historical events for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    def _get_historical_data(self, symbol: str, timeframe: str, days: int) -> List[Dict[str, Any]]:
        """Get historical price data from database"""
        try:
            # Calculate date range
            from_date = datetime.now() - timedelta(days=days)
            
            # Query for market data
            query = {
                "key": {"$regex": f"^binance_{symbol}_{timeframe}_ohlcv"},
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
                        open_price = ohlcv[1]  # Open price is second element
                        high_price = ohlcv[2]  # High price is third element
                        low_price = ohlcv[3]   # Low price is fourth element
                        close_price = ohlcv[4] # Close price is fifth element
                        volume = ohlcv[5] if len(ohlcv) > 5 else 0  # Volume if available
                        
                        # Convert timestamp to datetime if it's in milliseconds
                        if isinstance(timestamp, int) and timestamp > 1000000000000:
                            timestamp = datetime.fromtimestamp(timestamp / 1000)
                        elif isinstance(timestamp, int):
                            timestamp = datetime.fromtimestamp(timestamp)
                        
                        price_data.append({
                            "timestamp": timestamp,
                            "price": close_price,
                            "open": open_price,
                            "high": high_price,
                            "low": low_price,
                            "volume": volume
                        })
            
            # Sort by timestamp
            price_data.sort(key=lambda x: x["timestamp"])
            
            logger.debug(f"Retrieved {len(price_data)} historical data points for {symbol}")
            return price_data
        
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []
    
    def _get_historical_news_events(self, symbol: str, event_type: str = None) -> List[Dict[str, Any]]:
        """Get historical news events from database"""
        try:
            # Query for news events
            query = {
                "cryptocurrencies": symbol,
                "source_type": "news"
            }
            
            # Add event type filter if specified
            if event_type:
                query["info.key_events.type"] = event_type
            
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
                        # Skip if event_type is specified and doesn't match
                        if event_type and event.get("type", "") != event_type:
                            continue
                        
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
            
            logger.debug(f"Retrieved {len(news_events)} historical news events for {symbol}")
            return news_events
        
        except Exception as e:
            logger.error(f"Error getting historical news events for {symbol}: {e}")
            return []
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend component of time series"""
        try:
            # Apply moving average to identify trend
            df['ma_7'] = df['price'].rolling(window=7).mean()
            df['ma_30'] = df['price'].rolling(window=30).mean()
            df['ma_90'] = df['price'].rolling(window=90).mean()
            
            # Drop NaN values for analysis
            df_clean = df.dropna()
            
            if len(df_clean) < 90:
                return {"error": "Not enough data for trend analysis"}
            
            # Calculate trend direction and strength
            current_price = df_clean['price'].iloc[-1]
            price_30_days_ago = df_clean['price'].iloc[-30] if len(df_clean) >= 30 else df_clean['price'].iloc[0]
            price_90_days_ago = df_clean['price'].iloc[-90] if len(df_clean) >= 90 else df_clean['price'].iloc[0]
            
            pct_change_30d = (current_price - price_30_days_ago) / price_30_days_ago * 100
            pct_change_90d = (current_price - price_90_days_ago) / price_90_days_ago * 100
            
            # Determine trend direction
            if pct_change_30d > 5 and pct_change_90d > 10:
                trend_direction = "strong_uptrend"
            elif pct_change_30d > 3 and pct_change_90d > 5:
                trend_direction = "uptrend"
            elif pct_change_30d < -5 and pct_change_90d < -10:
                trend_direction = "strong_downtrend"
            elif pct_change_30d < -3 and pct_change_90d < -5:
                trend_direction = "downtrend"
            elif abs(pct_change_30d) < 3 and abs(pct_change_90d) < 5:
                trend_direction = "sideways"
            else:
                trend_direction = "mixed"
            
            # Check for trend acceleration/deceleration
            ma_7_slope = (df_clean['ma_7'].iloc[-1] - df_clean['ma_7'].iloc[-7]) / df_clean['ma_7'].iloc[-7] * 100
            ma_30_slope = (df_clean['ma_30'].iloc[-1] - df_clean['ma_30'].iloc[-7]) / df_clean['ma_30'].iloc[-7] * 100
            
            trend_momentum = "stable"
            if ma_7_slope > 1.5 * ma_30_slope and ma_7_slope > 0:
                trend_momentum = "accelerating"
            elif ma_7_slope < 0.5 * ma_30_slope and ma_7_slope > 0:
                trend_momentum = "decelerating"
            
            # Check for potential trend reversal
            reversal_signal = False
            if (trend_direction.endswith("uptrend") and ma_7_slope < 0 and ma_30_slope > 0) or \
               (trend_direction.endswith("downtrend") and ma_7_slope > 0 and ma_30_slope < 0):
                reversal_signal = True
            
            return {
                "direction": trend_direction,
                "momentum": trend_momentum,
                "change_30d_pct": pct_change_30d,
                "change_90d_pct": pct_change_90d,
                "reversal_signal": reversal_signal,
                "ma_7_slope": ma_7_slope,
                "ma_30_slope": ma_30_slope
            }
        
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {"error": str(e)}
    
    def _analyze_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonality in time series data"""
        try:
            # Need enough data for seasonal analysis
            if len(df) < 90:
                return {"error": "Not enough data for seasonality analysis"}
            
            # Try to decompose the time series
            try:
                # Convert to regular frequency if needed
                if not df.index.is_monotonic:
                    df = df.sort_index()
                
                # Fill any missing values
                # Fill any missing values
                df_filled = df.copy()
                df_filled['price'] = df_filled['price'].fillna(method='ffill')
                
                # Decompose into trend, seasonal, and residual components
                result = seasonal_decompose(df_filled['price'], model='additive', period=7)  # Weekly seasonality
                
                # Extract components
                trend = result.trend
                seasonal = result.seasonal
                residual = result.resid
                
                # Calculate strength of seasonality
                seasonal_strength = np.var(seasonal.dropna()) / np.var(residual.dropna() + seasonal.dropna())
                
                # Significant if seasonal strength > 0.1
                has_seasonality = seasonal_strength > 0.1
                
                # Find weekly patterns
                daily_means = df.groupby(df.index.dayofweek)['price'].mean()
                daily_std = df.groupby(df.index.dayofweek)['price'].std()
                
                # Find the day with highest and lowest average prices
                highest_day = daily_means.idxmax()
                lowest_day = daily_means.idxmin()
                
                # Map day numbers to names
                day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                          4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
                
                # Calculate monthly patterns if we have enough data
                has_monthly_pattern = False
                highest_month = None
                lowest_month = None
                
                if len(df) >= 365:  # Need at least a year of data
                    monthly_means = df.groupby(df.index.month)['price'].mean()
                    monthly_std = df.groupby(df.index.month)['price'].std()
                    
                    # Check if there's significant monthly variation
                    monthly_variation = monthly_means.std() / monthly_means.mean()
                    has_monthly_pattern = monthly_variation > 0.05
                    
                    if has_monthly_pattern:
                        highest_month = monthly_means.idxmax()
                        lowest_month = monthly_means.idxmin()
                        
                        # Map month numbers to names
                        month_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 
                                    5: 'May', 6: 'June', 7: 'July', 8: 'August', 
                                    9: 'September', 10: 'October', 11: 'November', 12: 'December'}
                
                seasonality_analysis = {
                    "has_seasonality": has_seasonality,
                    "seasonal_strength": seasonal_strength,
                    "weekly_pattern": {
                        "highest_day": day_map[highest_day],
                        "lowest_day": day_map[lowest_day],
                        "day_of_week_variation": daily_std.mean() / daily_means.mean()
                    },
                    "monthly_pattern": {
                        "has_monthly_pattern": has_monthly_pattern,
                        "highest_month": month_map[highest_month] if highest_month else None,
                        "lowest_month": month_map[lowest_month] if lowest_month else None
                    } if len(df) >= 365 else {"has_monthly_pattern": False}
                }
                
                return seasonality_analysis
                
            except Exception as decompose_error:
                logger.warning(f"Seasonal decomposition failed: {decompose_error}")
                # Fallback to basic analysis
                return self._analyze_basic_seasonality(df)
        
        except Exception as e:
            logger.error(f"Error analyzing seasonality: {e}")
            return {"error": str(e)}
    
    def _analyze_basic_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback method for basic seasonality analysis"""
        try:
            # Group by day of week
            daily_means = df.groupby(df.index.dayofweek)['price'].mean()
            daily_std = df.groupby(df.index.dayofweek)['price'].std()
            
            # Find the day with highest and lowest average prices
            highest_day = daily_means.idxmax()
            lowest_day = daily_means.idxmin()
            
            # Map day numbers to names
            day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                      4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            
            # Calculate day-of-week variation
            dow_variation = daily_means.std() / daily_means.mean()
            has_seasonality = dow_variation > 0.02  # Lower threshold for basic analysis
            
            return {
                "has_seasonality": has_seasonality,
                "seasonal_strength": dow_variation,
                "weekly_pattern": {
                    "highest_day": day_map[highest_day],
                    "lowest_day": day_map[lowest_day],
                    "day_of_week_variation": daily_std.mean() / daily_means.mean()
                },
                "monthly_pattern": {"has_monthly_pattern": False}
            }
        
        except Exception as e:
            logger.error(f"Error performing basic seasonality analysis: {e}")
            return {"has_seasonality": False, "error": str(e)}
    
    def _analyze_cycles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cyclical components in the data"""
        try:
            # Need sufficient data for cycle analysis
            if len(df) < 90:
                return {"error": "Not enough data for cycle analysis"}
            
            # Calculate returns
            df['returns'] = df['price'].pct_change() * 100
            df_clean = df.dropna()
            
            # Calculate autocorrelation function
            acf_values = acf(df_clean['returns'], nlags=90, fft=True)
            
            # Find significant lags (above 95% confidence interval)
            confidence_level = 1.96 / np.sqrt(len(df_clean))
            significant_lags = [lag for lag, value in enumerate(acf_values) 
                               if abs(value) > confidence_level and lag > 0]
            
            # Find potential cycles
            cycles = {}
            for lag in significant_lags:
                if lag >= 3:  # Only consider cycles of at least 3 days
                    cycles[lag] = acf_values[lag]
            
            # Sort by correlation strength
            sorted_cycles = sorted(cycles.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Identify main cycles
            main_cycles = []
            for lag, corr in sorted_cycles[:3]:  # Top 3 cycles
                if abs(corr) > 0.1:  # Minimum correlation threshold
                    main_cycles.append({
                        "length_days": lag,
                        "correlation": corr,
                        "strength": "strong" if abs(corr) > 0.3 else "moderate" if abs(corr) > 0.2 else "weak"
                    })
            
            # Calculate market cycle phase if we have longer data
            cycle_phase = "unknown"
            if len(df) >= 365:
                # Simple heuristic based on 30-day and 200-day moving averages
                df['ma_30'] = df['price'].rolling(window=30).mean()
                df['ma_200'] = df['price'].rolling(window=200).mean()
                
                current_price = df['price'].iloc[-1]
                ma_30 = df['ma_30'].iloc[-1]
                ma_200 = df['ma_200'].iloc[-1]
                
                # Check price relative to moving averages
                if current_price > ma_30 > ma_200 and current_price / ma_200 > 1.5:
                    cycle_phase = "euphoria"
                elif current_price > ma_30 > ma_200:
                    cycle_phase = "accumulation"
                elif current_price < ma_30 < ma_200 and current_price / ma_200 < 0.7:
                    cycle_phase = "despair"
                elif current_price < ma_30 < ma_200:
                    cycle_phase = "distribution"
                elif ma_30 > ma_200:
                    cycle_phase = "early_uptrend"
                else:
                    cycle_phase = "early_downtrend"
            
            return {
                "has_cycles": len(main_cycles) > 0,
                "main_cycles": main_cycles,
                "cycle_phase": cycle_phase,
                "confidence": "high" if len(main_cycles) >= 2 else "medium" if len(main_cycles) == 1 else "low"
            }
        
        except Exception as e:
            logger.error(f"Error analyzing cycles: {e}")
            return {"has_cycles": False, "error": str(e)}
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility patterns in the data"""
        try:
            # Calculate daily returns
            df['returns'] = df['price'].pct_change() * 100
            df_clean = df.dropna()
            
            # Calculate rolling volatility
            df_clean['volatility_7d'] = df_clean['returns'].rolling(window=7).std()
            df_clean['volatility_30d'] = df_clean['returns'].rolling(window=30).std()
            
            # Drop rows with NaN volatility
            df_vol = df_clean.dropna()
            
            if len(df_vol) < 30:
                return {"error": "Not enough data for volatility analysis"}
            
            # Calculate current and historical volatility
            current_vol_7d = df_vol['volatility_7d'].iloc[-1]
            current_vol_30d = df_vol['volatility_30d'].iloc[-1]
            
            avg_vol_7d = df_vol['volatility_7d'].mean()
            avg_vol_30d = df_vol['volatility_30d'].mean()
            
            max_vol_7d = df_vol['volatility_7d'].max()
            max_vol_30d = df_vol['volatility_30d'].max()
            
            # Normalize volatility (percentage of maximum)
            norm_vol_7d = current_vol_7d / max_vol_7d * 100
            norm_vol_30d = current_vol_30d / max_vol_30d * 100
            
            # Determine volatility level
            vol_level = "medium"
            if current_vol_30d > avg_vol_30d * 1.5:
                vol_level = "high"
            elif current_vol_30d < avg_vol_30d * 0.75:
                vol_level = "low"
            
            # Determine volatility trend
            vol_trend = "stable"
            if current_vol_7d > current_vol_30d * 1.2:
                vol_trend = "increasing"
            elif current_vol_7d < current_vol_30d * 0.8:
                vol_trend = "decreasing"
            
            # Check for volatility clustering
            vol_clustering = False
            vol_autocorr = np.corrcoef(df_vol['volatility_7d'].iloc[:-1], df_vol['volatility_7d'].iloc[1:])[0, 1]
            if vol_autocorr > 0.7:
                vol_clustering = True
            
            # Analyze volatility by day of week
            vol_by_day = df_vol.groupby(df_vol.index.dayofweek)['returns'].std()
            
            # Find highest and lowest volatility days
            highest_vol_day = vol_by_day.idxmax()
            lowest_vol_day = vol_by_day.idxmin()
            
            # Map day numbers to names
            day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                      4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            
            return {
                "current_volatility_7d": current_vol_7d,
                "current_volatility_30d": current_vol_30d,
                "average_volatility_30d": avg_vol_30d,
                "normalized_volatility_30d": norm_vol_30d,
                "volatility_level": vol_level,
                "volatility_trend": vol_trend,
                "volatility_clustering": vol_clustering,
                "highest_volatility_day": day_map[highest_vol_day],
                "lowest_volatility_day": day_map[lowest_vol_day],
                "annualized_volatility": current_vol_30d * np.sqrt(365) if current_vol_30d else None
            }
        
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {"error": str(e)}
    
    def _detect_seasonality_patterns(self, df_daily: pd.DataFrame, 
                                   df_hourly: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Detect specific seasonality patterns"""
        try:
            patterns = {}
            
            # Analyze daily data for weekly patterns
            if not df_daily.empty:
                # Group by day of week
                price_by_day = df_daily.groupby(df_daily.index.dayofweek)['price'].mean()
                returns_by_day = df_daily.groupby(df_daily.index.dayofweek)['price'].pct_change().mean() * 100
                
                # Check for specific day effects
                day_effects = {}
                avg_return = df_daily['price'].pct_change().mean() * 100
                
                for day in range(7):
                    day_return = df_daily[df_daily.index.dayofweek == day]['price'].pct_change().mean() * 100
                    if pd.notna(day_return):
                        effect_size = day_return - avg_return
                        if abs(effect_size) > 0.2:  # Threshold for significance
                            day_effects[day_map[day]] = {
                                "avg_return": day_return,
                                "effect_size": effect_size,
                                "significance": "high" if abs(effect_size) > 0.5 else "medium"
                            }
                
                # Check for monthly patterns
                month_effects = {}
                if len(df_daily) >= 365:  # Need at least a year of data
                    price_by_month = df_daily.groupby(df_daily.index.month)['price'].mean()
                    returns_by_month = df_daily.groupby(df_daily.index.month)['price'].pct_change().mean() * 100
                    
                    for month in range(1, 13):
                        month_return = df_daily[df_daily.index.month == month]['price'].pct_change().mean() * 100
                        if pd.notna(month_return):
                            effect_size = month_return - avg_return
                            if abs(effect_size) > 0.5:  # Higher threshold for monthly
                                month_effects[month_map[month]] = {
                                    "avg_return": month_return,
                                    "effect_size": effect_size,
                                    "significance": "high" if abs(effect_size) > 1 else "medium"
                                }
                
                patterns["day_of_week_effects"] = day_effects
                patterns["month_effects"] = month_effects
            
            # Analyze hourly data for intraday patterns
            if df_hourly is not None and not df_hourly.empty:
                # Group by hour of day
                price_by_hour = df_hourly.groupby(df_hourly.index.hour)['price'].mean()
                returns_by_hour = df_hourly.groupby(df_hourly.index.hour)['price'].pct_change().mean() * 100
                
                # Find highest and lowest hours
                highest_hour = price_by_hour.idxmax()
                lowest_hour = price_by_hour.idxmin()
                
                # Check for specific hour effects
                hour_effects = {}
                avg_hourly_return = df_hourly['price'].pct_change().mean() * 100
                
                for hour in range(24):
                    hour_return = df_hourly[df_hourly.index.hour == hour]['price'].pct_change().mean() * 100
                    if pd.notna(hour_return):
                        effect_size = hour_return - avg_hourly_return
                        if abs(effect_size) > 0.1:  # Threshold for significance
                            hour_effects[f"{hour:02d}:00"] = {
                                "avg_return": hour_return,
                                "effect_size": effect_size,
                                "significance": "high" if abs(effect_size) > 0.2 else "medium"
                            }
                
                patterns["hour_of_day_effects"] = hour_effects
                patterns["peak_hours"] = {
                    "highest_price_hour": f"{highest_hour:02d}:00",
                    "lowest_price_hour": f"{lowest_hour:02d}:00"
                }
            
            return patterns
        
        except Exception as e:
            logger.error(f"Error detecting seasonality patterns: {e}")
            return {"error": str(e)}
    
    def _detect_cyclical_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect specific cyclical patterns"""
        try:
            patterns = {}
            
            # Calculate returns
            df['returns'] = df['price'].pct_change() * 100
            df_clean = df.dropna()
            
            # Calculate autocorrelation
            acf_values = acf(df_clean['returns'], nlags=90, fft=True)
            
            # Find significant cycles
            confidence_level = 1.96 / np.sqrt(len(df_clean))
            significant_lags = [lag for lag, value in enumerate(acf_values) 
                               if abs(value) > confidence_level and lag > 0]
            
            # Check for common cycle lengths
            common_cycles = {
                "weekly": {"length": 7, "detected": False, "strength": 0},
                "biweekly": {"length": 14, "detected": False, "strength": 0},
                "monthly": {"length": 30, "detected": False, "strength": 0},
                "quarterly": {"length": 90, "detected": False, "strength": 0}
            }
            
            # Check each common cycle
            for cycle_name, cycle_data in common_cycles.items():
                length = cycle_data["length"]
                # Check if we have a significant lag near this length
                for lag in significant_lags:
                    if abs(lag - length) <= max(2, length * 0.1):  # Allow some flexibility
                        common_cycles[cycle_name]["detected"] = True
                        common_cycles[cycle_name]["strength"] = abs(acf_values[lag])
                        common_cycles[cycle_name]["exact_length"] = lag
                        break
            
            # Check for short-term momentum (1-3 day autocorrelation)
            short_term_momentum = False
            if 1 in significant_lags or 2 in significant_lags or 3 in significant_lags:
                short_term_momentum = True
                
            # Check for mean reversion (negative autocorrelation at short lags)
            mean_reversion = False
            if acf_values[1] < -confidence_level or acf_values[2] < -confidence_level:
                mean_reversion = True
            
            patterns["common_cycles"] = common_cycles
            patterns["short_term_momentum"] = short_term_momentum
            patterns["mean_reversion"] = mean_reversion
            
            return patterns
        
        except Exception as e:
            logger.error(f"Error detecting cyclical patterns: {e}")
            return {"error": str(e)}
    
    def _detect_volatility_patterns(self, df_daily: pd.DataFrame, 
                                  df_hourly: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Detect specific volatility patterns"""
        try:
            patterns = {}
            
            # Calculate returns and volatility
            df_daily['returns'] = df_daily['price'].pct_change() * 100
            df_daily['volatility'] = df_daily['returns'].rolling(window=7).std()
            df_clean = df_daily.dropna()
            
            if len(df_clean) < 30:
                return {"error": "Not enough data for volatility pattern detection"}
            
            # Check for volatility clustering
            vol_autocorr = np.corrcoef(df_clean['volatility'].iloc[:-1], df_clean['volatility'].iloc[1:])[0, 1]
            patterns["volatility_clustering"] = {
                "detected": vol_autocorr > 0.7,
                "strength": vol_autocorr
            }
            
            # Check for weekend effect on volatility
            weekday_vol = df_clean[df_clean.index.dayofweek < 5]['volatility'].mean()
            weekend_vol = df_clean[df_clean.index.dayofweek >= 5]['volatility'].mean()
            
            if not pd.isna(weekday_vol) and not pd.isna(weekend_vol):
                weekend_effect = (weekend_vol - weekday_vol) / weekday_vol * 100
                patterns["weekend_effect"] = {
                    "detected": abs(weekend_effect) > 10,
                    "weekend_higher": weekend_vol > weekday_vol,
                    "difference_pct": weekend_effect
                }
            
            # Check for volatility seasonality
            vol_by_month = df_clean.groupby(df_clean.index.month)['volatility'].mean()
            if len(vol_by_month) >= 12:
                highest_vol_month = vol_by_month.idxmax()
                lowest_vol_month = vol_by_month.idxmin()
                
                month_var = vol_by_month.std() / vol_by_month.mean()
                
                patterns["monthly_volatility"] = {
                    "seasonal_effect": month_var > 0.1,
                    "highest_month": month_map[highest_vol_month],
                    "lowest_month": month_map[lowest_vol_month],
                    "variation": month_var
                }
            
            # Check for volatility smile (higher vol on weekends)
            vol_by_day = df_clean.groupby(df_clean.index.dayofweek)['volatility'].mean()
            if len(vol_by_day) >= 5:
                midweek_vol = vol_by_day[[1, 2, 3]].mean()  # Tue-Thu
                weekend_vol = vol_by_day[[0, 4, 5, 6]].mean()  # Mon, Fri-Sun
                
                smile_effect = (weekend_vol - midweek_vol) / midweek_vol * 100
                patterns["volatility_smile"] = {
                    "detected": smile_effect > 5,
                    "strength": smile_effect
                }
            
            # Analyze hourly volatility if available
            if df_hourly is not None and not df_hourly.empty:
                df_hourly['returns'] = df_hourly['price'].pct_change() * 100
                df_hourly = df_hourly.dropna()
                
                # Check for intraday volatility patterns
                vol_by_hour = df_hourly.groupby(df_hourly.index.hour)['returns'].std()
                
                if len(vol_by_hour) >= 12:
                    highest_vol_hour = vol_by_hour.idxmax()
                    lowest_vol_hour = vol_by_hour.idxmin()
                    
                    hour_var = vol_by_hour.std() / vol_by_hour.mean()
                    
                    # Check for specific patterns like U-shape (higher vol at open and close)
                    morning_hours = list(range(8, 11))  # 8-10 AM
                    midday_hours = list(range(11, 14))  # 11 AM - 1 PM
                    evening_hours = list(range(14, 17))  # 2-4 PM
                    
                    # Calculate average volatility for each period
                    morning_vol = vol_by_hour[vol_by_hour.index.isin(morning_hours)].mean()
                    midday_vol = vol_by_hour[vol_by_hour.index.isin(midday_hours)].mean()
                    evening_vol = vol_by_hour[vol_by_hour.index.isin(evening_hours)].mean()
                    
                    u_shape = (morning_vol > midday_vol * 1.1) and (evening_vol > midday_vol * 1.1)
                    
                    patterns["intraday_volatility"] = {
                        "highest_hour": f"{highest_vol_hour:02d}:00",
                        "lowest_hour": f"{lowest_vol_hour:02d}:00",
                        "variation": hour_var,
                        "u_shape_detected": u_shape
                    }
            
            return patterns
        
        except Exception as e:
            logger.error(f"Error detecting volatility patterns: {e}")
            return {"error": str(e)}
    
    def _analyze_event_price_impact(self, events: List[Dict[str, Any]], 
                                  price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze price impact of events"""
        try:
            # Initialize result structure
            impact_data = {
                "avg_1d_change": 0,
                "avg_3d_change": 0,
                "avg_7d_change": 0,
                "consistent_direction": False,
                "impact_significance": "low"
            }
            
            # Track price changes for each event
            changes_1d = []
            changes_3d = []
            changes_7d = []
            
            for event in events:
                event_time = event.get("timestamp")
                if not event_time:
                    continue
                
                # Find price at event time
                price_at_event = self._find_price_at_time(event_time, price_data)
                if not price_at_event:
                    continue
                
                # Find prices after event
                price_1d_after = self._find_price_at_time(event_time + timedelta(days=1), price_data)
                price_3d_after = self._find_price_at_time(event_time + timedelta(days=3), price_data)
                price_7d_after = self._find_price_at_time(event_time + timedelta(days=7), price_data)
                
                # Calculate changes
                if price_1d_after:
                    change_1d = self._calculate_change(price_at_event, price_1d_after)
                    changes_1d.append(change_1d)
                
                if price_3d_after:
                    change_3d = self._calculate_change(price_at_event, price_3d_after)
                    changes_3d.append(change_3d)
                
                if price_7d_after:
                    change_7d = self._calculate_change(price_at_event, price_7d_after)
                    changes_7d.append(change_7d)
            
            # Calculate average changes
            if changes_1d:
                impact_data["avg_1d_change"] = sum(changes_1d) / len(changes_1d)
            
            if changes_3d:
                impact_data["avg_3d_change"] = sum(changes_3d) / len(changes_3d)
            
            if changes_7d:
                impact_data["avg_7d_change"] = sum(changes_7d) / len(changes_7d)
            
            # Check for consistency in direction
            if changes_7d:
                positive_count = sum(1 for c in changes_7d if c > 0)
                negative_count = sum(1 for c in changes_7d if c < 0)
                
                consistency = max(positive_count, negative_count) / len(changes_7d)
                impact_data["consistent_direction"] = consistency >= 0.7
                impact_data["direction"] = "positive" if positive_count > negative_count else "negative"
                impact_data["direction_consistency"] = consistency
            
            # Determine significance
            if abs(impact_data["avg_7d_change"]) > 5 and impact_data["consistent_direction"]:
                impact_data["impact_significance"] = "high"
            elif abs(impact_data["avg_7d_change"]) > 2 or impact_data["consistent_direction"]:
                impact_data["impact_significance"] = "medium"
            
            return impact_data
        
        except Exception as e:
            logger.error(f"Error analyzing event price impact: {e}")
            return {"error": str(e)}
    
    def _analyze_event_temporal_distribution(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal distribution of events"""
        try:
            # Extract timestamps
            timestamps = [event.get("timestamp") for event in events if event.get("timestamp")]
            
            if not timestamps:
                return {"error": "No valid timestamps"}
            
            # Convert to datetime if needed
            datetimes = []
            for ts in timestamps:
                if isinstance(ts, str):
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        datetimes.append(dt)
                    except:
                        continue
                else:
                    datetimes.append(ts)
            
            # Sort timestamps
            datetimes.sort()
            
            # Calculate time differences between events
            time_diffs = []
            for i in range(1, len(datetimes)):
                diff = (datetimes[i] - datetimes[i-1]).total_seconds() / (60 * 60 * 24)  # in days
                time_diffs.append(diff)
            
            if not time_diffs:
                return {"error": "Could not calculate time differences"}
            
            # Calculate statistics
            avg_diff = sum(time_diffs) / len(time_diffs)
            median_diff = sorted(time_diffs)[len(time_diffs) // 2]
            std_diff = np.std(time_diffs) if len(time_diffs) > 1 else 0
            
            # Check for clustering
            has_clustering = std_diff > avg_diff
            
            # Analyze by month and day of week
            counts_by_month = {}
            counts_by_day = {}
            
            for dt in datetimes:
                month = dt.month
                day = dt.weekday()
                
                if month not in counts_by_month:
                    counts_by_month[month] = 0
                counts_by_month[month] += 1
                
                if day not in counts_by_day:
                    counts_by_day[day] = 0
                counts_by_day[day] += 1
            
            # Find month and day with most events
            most_common_month = max(counts_by_month.items(), key=lambda x: x[1])[0] if counts_by_month else None
            most_common_day = max(counts_by_day.items(), key=lambda x: x[1])[0] if counts_by_day else None
            
            # Map to names
            month_name = month_map.get(most_common_month, "Unknown") if most_common_month else None
            day_name = day_map.get(most_common_day, "Unknown") if most_common_day else None
            
            return {
                "event_count": len(datetimes),
                "first_event": datetimes[0] if datetimes else None,
                "last_event": datetimes[-1] if datetimes else None,
                "avg_days_between_events": avg_diff,
                "median_days_between_events": median_diff,
                "std_days_between_events": std_diff,
                "has_clustering": has_clustering,
                "most_common_month": month_name,
                "most_common_day": day_name
            }
        
        except Exception as e:
            logger.error(f"Error analyzing event temporal distribution: {e}")
            return {"error": str(e)}
    
    def _identify_most_impactful_event(self, event_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Identify the most impactful event type"""
        try:
            if not event_analysis:
                return {"error": "No event analysis data"}
            
            # Track impact scores
            impact_scores = {}
            
            for event_type, data in event_analysis.items():
                impact_data = data.get("price_impact", {})
                
                # Skip if no impact data
                if not impact_data or "error" in impact_data:
                    continue
                
                # Calculate impact score
                avg_7d_change = impact_data.get("avg_7d_change", 0)
                consistency = impact_data.get("direction_consistency", 0)
                
                # Score is absolute change * consistency
                impact_score = abs(avg_7d_change) * consistency
                
                impact_scores[event_type] = {
                    "score": impact_score,
                    "avg_7d_change": avg_7d_change,
                    "direction": impact_data.get("direction", "neutral"),
                    "consistency": consistency
                }
            
            # Find event with highest impact score
            if impact_scores:
                most_impactful = max(impact_scores.items(), key=lambda x: x[1]["score"])
                event_type = most_impactful[0]
                impact = most_impactful[1]
                
                return {
                    "event_type": event_type,
                    "impact_score": impact["score"],
                    "avg_7d_change": impact["avg_7d_change"],
                    "direction": impact["direction"],
                    "consistency": impact["consistency"]
                }
            else:
                return {"error": "No impact scores calculated"}
        
        except Exception as e:
            logger.error(f"Error identifying most impactful event: {e}")
            return {"error": str(e)}
    
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
    
    def _calculate_pattern_confidence(self, patterns: Dict[str, Any]) -> str:
        """Calculate overall confidence in detected patterns"""
        try:
            # Count significant patterns in each category
            pattern_counts = {}
            
            for category, pattern_data in patterns.items():
                if isinstance(pattern_data, list):
                    pattern_counts[category] = len(pattern_data)
                elif isinstance(pattern_data, dict):
                    # Count significant patterns in nested dictionaries
                    count = 0
                    for key, data in pattern_data.items():
                        if isinstance(data, dict) and data.get("detected", False):
                            count += 1
                        elif isinstance(data, dict) and data.get("significance") in ["high", "medium"]:
                            count += 1
                    pattern_counts[category] = count
            
            # Calculate total pattern count
            total_patterns = sum(pattern_counts.values())
            
            # Determine confidence level
            if total_patterns >= 5:
                return "high"
            elif total_patterns >= 3:
                return "medium"
            elif total_patterns >= 1:
                return "low"
            else:
                return "very_low"
        
        except Exception as e:
            logger.error(f"Error calculating pattern confidence: {e}")
            return "low"

# Global constants for day and month names
day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
          4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

month_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 
            5: 'May', 6: 'June', 7: 'July', 8: 'August', 
            9: 'September', 10: 'October', 11: 'November', 12: 'December'}

# Singleton instance
temporal_analysis = TemporalAnalysis()

# Helper function to get the singleton instance
def get_temporal_analysis():
    return temporal_analysis

# Example usage
if __name__ == "__main__":
    analyzer = get_temporal_analysis()
    # Analyze time series for Bitcoin
    btc_analysis = analyzer.analyze_time_series("BTC")
    print(f"BTC Time Series Analysis: {btc_analysis}")
    # Detect temporal patterns
    btc_patterns = analyzer.detect_temporal_patterns("BTC")
    print(f"BTC Temporal Patterns: {btc_patterns}")