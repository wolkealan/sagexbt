import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional
import json

from config.config import APIConfig, DatabaseConfig
from utils.logger import get_logger
from utils.database import get_database
from knowledge.entity_recognition import get_entity_recognition

logger = get_logger("geopolitical_events")

class GeopoliticalEventTracker:
    """Tracks geopolitical events that may impact cryptocurrency markets"""
    
    def __init__(self):
        self.db = get_database()
        self.entity_recognition = get_entity_recognition()
        self.event_cache = {}
        self.last_update = datetime.now() - timedelta(hours=24)  # Force initial update
        self.update_interval = timedelta(hours=6)  # Update every 6 hours
    
    def get_geopolitical_events(self, days: int = 7, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get recent geopolitical events
        
        Args:
            days: Number of days to look back
            force_refresh: Force refresh of event data
            
        Returns:
            List of geopolitical event dictionaries
        """
        try:
            # Check if we need to refresh data
            if force_refresh or datetime.now() - self.last_update > self.update_interval:
                self._update_events()
            
            # Calculate date range
            from_date = datetime.now() - timedelta(days=days)
            
            # Get events from database
            events = self._get_events_from_db(from_date)
            
            logger.info(f"Retrieved {len(events)} geopolitical events from the last {days} days")
            return events
        
        except Exception as e:
            logger.error(f"Error getting geopolitical events: {e}")
            return []
    
    def get_events_by_region(self, region: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get geopolitical events for a specific region
        
        Args:
            region: Region name (e.g., "Middle East", "Europe", "Asia")
            days: Number of days to look back
            
        Returns:
            List of geopolitical event dictionaries for the region
        """
        try:
            # Check if we need to refresh data
            if datetime.now() - self.last_update > self.update_interval:
                self._update_events()
            
            # Calculate date range
            from_date = datetime.now() - timedelta(days=days)
            
            # Get all events
            all_events = self._get_events_from_db(from_date)
            
            # Filter by region
            region_events = [event for event in all_events if self._region_match(event, region)]
            
            logger.info(f"Retrieved {len(region_events)} events for region {region}")
            return region_events
        
        except Exception as e:
            logger.error(f"Error getting events by region {region}: {e}")
            return []
    
    def get_events_by_type(self, event_type: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get geopolitical events of a specific type
        
        Args:
            event_type: Event type (e.g., "conflict", "election", "trade", "regulation")
            days: Number of days to look back
            
        Returns:
            List of geopolitical event dictionaries of the specified type
        """
        try:
            # Check if we need to refresh data
            if datetime.now() - self.last_update > self.update_interval:
                self._update_events()
            
            # Calculate date range
            from_date = datetime.now() - timedelta(days=days)
            
            # Get all events
            all_events = self._get_events_from_db(from_date)
            
            # Filter by type
            type_events = [event for event in all_events if event.get("event_type", "").lower() == event_type.lower()]
            
            logger.info(f"Retrieved {len(type_events)} events of type {event_type}")
            return type_events
        
        except Exception as e:
            logger.error(f"Error getting events by type {event_type}: {e}")
            return []
    
    def get_events_by_impact(self, min_impact_level: str = "medium", days: int = 30) -> List[Dict[str, Any]]:
        """
        Get geopolitical events with minimum impact level
        
        Args:
            min_impact_level: Minimum impact level ("low", "medium", "high", "critical")
            days: Number of days to look back
            
        Returns:
            List of geopolitical event dictionaries with specified minimum impact
        """
        try:
            # Check if we need to refresh data
            if datetime.now() - self.last_update > self.update_interval:
                self._update_events()
            
            # Calculate date range
            from_date = datetime.now() - timedelta(days=days)
            
            # Get all events
            all_events = self._get_events_from_db(from_date)
            
            # Impact level mapping (for comparison)
            impact_levels = {
                "low": 1,
                "medium": 2,
                "high": 3,
                "critical": 4
            }
            
            min_level = impact_levels.get(min_impact_level.lower(), 1)
            
            # Filter by impact level
            impact_events = [
                event for event in all_events 
                if impact_levels.get(event.get("impact_level", "low").lower(), 0) >= min_level
            ]
            
            logger.info(f"Retrieved {len(impact_events)} events with minimum impact level {min_impact_level}")
            return impact_events
        
        except Exception as e:
            logger.error(f"Error getting events by impact level {min_impact_level}: {e}")
            return []
    
    def analyze_event_impact(self, event_id: str) -> Dict[str, Any]:
        """
        Analyze the potential market impact of a specific event
        
        Args:
            event_id: ID of the event to analyze
            
        Returns:
            Dictionary with impact analysis
        """
        try:
            # Get event from database
            event = self._get_event_by_id(event_id)
            
            if not event:
                logger.warning(f"Event with ID {event_id} not found")
                return {"error": f"Event not found: {event_id}"}
            
            # Get related events (same type or region)
            related_events = self._get_related_events(event)
            
            # Analyze potential market impacts
            impact_analysis = self._analyze_market_impact(event, related_events)
            
            logger.info(f"Analyzed impact for event {event_id}")
            return impact_analysis
        
        except Exception as e:
            logger.error(f"Error analyzing event impact for {event_id}: {e}")
            return {"error": str(e)}
    
    def _update_events(self):
        """Update geopolitical events from news sources"""
        try:
            logger.info("Updating geopolitical events")
            
            # Get events from news API
            news_events = self._fetch_geopolitical_news()
            
            # Process and store events
            for event in news_events:
                self._process_and_store_event(event)
            
            # Update last update time
            self.last_update = datetime.now()
            
            logger.info(f"Updated {len(news_events)} geopolitical events")
        
        except Exception as e:
            logger.error(f"Error updating geopolitical events: {e}")
    
    def _fetch_geopolitical_news(self) -> List[Dict[str, Any]]:
        """Fetch geopolitical news from various sources"""
        try:
            events = []
            
            # Use NewsAPI if configured
            if APIConfig.NEWS_API_KEY:
                events.extend(self._fetch_from_news_api())
            
            # Add other news sources here if available
            
            # If no events fetched, use cached data
            if not events and self.event_cache:
                logger.warning("No fresh events found, using cached data")
                return list(self.event_cache.values())
            
            return events
        
        except Exception as e:
            logger.error(f"Error fetching geopolitical news: {e}")
            return []
    
    def _fetch_from_news_api(self) -> List[Dict[str, Any]]:
        """Fetch geopolitical news from NewsAPI"""
        try:
            # Define geopolitical keywords
            geopolitical_keywords = [
                "war", "conflict", "sanctions", "election", "protest", 
                "treaty", "agreement", "international relations", "diplomatic",
                "terrorism", "military", "coup", "trade war", "tariffs",
                "nuclear", "missile", "UN", "NATO", "EU"
            ]
            
            # Format keywords for query
            query = " OR ".join(geopolitical_keywords)
            
            # NewsAPI endpoint
            url = "https://newsapi.org/v2/everything"
            
            # Parameters
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100,
                'apiKey': APIConfig.NEWS_API_KEY
            }
            
            # Make the request
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            articles = data.get('articles', [])
            
            # Process articles into events
            events = []
            for article in articles:
                event = self._convert_article_to_event(article)
                if event:
                    events.append(event)
            
            logger.info(f"Fetched {len(events)} events from NewsAPI")
            return events
        
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
            return []
    
    def _convert_article_to_event(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a news article to a geopolitical event"""
        try:
            # Extract basic information
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')
            source = article.get('source', {}).get('name', 'Unknown')
            url = article.get('url', '')
            published_at = article.get('publishedAt', '')
            
            # Convert published_at to datetime if it's a string
            if isinstance(published_at, str):
                try:
                    published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                except ValueError:
                    published_at = datetime.now()
            
            # Combine text for analysis
            full_text = f"{title}. {description}. {content}"
            
            # Use entity recognition to extract information
            entities = self.entity_recognition.detect_entities(full_text)
            
            # Determine if this is actually a geopolitical event
            if not self._is_geopolitical_event(full_text, entities):
                return None
            
            # Determine event type
            event_type = self._determine_event_type(full_text, entities)
            
            # Determine region
            region = self._determine_region(full_text, entities)
            
            # Determine impact level
            impact_level = self._determine_impact_level(full_text, entities)
            
            # Create event object
            event = {
                'title': title,
                'description': description,
                'source': source,
                'url': url,
                'published_at': published_at,
                'event_type': event_type,
                'region': region,
                'impact_level': impact_level,
                'entities': entities,
                'timestamp': datetime.now()
            }
            
            return event
        
        except Exception as e:
            logger.error(f"Error converting article to event: {e}")
            return None
    
    def _is_geopolitical_event(self, text: str, entities: Dict[str, Any]) -> bool:
        """Determine if the text describes a geopolitical event"""
        try:
            # Define geopolitical keywords
            geopolitical_keywords = [
                "war", "conflict", "sanction", "election", "protest", 
                "treaty", "agreement", "relations", "diplomatic",
                "terrorism", "military", "coup", "trade war", "tariff",
                "nuclear", "missile", "UN", "NATO", "EU"
            ]
            
            # Check if any keywords are in the text
            if any(keyword in text.lower() for keyword in geopolitical_keywords):
                return True
            
            # Check if financial terms are mentioned (to filter out pure finance news)
            financial_terms = entities.get("financial_terms", [])
            if len(financial_terms) > 3:  # If there are too many financial terms, likely not geopolitical
                return False
            
            # Default to false
            return False
        
        except Exception as e:
            logger.error(f"Error determining if text is geopolitical event: {e}")
            return False
    
    def _determine_event_type(self, text: str, entities: Dict[str, Any]) -> str:
        """Determine the type of geopolitical event"""
        try:
            text_lower = text.lower()
            
            # Check for conflict
            if any(word in text_lower for word in ["war", "conflict", "attack", "invasion", "battle", "fighting"]):
                return "conflict"
            
            # Check for diplomacy
            if any(word in text_lower for word in ["treaty", "agreement", "diplomatic", "relations", "talks", "peace"]):
                return "diplomacy"
            
            # Check for elections
            if any(word in text_lower for word in ["election", "vote", "ballot", "campaign", "candidate", "president"]):
                return "election"
            
            # Check for trade
            if any(word in text_lower for word in ["trade", "tariff", "economic", "import", "export", "commerce"]):
                return "trade"
            
            # Check for regulation
            if any(word in text_lower for word in ["regulation", "law", "policy", "ban", "restrict", "compliance"]):
                return "regulation"
            
            # Check for terrorism
            if any(word in text_lower for word in ["terrorism", "terrorist", "attack", "bomb", "extremist"]):
                return "terrorism"
            
            # Default to "other"
            return "other"
        
        except Exception as e:
            logger.error(f"Error determining event type: {e}")
            return "other"
    
    def _determine_region(self, text: str, entities: Dict[str, Any]) -> str:
        """Determine the region of the geopolitical event"""
        try:
            text_lower = text.lower()
            
            # Define regions and their keywords
            regions = {
                "North America": ["united states", "us", "usa", "america", "canada", "mexico"],
                "Europe": ["europe", "eu", "european union", "uk", "britain", "germany", "france", "italy", "spain"],
                "Asia": ["asia", "china", "japan", "india", "south korea", "north korea"],
                "Middle East": ["middle east", "iran", "iraq", "saudi arabia", "israel", "syria", "turkey", "egypt"],
                "Africa": ["africa", "nigeria", "egypt", "south africa", "ethiopia", "kenya"],
                "Latin America": ["latin america", "brazil", "argentina", "colombia", "venezuela", "chile"],
                "Russia": ["russia", "russian", "moscow", "putin"],
                "Global": ["global", "world", "international", "un", "united nations", "g20", "g7"]
            }
            
            # Check each region
            for region, keywords in regions.items():
                if any(keyword in text_lower for keyword in keywords):
                    return region
            
            # Default to "Unknown"
            return "Unknown"
        
        except Exception as e:
            logger.error(f"Error determining event region: {e}")
            return "Unknown"
    
    def _determine_impact_level(self, text: str, entities: Dict[str, Any]) -> str:
        """Determine the impact level of the geopolitical event"""
        try:
            text_lower = text.lower()
            
            # High impact keywords
            high_impact = ["global crisis", "world war", "nuclear", "catastrophic", "major conflict", 
                          "international crisis", "economic collapse", "massive impact"]
            
            # Medium impact keywords
            medium_impact = ["sanctions", "trade war", "conflict", "significant", "important", 
                            "election", "diplomatic crisis", "economic impact"]
            
            # Check for high impact
            if any(keyword in text_lower for keyword in high_impact):
                return "high"
            
            # Check for medium impact
            if any(keyword in text_lower for keyword in medium_impact):
                return "medium"
            
            # Default to "low"
            return "low"
        
        except Exception as e:
            logger.error(f"Error determining event impact level: {e}")
            return "low"
    
    def _process_and_store_event(self, event: Dict[str, Any]) -> bool:
        """Process and store a geopolitical event in the database"""
        try:
            if not event:
                return False
            
            # Generate an event ID based on title and source
            event_title = event.get('title', '')
            event_source = event.get('source', '')
            
            if not event_title or not event_source:
                return False
            
            # Create a simple hash for the event
            import hashlib
            event_id = hashlib.md5(f"{event_title}_{event_source}".encode()).hexdigest()
            
            # Check if event already exists
            existing_event = self._get_event_by_id(event_id)
            if existing_event:
                # Event already exists, no need to store again
                return False
            
            # Add event ID
            event['event_id'] = event_id
            
            # Store in cache
            self.event_cache[event_id] = event
            
            # Store in database
            collection = self.db.get_collection("geopolitical_events")
            collection.insert_one(event)
            
            logger.debug(f"Stored geopolitical event: {event_title}")
            return True
        
        except Exception as e:
            logger.error(f"Error processing and storing event: {e}")
            return False
    
    def _get_events_from_db(self, from_date: datetime) -> List[Dict[str, Any]]:
        """Get geopolitical events from database after a specific date"""
        try:
            # Query for events
            query = {
                "published_at": {"$gte": from_date}
            }
            
            # Get results from MongoDB
            collection = self.db.get_collection("geopolitical_events")
            events = list(collection.find(query).sort("published_at", -1))
            
            # Convert ObjectId to string
            for event in events:
                if '_id' in event:
                    event['_id'] = str(event['_id'])
            
            return events
        
        except Exception as e:
            logger.error(f"Error getting events from database: {e}")
            return []
    
    def _get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific event by its ID"""
        try:
            # Query for the event
            query = {
                "event_id": event_id
            }
            
            # Get result from MongoDB
            collection = self.db.get_collection("geopolitical_events")
            event = collection.find_one(query)
            
            if event and '_id' in event:
                event['_id'] = str(event['_id'])
            
            return event
        
        except Exception as e:
            logger.error(f"Error getting event by ID {event_id}: {e}")
            return None
    
    def _get_related_events(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get events related to the specified event"""
        try:
            event_type = event.get('event_type', '')
            region = event.get('region', '')
            
            # Query for related events
            query = {
                "$or": [
                    {"event_type": event_type},
                    {"region": region}
                ],
                "event_id": {"$ne": event.get('event_id', '')}  # Exclude the current event
            }
            
            # Get results from MongoDB
            collection = self.db.get_collection("geopolitical_events")
            related_events = list(collection.find(query).sort("published_at", -1).limit(10))
            
            # Convert ObjectId to string
            for related_event in related_events:
                if '_id' in related_event:
                    related_event['_id'] = str(related_event['_id'])
            
            return related_events
        
        except Exception as e:
            logger.error(f"Error getting related events: {e}")
            return []
    
    def _analyze_market_impact(self, event: Dict[str, Any], related_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential market impact based on the event and related historical events"""
        try:
            # Simple analysis based on event type and impact level
            event_type = event.get('event_type', '')
            impact_level = event.get('impact_level', 'low')
            
            # Default impact assessments
            crypto_impact = {
                "direction": "neutral",
                "magnitude": "low",
                "affected_assets": [],
                "duration": "short-term"
            }
            
            # Modify based on event type
            if event_type == "conflict":
                crypto_impact["direction"] = "negative" if impact_level in ["high", "critical"] else "neutral"
                crypto_impact["magnitude"] = impact_level
                crypto_impact["affected_assets"] = ["BTC", "ETH", "XRP"]
                crypto_impact["duration"] = "medium-term" if impact_level in ["high", "critical"] else "short-term"
            
            elif event_type == "regulation":
                crypto_impact["direction"] = "negative"
                crypto_impact["magnitude"] = impact_level
                crypto_impact["affected_assets"] = ["All cryptocurrencies"]
                crypto_impact["duration"] = "long-term"
            
            elif event_type == "trade":
                crypto_impact["direction"] = "mixed"
                crypto_impact["magnitude"] = "medium" if impact_level in ["high", "critical"] else "low"
                crypto_impact["affected_assets"] = ["BTC", "ETH"]
                crypto_impact["duration"] = "medium-term"
            
            # Look at historical impact if we have related events
            if related_events:
                historical_impact = self._analyze_historical_impact(related_events)
                
                # Adjust current assessment based on historical patterns
                if historical_impact["confidence"] in ["medium", "high"]:
                    crypto_impact["historical_pattern"] = historical_impact["pattern"]
                    crypto_impact["historical_confidence"] = historical_impact["confidence"]
                    
                    # Update direction if historical pattern is strong
                    if historical_impact["confidence"] == "high":
                        crypto_impact["direction"] = historical_impact["pattern"].get("direction", crypto_impact["direction"])
            
            # Create impact analysis
            impact_analysis = {
                "event_id": event.get('event_id', ''),
                "event_type": event_type,
                "event_region": event.get('region', ''),
                "impact_level": impact_level,
                "crypto_market_impact": crypto_impact,
                "analysis_timestamp": datetime.now()
            }
            
            return impact_analysis
        
        except Exception as e:
            logger.error(f"Error analyzing market impact: {e}")
            return {"error": str(e)}
    
    def _analyze_historical_impact(self, related_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze historical market impact patterns from related events"""
        try:
            # This would ideally use real market data correlated with past events
            # For now, we'll use a simplified approach
            
            # Count impact directions
            direction_counts = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
            
            for event in related_events:
                # In a real implementation, we would look up actual market data following the event
                # Here we're just using a simplified model
                
                event_type = event.get('event_type', '')
                impact_level = event.get('impact_level', 'low')
                
                if event_type == "conflict" and impact_level in ["high", "critical"]:
                    direction_counts["negative"] += 1
                elif event_type == "regulation":
                    direction_counts["negative"] += 1
                elif event_type == "diplomacy" and impact_level in ["high", "medium"]:
                    direction_counts["positive"] += 1
                elif event_type == "trade":
                    direction_counts["mixed"] += 1
                else:
                    direction_counts["neutral"] += 1
            
            # Determine the dominant pattern
            total_events = len(related_events)
            if total_events == 0:
                return {"pattern": {}, "confidence": "low"}
            
            # Find most common direction
            dominant_direction = max(direction_counts.items(), key=lambda x: x[1])
            
            # Calculate confidence
            confidence_ratio = dominant_direction[1] / total_events
            confidence = "low"
            if confidence_ratio >= 0.7:
                confidence = "high"
            elif confidence_ratio >= 0.5:
                confidence = "medium"
            
            return {
                "pattern": {
                    "direction": dominant_direction[0],
                    "prevalence": f"{dominant_direction[1]}/{total_events} events"
                },
                "confidence": confidence
            }
        
        except Exception as e:
            logger.error(f"Error analyzing historical impact: {e}")
            return {"pattern": {}, "confidence": "low"}
    
    def _region_match(self, event: Dict[str, Any], region: str) -> bool:
        """Check if an event matches the specified region"""
        event_region = event.get('region', '')
        
        # Direct match
        if event_region.lower() == region.lower():
            return True
        
        # Check for subregion (e.g., "Europe" matches "Western Europe")
        if region.lower() in event_region.lower() or event_region.lower() in region.lower():
            return True
        
        return False

# Singleton instance
geopolitical_event_tracker = GeopoliticalEventTracker()

# Helper function to get the singleton instance
def get_geopolitical_event_tracker():
    return geopolitical_event_tracker

# Example usage
if __name__ == "__main__":
    tracker = get_geopolitical_event_tracker()
    # Get recent geopolitical events
    events = tracker.get_geopolitical_events(days=7)
    print(f"Recent events: {events}")
    # Get events by region
    europe_events = tracker.get_events_by_region("Europe")
    print(f"Europe events: {europe_events}")