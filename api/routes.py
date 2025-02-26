from fastapi import FastAPI, HTTPException, Depends, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import asyncio
import time
import json
from bson import ObjectId

from config.config import TradingConfig, AppConfig
from utils.logger import get_api_logger
from decision.recommendation_engine import get_recommendation_engine

logger = get_api_logger()

# Create custom JSON encoder to handle ObjectId
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

def convert_objectid_to_str(obj):
    """Recursively convert ObjectId to string in nested dictionaries and lists"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_objectid_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid_to_str(v) for v in obj]
    return obj

# Create FastAPI app
app = FastAPI(
    title="Crypto Trading Advisor API",
    description="API for getting AI-powered cryptocurrency trading recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production to be more restrictive
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get recommendation engine instance
recommendation_engine = get_recommendation_engine()

@app.get("/")
async def root():
    """Root endpoint, returns basic API information"""
    return {
        "name": "Crypto Trading Advisor API",
        "version": "1.0.0",
        "status": "online",
        "timestamp": time.time()
    }

@app.get("/coins")
async def get_supported_coins():
    """Get list of supported cryptocurrency coins"""
    return {
        "coins": TradingConfig.SUPPORTED_COINS,
        "count": len(TradingConfig.SUPPORTED_COINS)
    }

@app.get("/recommendation/{coin}")
async def get_recommendation(
    coin: str,
    action_type: str = Query("spot", enum=["spot", "futures"]),
    force_refresh: bool = Query(False, description="Force refresh data and recommendation")
):
    """
    Get trading recommendation for a specific coin
    
    - **coin**: Cryptocurrency symbol (e.g., BTC, ETH)
    - **action_type**: Type of trading (spot or futures)
    - **force_refresh**: Force refresh data and recommendation
    """
    # Validate coin
    coin = coin.upper()
    if coin not in TradingConfig.SUPPORTED_COINS:
        raise HTTPException(status_code=404, detail=f"Coin {coin} not supported")
    
    try:
        # Check if we have a cached recommendation
        if not force_refresh:
            cached_rec = recommendation_engine.get_cached_recommendation(coin, action_type)
            if cached_rec:
                logger.info(f"Returning cached recommendation for {coin}")
                # Convert any ObjectId to string
                cached_rec = convert_objectid_to_str(cached_rec)
                return cached_rec
        
        # Generate new recommendation
        recommendation = await recommendation_engine.generate_recommendation(
            coin=coin,
            action_type=action_type,
            force_refresh=force_refresh
        )
        
        # Convert any ObjectId to string
        recommendation = convert_objectid_to_str(recommendation)
        return recommendation
    
    except Exception as e:
        logger.error(f"Error generating recommendation for {coin}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations")
async def get_all_recommendations(
    action_type: str = Query("spot", enum=["spot", "futures"]),
    force_refresh: bool = Query(False, description="Force refresh all recommendations")
):
    """
    Get trading recommendations for all supported coins
    
    - **action_type**: Type of trading (spot or futures)
    - **force_refresh**: Force refresh all data and recommendations
    """
    try:
        recommendations = await recommendation_engine.generate_all_recommendations(action_type)
        
        # Prepare summary for easier consumption
        summary = {
            "recommendations": [],
            "timestamp": time.time(),
            "count": len(recommendations)
        }
        
        # Add simplified recommendation data
        for coin, rec in recommendations.items():
            # Convert any ObjectId to string
            rec = convert_objectid_to_str(rec)
            
            summary["recommendations"].append({
                "coin": coin,
                "action": rec.get("action", "HOLD"),
                "confidence": rec.get("confidence", "Low"),
                "price": rec.get("context", {}).get("market_data", {}).get("price", 0)
            })
        
        return summary
    
    except Exception as e:
        logger.error(f"Error generating all recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_custom_query(
    query: Dict[str, Any] = Body(..., 
        example={
            "message": "Should I buy Bitcoin now given the recent market conditions?",
            "context": {
                "risk_tolerance": "medium",
                "investment_horizon": "long",
                "portfolio": ["ETH", "SOL", "ADA"]
            }
        }
    )
):
    """
    Analyze a custom query about cryptocurrency trading
    
    - **message**: User's question or query
    - **context**: Additional context like risk tolerance, investment horizon, etc.
    """
    try:
        message = query.get("message", "")
        context = query.get("context", {})
        
        # TODO: Implement custom query analysis using DeepSeek LLM
        # This is a placeholder response
        response = {
            "response": "This endpoint is not fully implemented yet. Please use /recommendation/{coin} instead.",
            "timestamp": time.time()
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error analyzing custom query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    # TODO: Add more comprehensive health checks
    return {
        "status": "healthy",
        "timestamp": time.time()
    }