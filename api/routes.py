from fastapi import FastAPI, HTTPException, Depends, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import asyncio
import time
import json
from bson import ObjectId
import re

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

# New endpoint for supported coins with refresh option
@app.get("/api/supported-coins")
async def get_supported_coins_api(refresh: bool = Query(False, description="Force refresh the coin list from Binance API")):
    """
    Get a list of supported cryptocurrency coins.
    
    Args:
        refresh (bool): Force refresh the coin list from Binance API
    
    Returns:
        Dict[str, List[str]]: Dictionary with list of supported coins
    """
    try:
        # Create an instance of TradingConfig to access the property
        trading_config = TradingConfig()
        
        # Get the supported coins, with optional refresh
        if refresh:
            coins = trading_config.get_supported_coins(refresh=True)
        else:
            coins = trading_config.SUPPORTED_COINS
            
        return {"coins": coins}
    except Exception as e:
        logger.error(f"Failed to fetch supported coins: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch supported coins: {str(e)}")

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
    # Use dynamic coin list from TradingConfig
    trading_config = TradingConfig()
    if coin not in trading_config.SUPPORTED_COINS:
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

def _sanitize_market_data(market_summary, news_summary, market_context):
    """
    Sanitize and prepare market data for LLM prompt to ensure accuracy
    """
    # Prepare a clean dictionary with verified data points
    sanitized_data = {
        "coin_price": market_summary.get('current_price', 'N/A'),
        "daily_change": market_summary.get('daily_change_pct', 'N/A'),
        "rsi_1d": market_summary.get('indicators', {}).get('1d', {}).get('rsi', 'N/A'),
        "volume_24h": market_summary.get('volume_24h', 'N/A'),
        "news_sentiment": news_summary.get('sentiment', {}).get('sentiment', 'Neutral'),
        "sentiment_score": news_summary.get('sentiment', {}).get('sentiment_score', 'N/A'),
        "article_count": news_summary.get('sentiment', {}).get('article_count', 0),
        "market_sentiment": market_context.get('market', {}).get('sentiment', {}).get('sentiment', 'Neutral'),
        "recent_market_headline": market_context.get('market', {}).get('recent_headline', 'No recent headline')
    }
    return sanitized_data
def extract_trading_intent(msg: str) -> str:
    """
    Extract trading intent (buy, sell, or hold) from the user's message.
    
    Args:
        msg (str): The user's message
        
    Returns:
        str: The extracted intent ('buy', 'sell', 'hold', or 'general')
    """
    # Normalize message
    msg_lower = msg.lower()
    
    # Check for buying intent
    buy_patterns = [
        r'\bbuy\b', r'\bbuying\b', r'\bpurchase\b', r'\bacquire\b', r'\binvest\b', 
        r'\blong\b', r'\bentry\b', r'\bget in\b', r'\baccumulate\b'
    ]
    for pattern in buy_patterns:
        if re.search(pattern, msg_lower):
            return 'buy'
    
    # Check for selling intent
    sell_patterns = [
        r'\bsell\b', r'\bselling\b', r'\bexit\b', r'\bdump\b', r'\bshort\b', 
        r'\bget out\b', r'\bliquidate\b', r'\boffload\b'
    ]
    for pattern in sell_patterns:
        if re.search(pattern, msg_lower):
            return 'sell'
    
    # Check for holding intent
    hold_patterns = [
        r'\bhold\b', r'\bhodl\b', r'\bkeep\b', r'\bholding\b', r'\bwait\b', 
        r'\bstay\b', r'\bmaintain\b'
    ]
    for pattern in hold_patterns:
        if re.search(pattern, msg_lower):
            return 'hold'
    
    # Default to general analysis if no specific intent detected
    return 'general'

def extract_coin_from_message(msg: str) -> Optional[str]:
    """
    Enhanced function to extract cryptocurrency symbols from user messages.
    Maps common names to symbols and handles edge cases better.
    
    Args:
        msg (str): The user's message
        
    Returns:
        Optional[str]: The extracted coin symbol or None if no match found
    """
    # Normalize message
    msg_lower = msg.lower()
    
    # Get dynamic coin list from TradingConfig
    trading_config = TradingConfig()
    supported_coins = trading_config.SUPPORTED_COINS
    
    # Comprehensive name to symbol mapping (extending the COIN_NAME_MAPPING from frontend)
    name_to_symbol = {
        # Major Cryptocurrencies
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "binance coin": "BNB",
        "bnb": "BNB",
        "solana": "SOL",
        "ripple": "XRP",
        "xrp": "XRP",
        "cardano": "ADA",
        "dogecoin": "DOGE",
        "doge": "DOGE",
        "shiba inu": "SHIB",
        "shib": "SHIB",
        "polkadot": "DOT",
        "polygon": "MATIC",
        "avalanche": "AVAX",
        "chainlink": "LINK",
        "uniswap": "UNI",
        "litecoin": "LTC",
        "cosmos": "ATOM",
        "toncoin": "TON",
        "ton": "TON",
        "near protocol": "NEAR",
        "near": "NEAR",
        "internet computer": "ICP",
        "aptos": "APT",
        "bitcoin cash": "BCH",
        
        # Layer-1 & Layer-2 Solutions
        "fantom": "FTM",
        "algorand": "ALGO",
        "optimism": "OP",
        "arbitrum": "ARB",
        "stacks": "STX",
        "hedera": "HBAR",
        "hbar": "HBAR",
        "ethereum classic": "ETC",
        "flow": "FLOW",
        "multiversx": "EGLD",
        "elrond": "EGLD",
        "harmony": "ONE",
        "celo": "CELO",
        "kava": "KAVA",
        "klaytn": "KLAY",
        "zilliqa": "ZIL",
        "kaspa": "KAS",
        "sei network": "SEI",
        "sei": "SEI",
        "sui": "SUI",
        "tron": "TRX",
        "immutable x": "IMX",
        "immutable": "IMX",
        "astar": "ASTR",
        
        # DeFi Tokens
        "maker": "MKR",
        "aave": "AAVE",
        "curve": "CRV",
        "pancakeswap": "CAKE",
        "cake": "CAKE",
        "compound": "COMP",
        "synthetix": "SNX",
        "1inch": "1INCH",
        "yearn.finance": "YFI",
        "yearn": "YFI",
        "sushiswap": "SUSHI",
        "sushi": "SUSHI",
        "convex finance": "CVX",
        "convex": "CVX",
        "lido dao": "LDO",
        "lido": "LDO",
        "balancer": "BAL",
        "dydx": "DYDX",
        "quant": "QNT",
        "the graph": "GRT",
        "graph": "GRT",
        "vechain": "VET",
        "injective": "INJ",
        
        # Stablecoins
        "tether": "USDT",
        "usd coin": "USDC",
        "binance usd": "BUSD",
        "dai": "DAI",
        "trueusd": "TUSD",
        "first digital usd": "FDUSD",
        
        # Gaming & Metaverse
        "the sandbox": "SAND",
        "sandbox": "SAND",
        "decentraland": "MANA",
        "axie infinity": "AXS",
        "axie": "AXS",
        "enjin coin": "ENJ",
        "enjin": "ENJ",
        "gala games": "GALA",
        "gala": "GALA",
        "illuvium": "ILV",
        "blur": "BLUR",
        "render": "RNDR",
        "chiliz": "CHZ",
        "dusk network": "DUSK",
        "dusk": "DUSK",
        "stepn": "GMT",
        "apecoin": "APE",
        "ape": "APE",
        "thorchain": "RUNE",
        
        # Exchange Tokens
        "crypto.com coin": "CRO",
        "cronos": "CRO",
        "okb": "OKB",
        "kucoin token": "KCS",
        "kucoin": "KCS",
        "gatetoken": "GT",
        "ftx token": "FTT",
        "huobi token": "HT",
        
        # Privacy Coins
        "monero": "XMR",
        "zcash": "ZEC",
        "dash": "DASH",
        "oasis network": "ROSE",
        "oasis": "ROSE",
        
        # Storage & Computing
        "filecoin": "FIL",
        "arweave": "AR",
        
        # Newer & Trending Tokens
        "pyth network": "PYTH",
        "pyth": "PYTH",
        "jito": "JTO",
        "bonk": "BONK",
        "book of meme": "BOME",
        "bome": "BOME",
        "pepe": "PEPE",
        "dogwifhat": "WIF",
        "wif": "WIF",
        "jupiter": "JUP",
        "cyberconnect": "CYBER",
        "cyber": "CYBER",
        "celestia": "TIA",
        "fetch.ai": "FET",
        "fetch": "FET",
        "ordinals": "ORDI",
        "starknet": "STRK",
        "beam": "BEAM",
        "blast": "BLAST",
        "mousepad": "MOUSE",
        "singularitynet": "AGIX",
        "space id": "ID",
        "ace": "ACE",
        
        # Other Significant Coins
        "airswap": "AST",
        "ast": "AST",
        "tezos": "XTZ",
        "eos": "EOS",
        "theta network": "THETA",
        "theta": "THETA",
        "neo": "NEO",
        "iota": "IOTA",
        "stellar": "XLM",
        "0x": "ZRX",
        "basic attention token": "BAT",
        "basic attention": "BAT",
        "bat": "BAT",
        "ravencoin": "RVN",
        "icon": "ICX",
        "ontology": "ONT",
        "waves": "WAVES",
        "digibyte": "DGB",
        "qtum": "QTUM",
        "kusama": "KSM",
        "decred": "DCR",
        "horizen": "ZEN",
        "siacoin": "SC",
        "stargate finance": "STG",
        "stargate": "STG",
        "woo network": "WOO",
        "woo": "WOO",
        "conflux": "CFX",
        "skale": "SKL",
        "mask network": "MASK",
        "mask": "MASK",
        "api3": "API3",
        "omg network": "OMG",
        "omg": "OMG",
        "ethereum name service": "ENS",
        "ens": "ENS",
        "magic": "MAGIC",
        "ankr": "ANKR",
        "ssv network": "SSV",
        "ssv": "SSV",
        "binaryx": "BNX",
        "nem": "XEM",
        "helium": "HNT",
        "swipe": "SXP",
        "linear": "LINA",
        "loopring": "LRC",
        "rocket pool": "RPL",
        "origin protocol": "OGN",
        "origin": "OGN",
        "constitutiondao": "PEOPLE",
        "people": "PEOPLE",
        "pax gold": "PAXG",
        "marlin": "POND",
        "ethereumpow": "ETHW",
        "trust wallet token": "TWT",
        "trust wallet": "TWT",
        "jasmy": "JASMY",
        "jasmycoin": "JASMY",
        "ocean protocol": "OCEAN",
        "ocean": "OCEAN",
        "alpha venture dao": "ALPHA",
        "alpha": "ALPHA",
        "dodo": "DODO",
        "iotex": "IOTX",
        "verge": "XVG",
        "storj": "STORJ",
        "bakerytoken": "BAKE",
        "bakery": "BAKE",
        "reserve rights": "RSR",
        "rsk infrastructure framework": "RIF",
        "certik": "CTK",
        "bounce finance": "AUCTION",
        "bounce": "AUCTION",
        "safepal": "SFP",
        "measurable data token": "MDT",
        "mobox": "MBOX",
        "bella protocol": "BEL",
        "bella": "BEL",
        "wing finance": "WING",
        "wing": "WING",
        "komodo": "KMD",
        "iexec rlc": "RLC",
        "iexec": "RLC",
        "nkn": "NKN",
        "arpa": "ARPA"
    }
    
    # First check for exact symbol mentions (case-insensitive)
    for coin in supported_coins:
        # Check for the exact symbol with word boundaries
        symbol_pattern = r'\b' + coin.lower() + r'\b'
        if re.search(symbol_pattern, msg_lower):
            return coin
            
    # Next check for name mentions in our mapping
    for name, symbol in name_to_symbol.items():
        if name in msg_lower:
            # Verify this coin exists in our supported list or try to find it
            if symbol in supported_coins:
                return symbol
            else:
                # Try to find by fetching directly
                try:
                    # Check if this coin exists on Binance (cached check)
                    exists = symbol in trading_config.SUPPORTED_COINS
                    if exists:
                        return symbol
                except Exception:
                    pass
    
    # No match found
    return None
def format_news_headlines(news_data):
    """
    Format recent news headlines for inclusion in the LLM prompt
    
    Args:
        news_data (dict): Dictionary containing news data including recent articles
        
    Returns:
        str: Formatted string of news headlines
    """
    articles = news_data.get('recent_articles', [])
    if not articles:
        return "No recent significant news available."
    
    headlines = []
    for article in articles[:3]:  # Top 3 articles
        source = article.get('source', 'Unknown')
        title = article.get('title', 'No title')
        headlines.append(f"- {title} ({source})")
    
    return "\n".join(headlines)

@app.post("/analyze")
async def analyze_custom_query(
    query: Dict[str, Any] = Body(..., 
        example={
            "message": "Should I buy Solana now given the recent market conditions?",
            "context": {
                "risk_tolerance": "medium",
                "investment_horizon": "long",
                "portfolio": ["ETH", "SOL", "ADA"]
            }
        }
    )
):
    """
    Comprehensive cryptocurrency trading analysis using multiple data sources
    
    - **message**: User's question or query
    - **context**: Additional context like risk tolerance, investment horizon, portfolio, etc. (Optional)
    """
    try:
        # Extract necessary components
        message = query.get("message", "")
        context = query.get("context", {})
        
        # Validate input
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Extract coin from the message using the improved function
        target_coin = extract_coin_from_message(message)
        
        # Extract trading intent (buy/sell/hold/general)
        trading_intent = extract_trading_intent(message)
        
        # Get recommendation engine instance
        recommendation_engine = get_recommendation_engine()
        
        # Check if context was explicitly provided and log
        if context:
            logger.info(f"User provided context: {context}")
            risk_tolerance = context.get("risk_tolerance", "not specified")
            investment_horizon = context.get("investment_horizon", "not specified")
            portfolio = context.get("portfolio", [])
        else:
            logger.info("No context provided by user")
            risk_tolerance = "not specified"
            investment_horizon = "not specified"
            portfolio = []
        
        # Log analysis request details
        logger.info(f"Analysis query - Message: '{message}', Coin: {target_coin}, Intent: {trading_intent}")
        
        # If a specific coin is found, provide detailed coin analysis
        if target_coin:
            # Fetch comprehensive market data for the target coin
            market_summary = await recommendation_engine.market_data.get_market_summary(target_coin)
            news_summary = recommendation_engine.news_provider.get_coin_news_summary(target_coin)
            market_context = recommendation_engine.news_provider.get_market_context()
            
            # Sanitize market data to ensure accuracy
            sanitized_data = _sanitize_market_data(market_summary, news_summary, market_context)
            
            # Format recent news headlines
            recent_headlines = format_news_headlines(news_summary)
            
            # Prepare system prompt based on trading intent
            intent_prompt = ""
            if trading_intent == 'buy':
                intent_prompt = f"The user is asking about BUYING {target_coin}. Focus your analysis on whether it's a good time to BUY based on the data."
            elif trading_intent == 'sell':
                intent_prompt = f"The user is asking about SELLING {target_coin}. Focus your analysis on whether it's a good time to SELL based on the data."
            elif trading_intent == 'hold':
                intent_prompt = f"The user is asking about HOLDING {target_coin}. Focus your analysis on whether they should continue HOLDING based on the data."
            else:
                intent_prompt = f"The user is asking for a general analysis of {target_coin}. Provide a balanced view of current opportunities and risks."
            
            # Prepare context section only if context was explicitly provided
            context_section = ""
            if context:
                context_section = "USER CONTEXT:"
                if risk_tolerance != "not specified":
                    context_section += f"\n- Risk Tolerance: {risk_tolerance}"
                if investment_horizon != "not specified":
                    context_section += f"\n- Investment Horizon: {investment_horizon}"
                if portfolio:
                    context_section += f"\n- Current Portfolio: {', '.join(portfolio)}"
                context_section += "\n\n"
            
            # Prepare messages for LLM with consistent formatting
            system_prompt = f"""You are a strategic cryptocurrency trading advisor. 
{intent_prompt}

Provide a precise 200-word analysis based STRICTLY on the provided market data:
- Use ONLY the factual data points given
- DO NOT invent or speculate about market conditions
- Focus on interpreting the EXACT data provided
- Avoid making claims not supported by the given information

Your analysis should include:
1. Direct investment recommendation for the requested action ({trading_intent.upper()})
2. Technical indicators interpretation
3. Market sentiment analysis
4. Key insights from the provided data
5. Actionable recommendations"""
            
            user_prompt = f"""Analysis for {target_coin}:

FACTUAL MARKET DATA:
- Current Price: ${sanitized_data['coin_price']}
- 24h Change: {sanitized_data['daily_change']}%
- 24h Volume: ${sanitized_data['volume_24h']}
- 1-Day RSI: {sanitized_data['rsi_1d']}

NEWS & SENTIMENT:
- Sentiment: {sanitized_data['news_sentiment']}
- Sentiment Score: {sanitized_data['sentiment_score']}
- Article Count: {sanitized_data['article_count']}

MARKET CONTEXT:
- Market Sentiment: {sanitized_data['market_sentiment']}
- Recent Market Headline: {sanitized_data['recent_market_headline']}

{context_section}Recent News Headlines:
{recent_headlines}

GENERATE A PRECISE 200-WORD ANALYSIS REGARDING WHETHER TO {trading_intent.upper()} {target_coin}, USING ONLY THE ABOVE INFORMATION."""
        
        # If no specific coin found, similar pattern for general market analysis
        else:
            # Fetch market-wide data
            market_context = recommendation_engine.news_provider.get_market_context()
            
            # Try to get BTC data as a market benchmark
            try:
                btc_summary = await recommendation_engine.market_data.get_market_summary("BTC")
                market_benchmark = f"- Bitcoin Price: ${btc_summary.get('current_price', 'N/A')}\n"
                market_benchmark += f"- Bitcoin 24h Change: {btc_summary.get('daily_change_pct', 'N/A')}%\n"
                market_benchmark += f"- Bitcoin 1-Day RSI: {btc_summary.get('indicators', {}).get('1d', {}).get('rsi', 'N/A')}\n"
            except Exception:
                market_benchmark = "- Market benchmark data unavailable\n"
            
            # Prepare intent-based prompt for general market
            intent_prompt = ""
            if trading_intent == 'buy':
                intent_prompt = "The user is asking about BUYING in the current market. Focus your analysis on whether it's a good time to BUY cryptocurrency based on the data."
            elif trading_intent == 'sell':
                intent_prompt = "The user is asking about SELLING in the current market. Focus your analysis on whether it's a good time to SELL cryptocurrency based on the data."
            elif trading_intent == 'hold':
                intent_prompt = "The user is asking about HOLDING in the current market. Focus your analysis on whether they should continue HOLDING cryptocurrency based on the data."
            else:
                intent_prompt = "The user is asking for a general market analysis. Provide a balanced view of current opportunities and risks."
            
            # Prepare context section only if context was explicitly provided
            context_section = ""
            if context:
                context_section = "USER CONTEXT:"
                if risk_tolerance != "not specified":
                    context_section += f"\n- Risk Tolerance: {risk_tolerance}"
                if investment_horizon != "not specified":
                    context_section += f"\n- Investment Horizon: {investment_horizon}"
                if portfolio:
                    context_section += f"\n- Current Portfolio: {', '.join(portfolio)}"
                context_section += "\n\n"
                
            # Use a more structured prompt for general analysis too
            system_prompt = f"""You are a strategic cryptocurrency market advisor.
{intent_prompt}

Provide a precise 200-word analysis based STRICTLY on the provided market data:
- Use ONLY the factual data points given
- DO NOT invent or speculate about market conditions
- Focus on interpreting the EXACT data provided
- Avoid making claims not supported by the given information

Your analysis should include:
1. Direct investment recommendation for the general market
2. Technical indicators interpretation
3. Market sentiment analysis
4. Key insights from the provided data
5. Actionable recommendations"""
            
            user_prompt = f"""General Market Analysis Query: {message}

FACTUAL MARKET DATA:
{market_benchmark}
{await _get_market_indicators(recommendation_engine.market_data)}

MARKET CONTEXT:
- Market Sentiment: {market_context.get('market', {}).get('sentiment', {}).get('sentiment', 'Neutral')}
- Geopolitical Sentiment: {market_context.get('geopolitical', {}).get('sentiment', {}).get('sentiment', 'Neutral')}
- Regulatory Sentiment: {market_context.get('regulatory', {}).get('sentiment', {}).get('sentiment', 'Neutral')}
- Recent Market Headline: {market_context.get('market', {}).get('recent_headline', 'No recent headline')}

{context_section}Top Performing Cryptocurrencies:
{await _get_top_performers(recommendation_engine.market_data)}

GENERATE A PRECISE 200-WORD ANALYSIS REGARDING THE CRYPTOCURRENCY MARKET, USING ONLY THE ABOVE INFORMATION."""
        
        # Get LLM instance
        llm = recommendation_engine.llm
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate response from LLM
        response = llm.chat_completion(messages, temperature=0.2)
        
        # Extract the response text
        if 'choices' in response and len(response['choices']) > 0:
            analysis_text = response['choices'][0]['message']['content']
            
            # Prepare structured response (with consistent format regardless of coin detection)
            structured_response = {
                "query": message,
                "detected_coin": target_coin if target_coin else "General Market",
                "detected_intent": trading_intent,
                "context": context if context else None,  # Only include context if provided
                "response": analysis_text,
                "timestamp": time.time()
            }
            
            return structured_response
        else:
            raise HTTPException(status_code=500, detail="Failed to generate analysis from LLM")
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error analyzing custom query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _generate_general_market_analysis(message: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a general market analysis when no specific coin is identified
    """
    try:
        # Get recommendation engine instance
        recommendation_engine = get_recommendation_engine()
        
        # Fetch overall market context
        market_context = recommendation_engine.news_provider.get_market_context()
        
        # Prepare messages for DeepSeek LLM
        system_prompt = """You are a comprehensive market analyst for cryptocurrencies.
Your task is to provide an insightful, strategic overview of the current cryptocurrency market.
Use available market data, sentiment analysis, and broader economic context to craft a nuanced response.

Key elements to address:
1. Current market conditions
2. Emerging trends
3. Potential opportunities and risks
4. Strategic considerations for different investment approaches
5. Risk management recommendations"""
        
        # Prepare user prompt with market context
        user_prompt = f"""General Market Analysis Query: {message}

USER CONTEXT:
- Risk Tolerance: {context.get('risk_tolerance', 'not specified')}
- Investment Horizon: {context.get('investment_horizon', 'not specified')}
- Current Portfolio: {', '.join(context.get('portfolio', []))}

MARKET CONTEXT:
Market Sentiment: {market_context.get('market', {}).get('sentiment', {}).get('sentiment', 'Neutral')}
Geopolitical Sentiment: {market_context.get('geopolitical', {}).get('sentiment', {}).get('sentiment', 'Neutral')}
Regulatory Sentiment: {market_context.get('regulatory', {}).get('sentiment', {}).get('sentiment', 'Neutral')}

Top Performing Cryptocurrencies:
{await _get_top_performers(recommendation_engine.market_data)}

Key Market Indicators:
{await _get_market_indicators(recommendation_engine.market_data)}

Please provide a comprehensive market analysis that addresses the query, offering strategic insights, potential opportunities, and risk management advice."""
        
        # Get LLM instance
        llm = recommendation_engine.llm
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate response from LLM
        response = llm.chat_completion(messages, temperature=0.3)
        
        # Extract the response text
        if 'choices' in response and len(response['choices']) > 0:
            analysis_text = response['choices'][0]['message']['content']
            
            # Prepare structured response
            structured_response = {
                "query": message,
                "context": context,
                "response": analysis_text,
                "market_context": market_context,
                "timestamp": time.time()
            }
            
            return structured_response
        else:
            raise HTTPException(status_code=500, detail="Failed to generate market analysis from LLM")
    
    except Exception as e:
        logger.error(f"Error generating general market analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _format_technical_indicators(indicators: Dict[str, Any]) -> str:
    """Format technical indicators for the prompt"""
    result = []
    
    for timeframe, data in indicators.items():
        result.append(f"Timeframe {timeframe}:")
        
        # Add RSI if available
        if 'rsi' in data:
            rsi = data['rsi']
            rsi_interpretation = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
            result.append(f"  - RSI: {rsi:.1f} ({rsi_interpretation})")
        
        # Add moving averages if available
        if 'moving_averages' in data and data['moving_averages']:
            mas = data['moving_averages']
            current_price = mas.get('current_price', 0)
            
            for ma_type, ma_value in mas.items():
                if ma_type != 'current_price' and ma_value:
                    position = "above" if current_price > ma_value else "below"
                    result.append(f"  - {ma_type.upper()}: {ma_value:.2f} (price is {position})")
    
    return "\n".join(result) if result else "No technical indicators available"

def _format_recent_headlines(articles: List[Dict[str, Any]], max_articles: int = 3) -> str:
    """Format recent news headlines"""
    if not articles:
        return "No recent headlines available"
    
    headlines = []
    for article in articles[:max_articles]:
        source = article.get('source', 'Unknown')
        title = article.get('title', 'No title')
        headlines.append(f"  - {title} ({source})")
    
    return "\n" + "\n".join(headlines)

async def _get_top_performers(market_data_provider) -> str:
    """
    Retrieve top performing cryptocurrencies using the market data provider
    """
    try:
        # Get dynamic list of supported coins
        trading_config = TradingConfig()
        top_coins = trading_config.SUPPORTED_COINS
        
        # Fetch market data for each coin
        coin_performances = []
        for coin in top_coins:
            try:
                # Use await for async method
                market_summary = await market_data_provider.get_market_summary(coin)
                daily_change = market_summary.get('daily_change_pct', 0)
                current_price = market_summary.get('current_price', 0)
                
                coin_performances.append({
                    'coin': coin,
                    'change': daily_change,
                    'price': current_price
                })
            except Exception as e:
                logger.warning(f"Error fetching market data for {coin}: {e}")
        
        # Sort by daily change, descending
        sorted_performances = sorted(coin_performances, key=lambda x: x['change'], reverse=True)
        
        # Format the output
        performance_str = []
        for perf in sorted_performances[:5]:  # Top 5 performers
            performance_str.append(
                f"  - {perf['coin']}: Up {perf['change']:.1f}% (${perf['price']:.2f})"
            )
        
        return "\n".join(performance_str) if performance_str else "No performance data available"
    
    except Exception as e:
        logger.error(f"Error getting top performers: {e}")
        return "Unable to retrieve top performers"

async def _get_market_indicators(market_data_provider) -> str:
    """
    Retrieve key market indicators using the market data provider
    """
    try:
        # Use a primary coin like BTC to get market-wide indicators
        market_summary = await market_data_provider.get_market_summary('BTC')
        
        # Get dynamic count of active cryptocurrencies
        trading_config = TradingConfig()
        
        # Prepare market indicators string
        indicators = [
            f"  - Total Crypto Market Cap: ${market_summary.get('total_market_cap', 'N/A')} (±{market_summary.get('market_cap_change_pct', 0):.1f}%)",
            f"  - Bitcoin Dominance: {market_summary.get('bitcoin_dominance', 'N/A')}%",
            f"  - Market Sentiment: {market_summary.get('market_sentiment', 'Neutral')}",
            f"  - Total 24h Volume: ${market_summary.get('volume_24h', 'N/A')}",
            f"  - Number of Active Cryptocurrencies: {len(trading_config.SUPPORTED_COINS)}"
        ]
        
        return "\n".join(indicators)
    
    except Exception as e:
        logger.error(f"Error getting market indicators: {e}")
        return "Unable to retrieve market indicators"

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    # TODO: Add more comprehensive health checks
    return {
        "status": "healthy",
        "timestamp": time.time()
    }