import discord
import aiohttp
import json
import os
import re
from discord.ext import commands
from typing import Dict, Any, Optional

# Configure the bot
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content
bot = commands.Bot(command_prefix='!', intents=intents)

# API URL for your crypto trading advisor
# Change this to use the single API server on port 8000
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")  # Use main API on port 8000
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

async def fetch_from_api(endpoint: str, json_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make an API request to the crypto trading advisor"""
    print(f"Connecting to API: {API_BASE_URL}/{endpoint}")  # Debug print
    async with aiohttp.ClientSession() as session:
        if json_data:
            async with session.post(f"{API_BASE_URL}/{endpoint}", json=json_data) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    print(f"API Error: {response.status} - {error_text}")
                return await response.json()
        else:
            async with session.get(f"{API_BASE_URL}/{endpoint}") as response:
                if response.status >= 400:
                    error_text = await response.text()
                    print(f"API Error: {response.status} - {error_text}")
                return await response.json()

@bot.event
async def on_ready():
    """Event triggered when the bot is ready"""
    print(f"Bot is ready! Logged in as {bot.user}")
    print(f"Using API at {API_BASE_URL}")

@bot.command(name="coins")
async def list_coins(ctx):
    """Command to list supported coins"""
    try:
        coins_data = await fetch_from_api("coins")
        
        # Format the response
        supported_coins = coins_data.get("coins", [])
        coin_count = len(supported_coins)
        
        # Split into multiple messages if needed (Discord has a 2000 char limit)
        chunks = [supported_coins[i:i + 30] for i in range(0, len(supported_coins), 30)]
        
        await ctx.send(f"**Found {coin_count} supported coins:**")
        for chunk in chunks:
            await ctx.send(", ".join(chunk))
            
    except Exception as e:
        await ctx.send(f"Error fetching coins: {str(e)}")

@bot.command(name="analyze")
async def analyze_query(ctx, *, query: str):
    """Command to analyze a cryptocurrency query"""
    # Check if message is not too short
    if len(query) < 5:
        await ctx.send("Please provide a more detailed query. Example: `!analyze Should I buy Bitcoin now?`")
        return
    
    # Extract risk tolerance if mentioned
    risk_tolerance = None
    if re.search(r"risk\s+is\s+low|low\s+risk", query.lower()):
        risk_tolerance = "low"
    elif re.search(r"risk\s+is\s+medium|medium\s+risk", query.lower()):
        risk_tolerance = "medium"
    elif re.search(r"risk\s+is\s+high|high\s+risk", query.lower()):
        risk_tolerance = "high"
    
    # Create request data
    request_data = {
        "message": query,
        "context": {}
    }
    
    # Add risk tolerance if specified
    if risk_tolerance:
        request_data["context"]["risk_tolerance"] = risk_tolerance
    
    try:
        # Send typing indicator while waiting for API response
        async with ctx.typing():
            # Call your API's analyze endpoint
            result = await fetch_from_api("analyze", request_data)
            
            # Extract key information
            response_text = result.get("response", "No analysis available")
            detected_coin = result.get("detected_coin", "General Market")
            detected_intent = result.get("detected_intent", "analysis")
            
            # Format the response
            if len(response_text) > 1900:  # Discord message limit is 2000 chars
                # Split the response into multiple messages
                chunks = [response_text[i:i+1900] for i in range(0, len(response_text), 1900)]
                
                # Send header
                await ctx.send(f"**{detected_coin} {detected_intent.upper()} Analysis**")
                
                # Send response chunks
                for chunk in chunks:
                    await ctx.send(chunk)
            else:
                # Send as a single message
                await ctx.send(f"**{detected_coin} {detected_intent.upper()} Analysis**\n\n{response_text}")
    
    except Exception as e:
        await ctx.send(f"Error analyzing query: {str(e)}")

@bot.command(name="recommend")
async def get_recommendation(ctx, *, query: str):
    """Command to get a trading recommendation from a natural language query"""
    if len(query) < 5:
        await ctx.send("Please provide a more detailed query. Example: `!recommend Should I buy Bitcoin now? Medium risk.`")
        return
    
    try:
        # Extract risk tolerance if mentioned
        risk_tolerance = "medium"  # Default risk level
        if re.search(r"risk\s+is\s+low|low\s+risk", query.lower()):
            risk_tolerance = "low"
        elif re.search(r"risk\s+is\s+medium|medium\s+risk", query.lower()):
            risk_tolerance = "medium"
        elif re.search(r"risk\s+is\s+high|high\s+risk", query.lower()):
            risk_tolerance = "high"
        
        # Extract coin from the message
        coin = extract_coin_from_message(query)
        
        if not coin:
            await ctx.send("I couldn't identify which cryptocurrency you're asking about. Please mention a specific coin like Bitcoin, Ethereum, etc.")
            return
        
        # Determine if it's a long or short query
        position_type = "futures"
        action_type = "spot"
        if re.search(r"\blong\b|\bbuy\b|\benter\s+long\b", query.lower()):
            position_type = "long"
        elif re.search(r"\bshort\b|\bsell\b|\benter\s+short\b", query.lower()):
            position_type = "short"
            
        # Prepare URL with parameters
        params = [f"action_type={action_type}", "force_refresh=true"]
        if position_type in ["long", "short"]:
            params.append(f"position={position_type}")
        if risk_tolerance:
            params.append(f"risk_tolerance={risk_tolerance}")
        
        endpoint = f"recommendation/{coin}?{'&'.join(params)}"
        
        # Send typing indicator while waiting for API response
        async with ctx.typing():
            # Call your API
            result = await fetch_from_api(endpoint)
            
            # Extract key information
            explanation = result.get("explanation", "No recommendation available")
            action = result.get("action", "UNKNOWN")
            confidence = result.get("confidence", "Medium")
            
            # Extract leverage recommendation if present
            leverage_match = re.search(r"Recommended Leverage: (\d+(?:-\d+)?×)", explanation)
            leverage_info = ""
            if leverage_match:
                leverage_info = f"\n**{leverage_match.group(0)}**"
            
            # Format the response
            if len(explanation) > 1900:  # Discord message limit is 2000 chars
                # Split the response into multiple messages
                chunks = [explanation[i:i+1900] for i in range(0, len(explanation), 1900)]
                
                # Send header
                header = f"**{coin} Recommendation: {action}** (Confidence: {confidence}){leverage_info}"
                await ctx.send(header)
                
                # Send response chunks
                for chunk in chunks:
                    await ctx.send(chunk)
            else:
                # Send as a single message
                header = f"**{coin} Recommendation: {action}** (Confidence: {confidence}){leverage_info}\n\n"
                await ctx.send(header + explanation)
    
    except Exception as e:
        await ctx.send(f"Error processing recommendation: {str(e)}")


# Function to extract coin from message
def extract_coin_from_message(msg: str) -> Optional[str]:
    """
    Extract cryptocurrency symbol from message
    """
    # Simplified version - you should use your more comprehensive version from routes.py
    coins_map = {
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "solana": "SOL",
        "cardano": "ADA",
        "ripple": "XRP",
        "polkadot": "DOT",
        "dogecoin": "DOGE",
        "shiba": "SHIB",
        "bnb": "BNB",
        "binance": "BNB"
    }
    
    msg_lower = msg.lower()
    
    # Try to find exact symbol mentions
    for symbol in ["BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "DOGE", "SHIB", "BNB"]:
        if symbol.lower() in msg_lower:
            return symbol
    
    # Try to find coin name mentions
    for name, symbol in coins_map.items():
        if name in msg_lower:
            return symbol
    
    # Default to BTC if no coin found
    return "BTC"


@bot.command(name="crypto_help")
async def bot_help(ctx):
    """Display help information"""
    help_text = """
**Crypto Trading Advisor Bot Commands**

`!coins` - List all supported cryptocurrency coins

`!analyze <query>` - Analyze a cryptocurrency query
Example: `!analyze Should I buy Ethereum now with medium risk?`

`!recommend <query>` - Get a recommendation based on your query
Examples:
- `!recommend Should I enter a long position on Bitcoin now? Low risk.`
- `!recommend Is it a good time to short Solana? High risk.`
- `!recommend Should I buy Ethereum?`

`!crypto_help` - Show this help message
"""
    await ctx.send(help_text)

# Run the bot
def run_discord_bot():
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token or token == "your_discord_bot_token_here":
        print("ERROR: No valid Discord bot token found. Please set DISCORD_BOT_TOKEN in your .env file.")
        return
    print(f"Starting Discord bot with API URL: {API_BASE_URL}")
    bot.run(token)

if __name__ == "__main__":
    run_discord_bot()