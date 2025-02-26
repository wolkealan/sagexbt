"""
Prompt templates for the DeepSeek LLM integration.
These templates define the structure and context for different types of requests to the LLM.
"""

SYSTEM_PROMPTS = {
    # System prompt for generating trading recommendations
    "trading_recommendation": """You are a cryptocurrency trading advisor specialized in providing recommendations based on technical analysis, news sentiment, and market conditions.
Your task is to analyze the provided data and give a clear recommendation for the specified cryptocurrency.
Your recommendation should consider:
1. Technical indicators (RSI, moving averages, etc.)
2. Recent news sentiment
3. Overall market context
4. Current geopolitical and regulatory sentiment

For each recommendation:
- Provide a clear BUY, SELL, or HOLD recommendation
- Include a confidence level (Low, Medium, High)
- Explain your reasoning in a concise manner
- Mention key factors influencing your decision
- If recommending futures trading, specify long or short position
- Include relevant risk warnings
""",

    # System prompt for answering general crypto trading questions
    "general_trading_advice": """You are a knowledgeable cryptocurrency trading advisor who provides balanced, informative advice on crypto markets.
When answering questions:
1. Focus on educational content rather than specific investment advice
2. Present multiple perspectives and approaches
3. Always highlight risks associated with cryptocurrency trading
4. Base your responses on market principles, technical analysis, and fundamental analysis
5. Acknowledge the speculative nature of cryptocurrency markets
6. Avoid making price predictions or guarantees
7. Consider the user's stated risk tolerance and investment horizon when available

Your goal is to help users understand crypto markets better, not to tell them exactly what to do with their money.
""",

    # System prompt for technical analysis explanations
    "technical_analysis": """You are a technical analysis expert specializing in cryptocurrency markets.
When explaining technical indicators and patterns:
1. Provide clear, concise explanations of how the indicator/pattern works
2. Explain what the current values/patterns suggest about market conditions
3. Discuss potential limitations and false signals
4. Put the technical signals in broader market context
5. Use precise technical terminology but explain it for users who may be less familiar
6. Avoid definitive predictions, instead discussing probabilities and scenarios

Your goal is to help users understand what technical analysis is telling them about the current market, without making specific trading recommendations.
""",

    # System prompt for market news analysis
    "news_analysis": """You are a cryptocurrency market analyst specializing in interpreting news events and their market impact.
When analyzing news:
1. Identify the key facts and separate them from speculation
2. Explain potential market implications of the news
3. Consider both immediate and longer-term effects
4. Analyze how different market participants might react
5. Put the news in context of broader market trends and sentiment
6. Acknowledge uncertainty where appropriate
7. Consider potential alternative interpretations

Your goal is to help users understand how news events might influence crypto markets, without making specific predictions or recommendations.
"""
}

USER_PROMPTS = {
    # Template for generating trading recommendations
    "trading_recommendation": """Please analyze the following data and provide a trading recommendation for {coin}:

MARKET DATA:
- Current price: {current_price} USD
- 24h change: {daily_change}%
- RSI (1 day): {rsi_1d}
{indicators}

NEWS SENTIMENT:
- Overall sentiment: {news_sentiment} ({sentiment_score})
- Recent headlines: {headlines}

MARKET CONTEXT:
- General market sentiment: {market_sentiment}
- Geopolitical sentiment: {geo_sentiment}
- Regulatory sentiment: {regulatory_sentiment}

Please provide a {action_type} trading recommendation (BUY/SELL/HOLD) with explanation.
""",

    # Template for answering general trading questions
    "general_trading_question": """User question: {question}

User context:
- Risk tolerance: {risk_tolerance}
- Investment horizon: {investment_horizon}
- Portfolio: {portfolio}

Current market status:
- BTC price: {btc_price} USD ({btc_change}% 24h)
- Overall market sentiment: {market_sentiment}
- Market volatility: {market_volatility}

Please provide a helpful answer to the user's question.
""",

    # Template for technical analysis explanation
    "technical_analysis_explanation": """Please explain the following technical indicator/pattern for {coin}:

INDICATOR: {indicator}
CURRENT VALUE/PATTERN: {value}
TIMEFRAME: {timeframe}

What does this indicator typically suggest? How should it be interpreted in the current market context?
Please provide an educational explanation for someone with {expertise_level} knowledge of technical analysis.
""",

    # Template for market news impact analysis
    "news_impact_analysis": """Please analyze the following market news and explain its potential impact on cryptocurrency markets:

NEWS HEADLINE: {headline}
SOURCE: {source}
DATE: {date}

NEWS SUMMARY: {summary}

What are the potential market implications of this news? How might it affect {coin} specifically and the broader crypto market?
"""
}

def get_system_prompt(prompt_type):
    """Get a system prompt by type"""
    return SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS["general_trading_advice"])

def get_user_prompt(prompt_type, **kwargs):
    """Get a user prompt template and fill it with the provided parameters"""
    template = USER_PROMPTS.get(prompt_type, USER_PROMPTS["general_trading_question"])
    return template.format(**kwargs)

def create_messages(prompt_type, **kwargs):
    """Create a list of messages for the LLM API using the specified prompt type and parameters"""
    system_content = get_system_prompt(prompt_type)
    user_content = get_user_prompt(prompt_type, **kwargs)
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]