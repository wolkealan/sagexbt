# DeepSeek Crypto Advisor

A DeepSeek-integrated AI agent that provides real-time cryptocurrency trading recommendations based on technical analysis, news sentiment, and market conditions.

## Features

- **Real-time Market Data**: Fetches and analyzes cryptocurrency market data from major exchanges
- **News Sentiment Analysis**: Monitors crypto news and calculates sentiment scores
- **DeepSeek LLM Integration**: Leverages DeepSeek's language models for intelligent recommendations
- **Retrieval-Augmented Generation**: Enhances responses with up-to-date information
- **API Endpoints**: RESTful API for accessing recommendations and analysis

## Project Structure

```
crypto_advisor/
├── config/               # Configuration settings
├── data/                 # Data providers and sources
├── knowledge/            # Information extraction and processing
├── decision/             # Trading strategies and recommendations
├── deepseek/             # DeepSeek LLM integration
├── api/                  # API endpoints
├── utils/                # Utility functions
├── main.py               # Application entry point
└── README.md             # Documentation
```

## Installation

### Prerequisites

- Python 3.9+
- API keys for DeepSeek, NewsAPI, and crypto exchanges
- Internet connection for real-time data

### Steps

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/deepseek-crypto-advisor.git
   cd deepseek-crypto-advisor
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```
   cp .env.template .env
   # Edit .env with your API keys
   ```

5. Run the application:
   ```
   python main.py
   ```

## API Endpoints

The application provides the following API endpoints:

- `GET /`: Root endpoint, returns basic API information
- `GET /coins`: Get a list of supported cryptocurrencies
- `GET /recommendation/{coin}`: Get trading recommendation for a specific coin
  - Query parameters:
    - `action_type`: "spot" or "futures" (default: "spot")
    - `force_refresh`: Force refresh data (default: false)
- `GET /recommendations`: Get trading recommendations for all supported coins
- `POST /analyze`: Analyze a custom query about cryptocurrency trading
- `GET /health`: Health check endpoint

## Configuration

The application is configured through environment variables in the `.env` file:

### API Keys
- `DEEPSEEK_API_KEY`: Your DeepSeek API key
- `DEEPSEEK_API_BASE`: DeepSeek API base URL
- `NEWS_API_KEY`: Your NewsAPI key
- `BINANCE_API_KEY` and `BINANCE_API_SECRET`: Binance API credentials
- `COINBASE_API_KEY` and `COINBASE_API_SECRET`: Coinbase API credentials
- `TWITTER_API_KEY` and related keys: Twitter API credentials

### Application Settings
- `DEBUG`: Enable debug mode (true/false)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
- `API_HOST` and `API_PORT`: API server settings

### LLM Settings
- `LLM_MODEL`: DeepSeek model to use
- `LLM_TEMPERATURE`: Temperature for LLM responses
- `LLM_MAX_TOKENS`: Maximum tokens for LLM responses

### Data Refresh Intervals
- `MARKET_DATA_REFRESH`: Market data refresh interval (seconds)
- `NEWS_REFRESH`: News data refresh interval (seconds)
- `SOCIAL_SENTIMENT_REFRESH`: Social sentiment refresh interval (seconds)

## Data Sources

The application uses the following data sources:

1. **Cryptocurrency Market Data**:
   - Exchanges: Binance, Coinbase
   - Data: Price, volume, order book, OHLCV

2. **News Data**:
   - NewsAPI.org for crypto and financial news
   - Categorized by: Crypto-specific, general market, geopolitical, regulatory

3. **Technical Indicators**:
   - RSI (Relative Strength Index)
   - Moving Averages (20, 50, 200 periods)
   - More indicators can be added in `market_data.py`

## Extending the System

### Adding New Exchanges
Add new exchange integration in `data/market_data.py` by extending the `initialize_exchanges` method.

### Adding New Technical Indicators
Implement new indicators in `data/market_data.py` by adding calculation methods.

### Adding News Sources
Extend `data/news_provider.py` with additional news API integrations.

### Customizing Recommendations
Modify recommendation strategies in `decision/recommendation_engine.py` and adjust prompt templates in `deepseek/prompt_templates.py`.

## Development

### Testing
Run tests with pytest:
```
pytest
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Disclaimer

This project is for educational and research purposes only. The trading recommendations provided are not financial advice. Always do your own research and consider your risk tolerance before making investment decisions.

## License

[MIT License](LICENSE)

## Acknowledgements

- [DeepSeek AI](https://deepseek.com/) for LLM capabilities
- [CCXT](https://github.com/ccxt/ccxt) for cryptocurrency exchange integration
- [NewsAPI](https://newsapi.org/) for news data
- [FastAPI](https://fastapi.tiangolo.com/) for API framework
- [ChromaDB](https://github.com/chroma-core/chroma) for vector database