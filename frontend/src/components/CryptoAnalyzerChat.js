import React, { useState, useEffect, useRef } from 'react';
import { Send, Settings } from 'lucide-react';

const COIN_NAME_MAPPING = {
  // Major Cryptocurrencies
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
  
  // Layer-1 & Layer-2 Solutions
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
  
  // DeFi Tokens
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
  
  // Stablecoins
  "tether": "USDT",
  "usd coin": "USDC",
  "binance usd": "BUSD",
  "dai": "DAI",
  "trueusd": "TUSD",
  "first digital usd": "FDUSD",
  
  // Gaming & Metaverse
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
  
  // Exchange Tokens
  "crypto.com coin": "CRO",
  "cronos": "CRO",
  "okb": "OKB",
  "kucoin token": "KCS",
  "kucoin": "KCS",
  "gatetoken": "GT",
  "ftx token": "FTT",
  "huobi token": "HT",
  
  // Privacy Coins
  "monero": "XMR",
  "zcash": "ZEC",
  "dash": "DASH",
  "oasis network": "ROSE",
  "oasis": "ROSE",
  
  // Storage & Computing
  "filecoin": "FIL",
  "arweave": "AR",
  
  // Newer & Trending Tokens
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
  
  // Other Significant Coins
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
};
// Suggestions Component
const Suggestions = ({ onSuggestionSelect }) => {
  const suggestions = [
    "Should I buy Solana now?",
    "What's happening in the crypto market?",
    "Should I enter a long position on ETH?",
    "Is Bitcoin a good short right now?",
    "Should I buy SOL for short term with low risk?",
    "Analyze Cardano's price movement for long-term investment"
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-3 p-4">
      {suggestions.map((suggestion, index) => (
        <button 
          key={index}
          className="bg-gray-900 hover:bg-gray-800 text-left p-3 rounded-lg text-sm text-gray-300 transition-colors"
          onClick={() => onSuggestionSelect(suggestion)}
        >
          {suggestion}
        </button>
      ))}
    </div>
  );
};

// Message formatting helper
const formatMessage = (content) => {
  if (!content) return "No content available";
  
  // Split by numbered sections with asterisks/numbers
  const parts = content.split(/(\d\.\s\*\*.*?\*\*:)/g);
  
  if (parts.length > 1) {
    return parts.map((part, index) => {
      // Check if this is a section header
      if (part.match(/\d\.\s\*\*.*?\*\*:/)) {
        return <div key={index} className="font-semibold mt-2">{part.replace(/\*\*/g, '')}</div>;
      }
      // This is section content
      return <div key={index} className="mb-2">{part}</div>;
    });
  }
  
  // Check for Markdown-style bold text
  if (content.includes('**')) {
    return content.split('\n').map((line, index) => {
      // Replace bold markdown with styled spans
      const processedLine = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      return (
        <div 
          key={index} 
          className="mb-1" 
          dangerouslySetInnerHTML={{ __html: processedLine }}
        />
      );
    });
  }
  
  // If no structured format detected, just return the content with line breaks
  return content.split('\n').map((line, index) => (
    <div key={index} className="mb-1">{line}</div>
  ));
};

// User Context Form Component
// UserContextForm Component with comprehensive coin support
const UserContextForm = ({ userContext, setUserContext, onClose }) => {
  const [localContext, setLocalContext] = useState(userContext);
  const [availableCoins, setAvailableCoins] = useState([]);
  const [isLoadingCoins, setIsLoadingCoins] = useState(false);
  const [coinSearchQuery, setCoinSearchQuery] = useState('');
  
  // Update local state when props change
  useEffect(() => {
    setLocalContext(userContext);
  }, [userContext]);
  
  // Load available coins from the backend
  useEffect(() => {
    const fetchAvailableCoins = async () => {
      setIsLoadingCoins(true);
      try {
        // Get list of supported coins from your backend
        const response = await fetch('/api/supported-coins');
        if (!response.ok) {
          throw new Error('Failed to fetch supported coins');
        }
        
        const data = await response.json();
        
        // If we have coins from API, use them
        if (data.coins && Array.isArray(data.coins) && data.coins.length > 0) {
          setAvailableCoins(data.coins);
          console.log(`Loaded ${data.coins.length} coins from API`);
        } else {
          // Otherwise use our default comprehensive list
          setAvailableCoins(defaultCoinList);
          console.log("Using default coin list");
        }
      } catch (error) {
        console.error('Error fetching supported coins:', error);
        // Fallback to the most popular coins as a default
        setAvailableCoins(defaultCoinList);
        console.log("Using default coin list due to API error");
      } finally {
        setIsLoadingCoins(false);
      }
    };
    
    fetchAvailableCoins();
  }, []);
  
  // Default comprehensive list of the most popular coins
  const defaultCoinList = [
    'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'SHIB', 'DOT', 'MATIC',
    'AVAX', 'LINK', 'UNI', 'LTC', 'ATOM', 'TON', 'NEAR', 'ICP', 'APT', 'BCH',
    'FTM', 'ALGO', 'OP', 'ARB', 'STX', 'HBAR', 'ETC', 'FLOW', 'EGLD', 'ONE',
    'AAVE', 'MKR', 'CRV', 'CAKE', 'COMP', 'SNX', '1INCH', 'YFI', 'SUSHI', 'LDO',
    'SAND', 'MANA', 'AXS', 'ENJ', 'GALA', 'APE', 'RUNE', 'PEPE', 'TIA', 'SUI',
    'FET', 'AST', 'XMR', 'DYDX', 'GRT', 'VET', 'INJ', 'XTZ', 'EOS', 'NEO'
  ];
  
  // Portfolio input field management
  const [portfolioInput, setPortfolioInput] = useState('');
  
  // Filter available coins based on search query
  const filteredCoins = availableCoins.filter(coin => 
    coin.includes(coinSearchQuery.toUpperCase())
  ).slice(0, 20); // Limit to 20 results for better UI performance
  
  // Check if the input might be a coin name rather than a symbol
  const getSymbolFromName = (name) => {
    // Check if the input might be a coin name, using our mapping
    const normalizedInput = name.trim().toLowerCase();
    for (const [coinName, symbol] of Object.entries(COIN_NAME_MAPPING)) {
      if (normalizedInput.includes(coinName)) {
        return symbol;
      }
    }
    return null;
  };
  
  const addPortfolioCoin = (coin = null) => {
    // Use either the selected coin or the input value
    let coinToAdd = coin || portfolioInput.trim().toUpperCase();
    
    // If no coin was passed and the input might be a name rather than a symbol
    if (!coin && coinToAdd.length > 5) {
      const symbolFromName = getSymbolFromName(coinToAdd);
      if (symbolFromName) {
        coinToAdd = symbolFromName;
      }
    }
    
    if (coinToAdd && !localContext.portfolio.includes(coinToAdd)) {
      setLocalContext({
        ...localContext,
        portfolio: [...localContext.portfolio, coinToAdd]
      });
      setPortfolioInput('');
      setCoinSearchQuery('');
    }
  };
  
  const removePortfolioCoin = (coin) => {
    setLocalContext({
      ...localContext,
      portfolio: localContext.portfolio.filter(c => c !== coin)
    });
  };
  
  const handleSave = () => {
    setUserContext(localContext);
    onClose();
  };
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50">
      <div className="bg-gray-900 text-white rounded-lg p-6 max-w-md w-full mx-4 border border-gray-700">
        <h2 className="text-xl font-bold mb-4 text-orange-500">Your Investment Profile</h2>
        
        <div className="mb-4">
          <label className="block text-sm font-medium mb-1 text-gray-300">Risk Tolerance</label>
          <select 
            className="w-full p-2 border border-gray-700 rounded bg-gray-800 text-white"
            value={localContext.risk_tolerance}
            onChange={(e) => setLocalContext({...localContext, risk_tolerance: e.target.value})}
          >
            <option value="low">Low (Conservative)</option>
            <option value="medium">Medium (Balanced)</option>
            <option value="high">High (Aggressive)</option>
          </select>
        </div>
        
        <div className="mb-4">
          <label className="block text-sm font-medium mb-1 text-gray-300">Investment Horizon</label>
          <select 
            className="w-full p-2 border border-gray-700 rounded bg-gray-800 text-white"
            value={localContext.investment_horizon}
            onChange={(e) => setLocalContext({...localContext, investment_horizon: e.target.value})}
          >
            <option value="short">Short-term (less than 1 year)</option>
            <option value="medium">Medium-term (1-3 years)</option>
            <option value="long">Long-term (more than 3 years)</option>
          </select>
        </div>
        
        <div className="mb-4">
          <label className="block text-sm font-medium mb-1 text-gray-300">Your Crypto Portfolio</label>
          
          <div className="relative">
            <input
              type="text"
              placeholder="Search or add coin (e.g., BTC, Bitcoin, Ethereum)"
              className="w-full p-2 border border-gray-700 rounded bg-gray-800 text-white"
              value={coinSearchQuery || portfolioInput}
              onChange={(e) => {
                const value = e.target.value;
                setCoinSearchQuery(value.toUpperCase());
                setPortfolioInput(value);
              }}
              onKeyPress={(e) => e.key === 'Enter' && addPortfolioCoin()}
            />
            
            {coinSearchQuery && filteredCoins.length > 0 && (
              <div className="absolute z-10 mt-1 w-full max-h-40 overflow-y-auto bg-gray-800 border border-gray-700 rounded">
                {filteredCoins.map((coin) => (
                  <div 
                    key={coin}
                    className="p-2 hover:bg-gray-700 cursor-pointer"
                    onClick={() => {
                      addPortfolioCoin(coin);
                    }}
                  >
                    <span className="font-medium">{coin}</span>
                    {/* Show full name if available */}
                    {getCoinFullName(coin) && (
                      <span className="text-xs text-gray-400 ml-2">
                        {getCoinFullName(coin)}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
          
          {isLoadingCoins && (
            <div className="text-sm text-gray-500 mt-1">Loading available coins...</div>
          )}
          
          <div className="flex flex-wrap gap-2 mt-3">
            {localContext.portfolio.map((coin) => (
              <div key={coin} className="bg-gray-800 px-2 py-1 rounded flex items-center border border-gray-700">
                <span className="mr-1">{coin}</span>
                <button 
                  onClick={() => removePortfolioCoin(coin)}
                  className="text-orange-500 text-xs font-bold"
                >
                  ×
                </button>
              </div>
            ))}
            {localContext.portfolio.length === 0 && (
              <span className="text-gray-500 text-sm">No coins added yet</span>
            )}
          </div>
        </div>
        
        {/* Popular coins quick-add section */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-1 text-gray-300">Popular Coins</label>
          <div className="flex flex-wrap gap-2">
            {['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'MATIC', 'AVAX', 'LINK'].map((coin) => (
              <button
                key={coin}
                className="bg-gray-800 hover:bg-gray-700 text-xs px-2 py-1 rounded border border-gray-700"
                onClick={() => {
                  if (!localContext.portfolio.includes(coin)) {
                    setLocalContext({
                      ...localContext,
                      portfolio: [...localContext.portfolio, coin]
                    });
                  }
                }}
              >
                {coin}
              </button>
            ))}
          </div>
        </div>
        
        <div className="flex justify-end gap-2 mt-4">
          <button 
            onClick={onClose}
            className="px-4 py-2 border border-gray-700 rounded text-gray-300 hover:bg-gray-800"
          >
            Cancel
          </button>
          <button 
            onClick={handleSave}
            className="px-4 py-2 bg-orange-600 text-white rounded hover:bg-orange-700"
          >
            Save Profile
          </button>
        </div>
      </div>
    </div>
  );
};

// Helper function to get the full name of a coin from its symbol
const getCoinFullName = (symbol) => {
  // Create a reverse mapping from symbol to name
  const symbolToName = {};
  
  // Populate the reverse mapping based on our COIN_NAME_MAPPING
  Object.entries(COIN_NAME_MAPPING).forEach(([name, sym]) => {
    // For each symbol, store the longest name (most descriptive)
    if (!symbolToName[sym] || name.length > symbolToName[sym].length) {
      symbolToName[sym] = name;
    }
  });
  
  // Mapping of common symbols to full names
  const fullNames = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum',
    'BNB': 'Binance Coin',
    'SOL': 'Solana',
    'XRP': 'XRP (Ripple)',
    'ADA': 'Cardano',
    'DOGE': 'Dogecoin',
    'SHIB': 'Shiba Inu',
    'DOT': 'Polkadot',
    'MATIC': 'Polygon',
    'AVAX': 'Avalanche',
    'LINK': 'Chainlink',
    'UNI': 'Uniswap',
    'LTC': 'Litecoin',
    'ATOM': 'Cosmos',
    'HBAR': 'Hedera',
    'AST': 'AirSwap',
    // Add more mappings as needed
  };
  
  // Use either the reverse-engineered name from COIN_NAME_MAPPING or the hardcoded mapping
  return symbolToName[symbol] ? 
    symbolToName[symbol].charAt(0).toUpperCase() + symbolToName[symbol].slice(1) : 
    fullNames[symbol] || '';
};

// Main Chat Component
const CryptoAnalyzerChat = () => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showContextForm, setShowContextForm] = useState(false);
  const [userContext, setUserContext] = useState({
    risk_tolerance: "medium",
    investment_horizon: "medium",
    portfolio: []
  });
  
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Context extraction function remains the same
  const extractContextFromMessage = (message) => {
    // ... existing logic ...
    const lowerMessage = message.toLowerCase();
    let contextOverride = {};
    
    // Only add risk_tolerance to context if explicitly mentioned in the message
    // Low risk patterns
    if (
      lowerMessage.includes("low risk") || 
      lowerMessage.includes("conservative") || 
      lowerMessage.includes("risk tolerance is low") || 
      lowerMessage.includes("risk tolerance is short") || 
      lowerMessage.includes("risk is low") ||
      lowerMessage.includes("risk is short") ||
      lowerMessage.includes("minimal risk") ||
      lowerMessage.includes("safe investment")
    ) {
      contextOverride.risk_tolerance = "low";
      console.log("Detected LOW risk tolerance from message");
    } 
    // High risk patterns
    else if (
      lowerMessage.includes("high risk") || 
      lowerMessage.includes("aggressive") || 
      lowerMessage.includes("risk tolerance is high") || 
      lowerMessage.includes("risk is high") ||
      lowerMessage.includes("risky investment") ||
      lowerMessage.includes("willing to take risk")
    ) {
      contextOverride.risk_tolerance = "high";
      console.log("Detected HIGH risk tolerance from message");
    } 
    // Medium risk patterns
    else if (
      lowerMessage.includes("medium risk") || 
      lowerMessage.includes("moderate risk") || 
      lowerMessage.includes("balanced") || 
      lowerMessage.includes("risk tolerance is medium") || 
      lowerMessage.includes("risk tolerance is moderate") ||
      lowerMessage.includes("risk is medium") ||
      lowerMessage.includes("risk is moderate")
    ) {
      contextOverride.risk_tolerance = "medium";
      console.log("Detected MEDIUM risk tolerance from message");
    }
    
    // Only add investment_horizon to context if explicitly mentioned in the message
    // Short term patterns
    if (
      lowerMessage.includes("short term") || 
      lowerMessage.includes("short-term") ||
      lowerMessage.includes("short investment") ||
      lowerMessage.includes("quick trade") ||
      lowerMessage.includes("near term") ||
      lowerMessage.includes("immediate") ||
      lowerMessage.includes("day trade") ||
      lowerMessage.includes("swing trade")
    ) {
      contextOverride.investment_horizon = "short";
      console.log("Detected SHORT term horizon from message");
    } 
    // Long term patterns
    else if (
      lowerMessage.includes("long term") || 
      lowerMessage.includes("long-term") ||
      lowerMessage.includes("long investment") ||
      lowerMessage.includes("hold for years") ||
      lowerMessage.includes("hodl") ||
      lowerMessage.includes("invest for the future")
    ) {
      contextOverride.investment_horizon = "long";
      console.log("Detected LONG term horizon from message");
    } 
    // Medium term patterns
    else if (
      lowerMessage.includes("medium term") || 
      lowerMessage.includes("medium-term") ||
      lowerMessage.includes("mid term") ||
      lowerMessage.includes("mid-term") ||
      lowerMessage.includes("intermediate") ||
      lowerMessage.includes("few months")
    ) {
      contextOverride.investment_horizon = "medium";
      console.log("Detected MEDIUM term horizon from message");
    }
    
    console.log("Context extracted from message:", contextOverride);
    return contextOverride;
  };

  // Other utility functions remain the same
  const isFuturesQuery = (queryText) => {
    const futuresPatterns = [
      /\blong position\b/i,
      /\bshort position\b/i,
      /\bgo long\b/i,
      /\bgo short\b/i,
      /\benter a long\b/i,
      /\benter a short\b/i,
      /\bfutures\b/i,
      /\bleveraged\b/i,
      /\blonging\b/i,
      /\bshorting\b/i
    ];
    
    return futuresPatterns.some(pattern => pattern.test(queryText));
  };

  // Comprehensive name-to-symbol mapping for improved user experience


  const extractCoin = (queryText) => {
    /**
     * Enhanced function to extract coin symbols from user messages
     * @param {string} queryText - The user's query text
     * @return {string} The extracted coin symbol or default "BTC"
     */
    // Normalize query text
    const queryLower = queryText.toLowerCase();
    
    // Import the comprehensive coin mapping defined above
    // This is the same mapping we've defined in the COIN_NAME_MAPPING constant
    
    // First check for exact symbol matches (case insensitive)
    for (const [name, symbol] of Object.entries(COIN_NAME_MAPPING)) {
        // Check each name->symbol mapping
        if (queryLower.includes(name)) {
            console.log(`Detected coin ${symbol} from name "${name}"`);
            return symbol;
        }
    }
    
    // Direct symbol detection for common coins
    // This helps catch symbols not in our mapping
    const commonCoins = [
        'BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE', 'AVAX', 'MATIC',
        'LINK', 'UNI', 'LTC', 'ATOM', 'HBAR', 'AST', 'BNB', 'SHIB', 'NEAR',
        'TON', 'SUI', 'TIA', 'FET', 'APT', 'ARB', 'OP', 'IMX', 'AAVE'
    ];
    
    for (const coin of commonCoins) {
        const regex = new RegExp(`\\b${coin.toLowerCase()}\\b`, 'i');
        if (regex.test(queryLower)) {
            console.log(`Detected coin ${coin} from direct symbol mention`);
            return coin;
        }
    }
    
    // If we reach here and still don't have a match, use a more general context check
    // This is for phrases like "current bitcoin price" where the coin is mentioned indirectly
    const contextPatterns = [
        { pattern: /\bbtc\b|\bbitcoin\b/i, symbol: 'BTC' },
        { pattern: /\beth\b|\bethereum\b/i, symbol: 'ETH' },
        { pattern: /\bsol\b|\bsolana\b/i, symbol: 'SOL' },
        { pattern: /\bhbar\b|\bhedera\b/i, symbol: 'HBAR' },
        { pattern: /\bast\b|\bairswap\b/i, symbol: 'AST' },
        { pattern: /\bbnb\b|\bbinance\b/i, symbol: 'BNB' }
    ];
    
    for (const { pattern, symbol } of contextPatterns) {
        if (pattern.test(queryLower)) {
            console.log(`Detected coin ${symbol} from context pattern`);
            return symbol;
        }
    }

    // Check for intent-based coin detection
    // This helps with queries like "should I buy a good defi token"
    const intentBasedPatterns = [
        { intent: /\bdefi\b|\byield\b|\blending\b|\bliquidity\b/i, suggestions: ['AAVE', 'UNI', 'COMP', 'MKR', 'CRV'] },
        { intent: /\bgaming\b|\bmetaverse\b|\bnft\b/i, suggestions: ['SAND', 'MANA', 'AXS', 'ENJ', 'GALA'] },
        { intent: /\blayer\s*2\b|\bscaling\b|\bl2\b/i, suggestions: ['MATIC', 'ARB', 'OP', 'IMX'] },
        { intent: /\bmeme\b|\bmemes\b/i, suggestions: ['DOGE', 'SHIB', 'PEPE', 'BONK', 'WIF'] },
        { intent: /\bstablecoin\b/i, suggestions: ['USDT', 'USDC', 'DAI', 'BUSD'] },
        { intent: /\bsmart\s*contract\b|\bplatform\b/i, suggestions: ['ETH', 'SOL', 'ADA', 'DOT', 'AVAX'] },
        { intent: /\bprivacy\b|\banonymous\b/i, suggestions: ['XMR', 'ZEC', 'DASH'] }
    ];

    for (const { intent, suggestions } of intentBasedPatterns) {
        if (intent.test(queryLower)) {
            const selected = suggestions[0]; // Default to first suggestion
            console.log(`Detected intent for ${selected} based on query context`);
            return selected;
        }
    }

    // If no specific coin detected, default to BTC
    console.log("No specific coin detected, defaulting to BTC");
    return 'BTC';
};

  const extractFuturesDirection = (queryText) => {
    // ... existing function ...
    const queryLower = queryText.toLowerCase();
    
    // Check for long patterns
    if (/long|buy|bull/i.test(queryLower) && !/short|sell|bear/i.test(queryLower)) {
      return 'long';
    }
    
    // Check for short patterns
    if (/short|sell|bear/i.test(queryLower) && !/long|buy|bull/i.test(queryLower)) {
      return 'short';
    }
    
    // Default to "long" if direction not clear
    return 'long';
  };

  const handleSubmit = async (e) => {
    e?.preventDefault();
    
    if (!query.trim()) return;

    // New function to check if query is crypto-related
  const isCryptoRelatedQuery = (queryText) => {
    const cryptoKeywords = [
      'bitcoin', 'crypto', 'blockchain', 'ethereum', 
      'trading', 'market', 'coin', 'investment', 
      'solana', 'xrp', 'buy', 'sell', 'portfolio',
      'futures', 'long', 'short', 'price', 'trend'
    ];
    
    const queryLower = queryText.toLowerCase();
    
    return cryptoKeywords.some(keyword => queryLower.includes(keyword));
  };

  // Check if query is crypto-related
  if (!isCryptoRelatedQuery(query)) {
    // Add a clarification message
    const clarificationMessage = {
      type: 'ai',
      content: `I'm an AI specialized in cryptocurrency and market analysis. Could you rephrase your query to be about crypto markets, trading, or specific cryptocurrencies? 

Some example queries:
- Should I buy Bitcoin?
- What's happening with Ethereum?
- Analyze Solana's price movement
- Give me a futures trading recommendation

I can help you with:
✓ Cryptocurrency analysis
✓ Trading recommendations
✓ Market sentiment
✓ Futures trading insights`,
      coin: 'Assistant',
      intent: 'Clarification',
      actionColor: 'blue',
      timestamp: new Date().toLocaleString()
    };

    // Add user and AI messages
    setMessages(prev => [
      ...prev, 
      { 
        type: 'user', 
        content: query,
        timestamp: new Date().toLocaleString()
      },
      clarificationMessage
    ]);

    // Reset query and loading state
    setQuery('');
    setIsLoading(false);
    return;
  }
    
    console.log("===============================");
    console.log("Processing query:", query);
  
    // Add user message
    const userMessage = { 
      type: 'user', 
      content: query,
      timestamp: new Date().toLocaleString()
    };
    setMessages(prev => [...prev, userMessage]);
  
    // Extract any context information from the message
    const contextOverride = extractContextFromMessage(query);
    
    // Only use the context explicitly mentioned in the message
    let requestContext = {};
    
    // Only if user explicitly mentioned context, send it
    if (Object.keys(contextOverride).length > 0) {
      requestContext = contextOverride;
      console.log("Using message-specific context:", requestContext);
    } else {
      console.log("No context mentioned in message, not sending any context information");
    }
  
    // Merge context for display purposes
    const messageContext = {
      ...userContext,
      ...contextOverride
    };
    
    console.log("Profile context:", userContext);
    console.log("Message context override:", contextOverride);
    console.log("Final context for request:", messageContext);
  
    // Determine if this is a futures trading query
    const isFutures = isFuturesQuery(query);
    let apiEndpoint, requestOptions, messageType;
    
    if (isFutures) {
      // Extract coin and direction for futures trading
      const coin = extractCoin(query);
      const direction = extractFuturesDirection(query);
      
      // Create URL with query parameters
      const params = new URLSearchParams({
        action_type: 'futures',
        position: direction, // Add position parameter
        force_refresh: 'true' // Force refresh to avoid caching issues
      });
      
      apiEndpoint = `/recommendation/${coin}?${params.toString()}`;
      
      requestOptions = {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      };
      
      messageType = direction === 'long' ? 'LONG' : 'SHORT';
      console.log(`Making futures ${direction} request for ${coin}`);
    } else {
      // Standard analysis query
      apiEndpoint = '/analyze';
      
      // Create request body with explicit context
      const requestBody = {
        message: query,
        context: requestContext   // Use the context with message-specific overrides
      };
      
      console.log("Making API call to:", apiEndpoint);
      console.log("Request body:", JSON.stringify(requestBody, null, 2));
      
      requestOptions = {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      };
      messageType = 'analysis';
    }
  
    setIsLoading(true);
    setQuery('');
  
    try {
      const response = await fetch(apiEndpoint, requestOptions);
  
      if (!response.ok) {
        throw new Error('Request failed');
      }
  
      const data = await response.json();
      console.log("API Response:", JSON.stringify(data, null, 2));
      
      // Check if the context in the response matches what we sent
      if (data.context) {
        console.log("Context match check:");
        console.log("- Expected risk_tolerance:", messageContext.risk_tolerance);
        console.log("- Received risk_tolerance:", data.context.risk_tolerance);
        console.log("- Expected investment_horizon:", messageContext.investment_horizon);
        console.log("- Received investment_horizon:", data.context.investment_horizon);
      }
  
      // Format response based on endpoint type
      let aiMessage;
      
      if (isFutures) {
        // Extract coin and direction for futures trading
        const coin = extractCoin(query);
        const direction = extractFuturesDirection(query);
        
        // Format recommendation using the new helper function
        const formatFuturesRecommendation = (data, direction) => {
          // Get the appropriate action label based on direction and API's action recommendation
          let actionLabel;
          let summary;
          let recommendationColor;
          
          // For LONG positions
          if (direction === 'long') {
            if (data.action === 'BUY') {
              actionLabel = 'ENTER LONG';
              summary = `RECOMMENDED to enter a long position on ${data.coin}`;
              recommendationColor = 'green';
            } else if (data.action === 'SELL') {
              actionLabel = 'AVOID LONG';
              summary = `NOT RECOMMENDED to go long on ${data.coin} at this time`;
              recommendationColor = 'yellow';
            } else {
              actionLabel = 'NEUTRAL';
              summary = `NEUTRAL outlook for ${data.coin} long position`;
              recommendationColor = 'gray';
            }
          } 
          // For SHORT positions
          else if (direction === 'short') {
            if (data.action === 'SELL') {
              actionLabel = 'ENTER SHORT';
              summary = `RECOMMENDED to enter a short position on ${data.coin}`;
              recommendationColor = 'red';
            } else if (data.action === 'BUY') {
              actionLabel = 'AVOID SHORT';
              summary = `NOT RECOMMENDED to short ${data.coin} at this time`;
              recommendationColor = 'yellow';
            } else {
              actionLabel = 'NEUTRAL';
              summary = `NEUTRAL outlook for ${data.coin} short position`;
              recommendationColor = 'gray';
            }
          }
          
          // Get explanation from available fields
          const explanation = data.explanation || data.analysis || data.summary || "No detailed analysis available.";
          
          // Format the content for display
          return {
            label: actionLabel,
            content: `**${actionLabel}** - ${summary}\n\n${explanation}`,
            color: recommendationColor
          };
        };
        
        const recommendation = formatFuturesRecommendation(data, direction);
        
        // Format recommendation response
        aiMessage = {
          type: 'ai',
          content: recommendation.content,
          coin: data.coin || extractCoin(query),
          intent: messageType,
          direction: direction,
          action: recommendation.label,
          actionColor: recommendation.color,
          context: messageContext,
          timestamp: new Date().toLocaleString()
        };
      } else {
        // Format analysis response
        aiMessage = {
          type: 'ai',
          content: data.response,
          coin: data.detected_coin || 'Market',
          intent: data.detected_intent || 'Analysis',
          // Use the context from the API response if available, otherwise use our calculated context
          context: data.context || messageContext,
          actionColor: 'blue', 
          timestamp: new Date(data.timestamp * 1000).toLocaleString()
        };
      }
      
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        type: 'error',
        content: 'Sorry, something went wrong. Please try again.',
        timestamp: new Date().toLocaleString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Format context for display
  const formatContext = (context) => {
    if (!context) return "";
    
    const riskMap = {
      "low": "low risk",
      "medium": "medium risk",
      "high": "high risk"
    };
    
    const horizonMap = {
      "short": "short-term",
      "medium": "medium-term",
      "long": "long-term"
    };
    
    return `(${riskMap[context.risk_tolerance] || context.risk_tolerance}, ${horizonMap[context.investment_horizon] || context.investment_horizon})`;
  };
  
  return (
    <div className="flex flex-col h-screen w-full bg-black text-white shadow-lg">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-gray-800">
        <a href="/landing.html" className="text-2xl font-bold text-orange-500 cursor-pointer hover:text-orange-400">AionX</a>
        <div className="flex items-center">
          <button 
            onClick={() => setShowContextForm(true)}
            className="flex items-center text-sm text-gray-400 hover:text-orange-500 mr-4"
          >
            <Settings size={16} className="mr-1" />
            Profile
          </button>
          <div className="text-sm text-gray-500">Powered by DeepSeek</div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-grow overflow-y-auto p-4 space-y-4" style={{backgroundImage: 'radial-gradient(circle at 10% 20%, rgba(50, 50, 50, 0.2) 0%, rgba(20, 20, 20, 0.2) 90%)', backgroundAttachment: 'fixed'}}>
        <div className="max-w-6xl mx-auto w-full">
          {messages.length === 0 ? (
            <>
              <div className="mb-8 text-center">
                <h2 className="text-3xl font-bold mb-2 text-orange-500">Welcome to AionX</h2>
                <p className="text-gray-300 mb-4">
                  Get AI-powered cryptocurrency analysis and trading recommendations
                </p>
                <div className="text-sm text-gray-400 mb-4">
                  <div>Your profile: {userContext.risk_tolerance} risk, {userContext.investment_horizon}-term horizon</div>
                  {userContext.portfolio.length > 0 && (
                    <div>Portfolio: {userContext.portfolio.join(', ')}</div>
                  )}
                </div>
                <p className="text-sm text-gray-500 italic">
                  Tip: You can mention risk tolerance and time horizon in your query, like "give me a recommendation for SOL with low risk and short term"
                </p>
              </div>
              <Suggestions onSuggestionSelect={(suggestion) => {
                setQuery(suggestion);
                handleSubmit({ preventDefault: () => {} });
              }} />
            </>
          ) : (
            // In the messages mapping section of the render method
messages.map((message, index) => (
  <div 
    key={index} 
    className={`flex ${
      message.type === 'user' 
        ? 'justify-end' 
        : 'justify-start'
    }`}
  >
    <div 
      className={`max-w-[85%] p-3 rounded-lg ${
        message.type === 'user'
          ? 'bg-gray-800 border border-gray-700 text-white'
          : message.type === 'error'
          ? 'bg-gray-800 border border-gray-700 text-red-400'
          : 'bg-gray-800 border border-gray-700'
      }`}
    >
      {message.type === 'ai' && (
        <div className="text-sm font-semibold mb-1 text-white">
          {message.coin} {message.intent === 'LONG' || message.intent === 'SHORT' 
            ? `${message.intent} Recommendation` 
            : message.intent && message.intent !== 'general' 
            ? message.intent.toUpperCase() 
            : 'Analysis'}
          {message.context && (
            <span className="text-xs text-gray-500 ml-2">
              {formatContext(message.context)}
            </span>
          )}
        </div>
      )}
      
      <div className="text-sm whitespace-pre-wrap">
        {message.type === 'ai' 
          ? formatMessage(message.content) 
          : message.content}
      </div>
      
      <div className="text-xs text-gray-500 mt-2 text-right">
        {/* {message.timestamp} */}
      </div>
    </div>
  </div>
))
          )}
          
          {/* Loading indicator */}
          {isLoading && (
            <div className="flex justify-center items-center p-4">
              <div className="loader w-8 h-8 border-4 border-gray-700 border-t-orange-500 rounded-full animate-spin"></div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <form 
        onSubmit={handleSubmit} 
        className="p-4 border-t border-gray-800"
      >
        <div className="max-w-6xl mx-auto w-full flex items-center space-x-2">
          <input 
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about cryptocurrency markets or futures positions..."
            disabled={isLoading}
            className="flex-grow p-3 bg-gray-900 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500 border border-gray-700"
          />
          <button 
            type="submit" 
            disabled={isLoading || !query.trim()}
            className="bg-orange-600 text-white p-3 rounded-lg hover:bg-orange-700 transition-colors disabled:opacity-50"
          >
            <Send size={20} />
          </button>
        </div>
      </form>

      {/* User Context Form Modal */}
      {showContextForm && (
        <UserContextForm 
          userContext={userContext}
          setUserContext={setUserContext}
          onClose={() => setShowContextForm(false)}
        />
      )}
    </div>
  );
};

export default CryptoAnalyzerChat;