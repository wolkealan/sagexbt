import React, { useState, useEffect, useRef } from 'react';
import { Send, Settings } from 'lucide-react';

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
const UserContextForm = ({ userContext, setUserContext, onClose }) => {
  const [localContext, setLocalContext] = useState(userContext);
  
  // Update local state when props change
  useEffect(() => {
    setLocalContext(userContext);
  }, [userContext]);
  
  // Portfolio input field management
  const [portfolioInput, setPortfolioInput] = useState('');
  
  const addPortfolioCoin = () => {
    if (portfolioInput.trim() && !localContext.portfolio.includes(portfolioInput.trim().toUpperCase())) {
      setLocalContext({
        ...localContext,
        portfolio: [...localContext.portfolio, portfolioInput.trim().toUpperCase()]
      });
      setPortfolioInput('');
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
          <div className="flex mb-2">
            <input
              type="text"
              placeholder="Add coin (e.g., BTC)"
              className="flex-grow p-2 border border-gray-700 rounded-l bg-gray-800 text-white"
              value={portfolioInput}
              onChange={(e) => setPortfolioInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && addPortfolioCoin()}
            />
            <button 
              onClick={addPortfolioCoin}
              className="bg-orange-600 text-white px-3 rounded-r hover:bg-orange-700"
            >
              Add
            </button>
          </div>
          
          <div className="flex flex-wrap gap-2">
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

  const extractCoin = (queryText) => {
    // ... existing function ...
    const coins = ['BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE', 'AVAX', 'MATIC'];
    
    // Check for exact matches
    for (const coin of coins) {
      if (queryText.toUpperCase().includes(coin)) {
        return coin;
      }
    }
    
    // Check for common names
    const coinNames = {
      'bitcoin': 'BTC',
      'ethereum': 'ETH',
      'solana': 'SOL',
      'cardano': 'ADA',
      'ripple': 'XRP',
      'polkadot': 'DOT',
      'dogecoin': 'DOGE',
      'avalanche': 'AVAX',
      'polygon': 'MATIC'
    };
    
    const queryLower = queryText.toLowerCase();
    for (const [name, symbol] of Object.entries(coinNames)) {
      if (queryLower.includes(name)) {
        return symbol;
      }
    }
    
    // Default to BTC if no coin detected
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
        <a href="/landing.html" className="text-2xl font-bold text-orange-500 cursor-pointer hover:text-orange-400">SageXbt</a>
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
                <h2 className="text-3xl font-bold mb-2 text-orange-500">Welcome to SageXbt</h2>
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
                      ? 'bg-orange-600 text-white'
                      : message.type === 'error'
                      ? 'bg-red-900 text-red-100'
                      : message.actionColor === 'green'
                      ? 'bg-green-900 border border-green-700'
                      : message.actionColor === 'red'
                      ? 'bg-red-900 border border-red-700'
                      : message.actionColor === 'yellow'
                      ? 'bg-yellow-900 border border-yellow-700'
                      : message.actionColor === 'gray'
                      ? 'bg-gray-800 border border-gray-600'
                      : 'bg-gray-800 border border-gray-700'
                  }`}
                >
                  {message.type === 'ai' && (
                    <div className={`text-sm font-semibold mb-1 ${
                      message.actionColor === 'green'
                        ? 'text-green-400' 
                        : message.actionColor === 'red'
                        ? 'text-red-400'
                        : message.actionColor === 'yellow'
                        ? 'text-yellow-400'
                        : message.actionColor === 'gray'
                        ? 'text-gray-400'
                        : 'text-orange-400'
                    }`}>
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
                    {message.timestamp}
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