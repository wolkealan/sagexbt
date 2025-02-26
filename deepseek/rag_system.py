import os
import json
import time
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from datetime import datetime, timedelta

from config.config import AppConfig, BASE_DIR
from utils.logger import get_logger
from deepseek.llm_interface import get_deepseek_llm

logger = get_logger("rag_system")

class RAGSystem:
    """
    Retrieval-Augmented Generation system for enhancing LLM responses 
    with up-to-date information from the knowledge base
    """
    
    def __init__(self):
        self.llm = get_deepseek_llm()
        self.vector_db_path = AppConfig.VECTOR_DB_PATH
        self.ensure_vector_db_dir()
        self.client = chromadb.PersistentClient(path=self.vector_db_path)
        self.collections = {}
        self.initialize_collections()
    
    def ensure_vector_db_dir(self):
        """Ensure vector database directory exists"""
        if not os.path.exists(self.vector_db_path):
            os.makedirs(self.vector_db_path)
            logger.info(f"Created vector database directory at {self.vector_db_path}")
    
    def initialize_collections(self):
        """Initialize vector database collections"""
        try:
            # Collection for market knowledge
            self.collections["market_knowledge"] = self.client.get_or_create_collection(
                name="market_knowledge",
                metadata={"description": "General cryptocurrency market knowledge"}
            )
            
            # Collection for news articles
            self.collections["news"] = self.client.get_or_create_collection(
                name="news",
                metadata={"description": "Cryptocurrency news articles"}
            )
            
            # Collection for technical analysis knowledge
            self.collections["technical_analysis"] = self.client.get_or_create_collection(
                name="technical_analysis",
                metadata={"description": "Technical analysis information for cryptocurrencies"}
            )
            
            # Collection for trading strategies
            self.collections["trading_strategies"] = self.client.get_or_create_collection(
                name="trading_strategies",
                metadata={"description": "Cryptocurrency trading strategies"}
            )
            
            logger.info("Initialized vector database collections")
            
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            raise
    
    def add_document(self, collection_name: str, document: Dict[str, Any], 
                     document_id: Optional[str] = None) -> bool:
        """
        Add a document to a collection
        
        Args:
            collection_name: Name of the collection
            document: Document to add
            document_id: Optional document ID
            
        Returns:
            Success status
        """
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                logger.error(f"Collection {collection_name} not found")
                return False
            
            # Extract text content and metadata
            content = document.get("content", "")
            metadata = document.get("metadata", {})
            
            # Generate ID if not provided
            doc_id = document_id or f"{collection_name}_{int(time.time())}_{len(content) % 1000}"
            
            # Add to collection
            collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Added document to {collection_name} with ID {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to {collection_name}: {e}")
            return False
    
    def query_collection(self, collection_name: str, query: str, 
                         n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query a collection for relevant documents
        
        Args:
            collection_name: Name of the collection
            query: Query string
            n_results: Number of results to return
            
        Returns:
            List of relevant documents
        """
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                logger.error(f"Collection {collection_name} not found")
                return []
            
            # Query the collection
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Process results
            documents = []
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if i < len(results['metadatas'][0]) else {},
                    "id": results['ids'][0][i] if i < len(results['ids'][0]) else f"result_{i}"
                })
            
            logger.info(f"Query '{query}' returned {len(documents)} results from {collection_name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error querying collection {collection_name}: {e}")
            return []
    
    def retrieve_relevant_context(self, query: str, collections: List[str] = None) -> str:
        """
        Retrieve relevant context from collections based on the query
        
        Args:
            query: User query
            collections: Collections to search (default: all)
            
        Returns:
            Concatenated context string
        """
        if not collections:
            collections = list(self.collections.keys())
        
        all_results = []
        for collection_name in collections:
            results = self.query_collection(collection_name, query)
            all_results.extend(results)
        
        # Sort results by relevance (currently just taking all)
        # In a more sophisticated implementation, we'd rank by semantic similarity
        
        # Combine into context
        context = ""
        for i, result in enumerate(all_results):
            context += f"\nINFORMATION {i+1}:\n"
            context += result["content"] + "\n"
            context += f"Source: {result['metadata'].get('source', 'Unknown')}, "
            context += f"Date: {result['metadata'].get('date', 'Unknown')}\n"
        
        if not context:
            context = "No relevant information found in the knowledge base."
        
        return context
    
    def generate_augmented_response(self, query: str, system_prompt: str) -> Dict[str, Any]:
        """
        Generate a response augmented with retrieved context
        
        Args:
            query: User query
            system_prompt: System prompt for the LLM
            
        Returns:
            Response dictionary
        """
        try:
            # Retrieve relevant context
            context = self.retrieve_relevant_context(query)
            
            # Create augmented system prompt
            augmented_system_prompt = (
                system_prompt + 
                "\n\nThe following information from our knowledge base may be helpful:\n" +
                context
            )
            
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": augmented_system_prompt},
                {"role": "user", "content": query}
            ]
            
            # Get response from LLM
            response = self.llm.chat_completion(messages)
            
            # Extract response content
            if 'choices' in response and len(response['choices']) > 0:
                response_content = response['choices'][0]['message']['content']
            else:
                response_content = "Unable to generate a response."
            
            return {
                "query": query,
                "response": response_content,
                "sources": len(context.split("INFORMATION")) - 1,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error generating augmented response: {e}")
            return {
                "query": query,
                "response": f"Error generating response: {str(e)}",
                "sources": 0,
                "timestamp": time.time(),
                "error": str(e)
            }
    
    def index_news_article(self, article: Dict[str, Any]) -> bool:
        """
        Index a news article in the vector database
        
        Args:
            article: News article dictionary
            
        Returns:
            Success status
        """
        try:
            # Extract article information
            title = article.get("title", "")
            description = article.get("description", "")
            content = article.get("content", "")
            source = article.get("source", {}).get("name", "Unknown")
            url = article.get("url", "")
            published_at = article.get("publishedAt", "")
            
            # Combine into document content
            document_content = f"TITLE: {title}\n\nDESCRIPTION: {description}\n\nCONTENT: {content}"
            
            # Create metadata
            metadata = {
                "source": source,
                "url": url,
                "date": published_at,
                "type": "news_article"
            }
            
            # Create document
            document = {
                "content": document_content,
                "metadata": metadata
            }
            
            # Generate ID based on URL
            import hashlib
            doc_id = f"news_{hashlib.md5(url.encode()).hexdigest()}"
            
            # Add to news collection
            return self.add_document("news", document, doc_id)
            
        except Exception as e:
            logger.error(f"Error indexing news article: {e}")
            return False
    
    def index_market_data(self, coin: str, market_data: Dict[str, Any]) -> bool:
        """
        Index market data in the vector database
        
        Args:
            coin: Cryptocurrency symbol
            market_data: Market data dictionary
            
        Returns:
            Success status
        """
        try:
            # Extract market data
            current_price = market_data.get("current_price", "Unknown")
            daily_change = market_data.get("daily_change_pct", "Unknown")
            volume = market_data.get("volume_24h", "Unknown")
            indicators = market_data.get("indicators", {})
            
            # Format indicators text
            indicators_text = ""
            for timeframe, data in indicators.items():
                indicators_text += f"\nTimeframe: {timeframe}\n"
                for indicator, value in data.items():
                    indicators_text += f"- {indicator}: {value}\n"
            
            # Combine into document content
            document_content = (
                f"MARKET DATA FOR {coin}\n\n"
                f"Price: {current_price} USD\n"
                f"24h Change: {daily_change}%\n"
                f"24h Volume: {volume} USD\n"
                f"Technical Indicators: {indicators_text}\n"
            )
            
            # Create metadata
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata = {
                "coin": coin,
                "date": timestamp,
                "type": "market_data"
            }
            
            # Create document
            document = {
                "content": document_content,
                "metadata": metadata
            }
            
            # Generate ID
            doc_id = f"market_{coin}_{int(time.time())}"
            
            # Add to technical analysis collection
            return self.add_document("technical_analysis", document, doc_id)
            
        except Exception as e:
            logger.error(f"Error indexing market data: {e}")
            return False

# Singleton instance
rag_system = RAGSystem()

# Helper function to get the singleton instance
def get_rag_system():
    return rag_system

# Example usage
if __name__ == "__main__":
    system = get_rag_system()
    # Test adding a document
    test_doc = {
        "content": "Bitcoin is a decentralized digital currency that was created in 2009.",
        "metadata": {
            "source": "Test",
            "date": "2023-01-01"
        }
    }
    system.add_document("market_knowledge", test_doc)
    # Test querying
    results = system.query_collection("market_knowledge", "What is Bitcoin?")
    print(f"Query results: {results}")