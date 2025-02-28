from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
import pymongo
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import certifi

from config.config import DatabaseConfig
from utils.logger import get_logger

logger = get_logger("database")

class MongoDB:
    """MongoDB database connection and operations"""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one database connection is created"""
        if cls._instance is None:
            cls._instance = super(MongoDB, cls).__new__(cls)
            cls._instance.client = None
            cls._instance.db = None
            cls._instance.initialize_connection()
        return cls._instance
    
    def initialize_connection(self):
        """Initialize connection to MongoDB"""
        try:
            # Connect to MongoDB with TLS/SSL
            self.client = MongoClient(DatabaseConfig.MONGODB_URI, tlsCAFile=certifi.where())
            self.db = self.client[DatabaseConfig.DB_NAME]
            
            # Verify connection
            self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {DatabaseConfig.DB_NAME}")
            
            # Create indexes for collections
            self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes for collections to optimize queries"""
        try:
            # Market data indexes
            market_collection = self.db[DatabaseConfig.MARKET_DATA_COLLECTION]
            market_collection.create_index([("symbol", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
            market_collection.create_index([("timeframe", pymongo.ASCENDING)])
            market_collection.create_index([("timestamp", pymongo.DESCENDING)], expireAfterSeconds=DatabaseConfig.MARKET_DATA_TTL)
            
            # # News data indexes
            # news_collection = self.db[DatabaseConfig.NEWS_COLLECTION]
            # news_collection.create_index([("query_hash", pymongo.ASCENDING)])
            # news_collection.create_index([("timestamp", pymongo.DESCENDING)], expireAfterSeconds=DatabaseConfig.NEWS_DATA_TTL)
            
            # Recommendations indexes
            rec_collection = self.db[DatabaseConfig.RECOMMENDATIONS_COLLECTION]
            rec_collection.create_index([("coin", pymongo.ASCENDING), ("action_type", pymongo.ASCENDING)])
            rec_collection.create_index([("timestamp", pymongo.DESCENDING)], expireAfterSeconds=DatabaseConfig.RECOMMENDATIONS_TTL)
            
            logger.info("Created MongoDB indexes")
            
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")
    
    def get_collection(self, collection_name: str) -> Collection:
        """Get a MongoDB collection"""
        return self.db[collection_name]
    
    def insert_one(self, collection_name: str, document: Dict[str, Any]) -> Optional[str]:
        """Insert a document into a collection"""
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert_one(document)
            logger.debug(f"Inserted document in {collection_name} with id {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting document to {collection_name}: {e}")
            return None
    
    def insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> Optional[List[str]]:
        """Insert multiple documents into a collection"""
        try:
            if not documents:
                return []
                
            collection = self.get_collection(collection_name)
            result = collection.insert_many(documents)
            logger.debug(f"Inserted {len(result.inserted_ids)} documents in {collection_name}")
            return [str(id) for id in result.inserted_ids]
        except Exception as e:
            logger.error(f"Error inserting documents to {collection_name}: {e}")
            return None
    
    def find_one(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a document in a collection"""
        try:
            collection = self.get_collection(collection_name)
            document = collection.find_one(query)
            return document
        except Exception as e:
            logger.error(f"Error finding document in {collection_name}: {e}")
            return None
    
    def find_many(self, collection_name: str, query: Dict[str, Any], 
                 sort: List[tuple] = None, limit: int = 0) -> List[Dict[str, Any]]:
        """Find documents in a collection"""
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.find(query)
            
            if sort:
                cursor = cursor.sort(sort)
            
            if limit > 0:
                cursor = cursor.limit(limit)
            
            return list(cursor)
        except Exception as e:
            logger.error(f"Error finding documents in {collection_name}: {e}")
            return []
    
    def update_one(self, collection_name: str, query: Dict[str, Any], 
              update: Dict[str, Any], upsert: bool = False) -> bool:
        """Update a document in a collection"""
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_one(query, update, upsert=upsert)
            success = result.modified_count > 0 or (upsert and result.upserted_id is not None)
            logger.debug(f"Updated document in {collection_name}: {success}")
            return success
        except pymongo.errors.DuplicateKeyError:
            # Silently ignore duplicate key errors
            logger.debug(f"Duplicate key in {collection_name}, skipping")
            return True
        except Exception as e:
            logger.error(f"Error updating document in {collection_name}: {e}")
            return False
    
    def delete_one(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """Delete a document from a collection"""
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_one(query)
            success = result.deleted_count > 0
            logger.debug(f"Deleted document from {collection_name}: {success}")
            return success
        except Exception as e:
            logger.error(f"Error deleting document from {collection_name}: {e}")
            return False
    
    def find_recent(self, collection_name: str, query: Dict[str, Any], 
                   hours: int = 24, sort_field: str = "timestamp") -> List[Dict[str, Any]]:
        """Find recent documents from the last specified hours"""
        try:
            # Calculate the timestamp for N hours ago
            time_ago = datetime.now() - timedelta(hours=hours)
            
            # Add timestamp condition to the query
            time_query = query.copy()
            time_query[sort_field] = {"$gte": time_ago}
            
            # Find documents
            collection = self.get_collection(collection_name)
            documents = list(collection.find(time_query).sort([(sort_field, pymongo.DESCENDING)]))
            
            logger.debug(f"Found {len(documents)} recent documents in {collection_name}")
            return documents
        except Exception as e:
            logger.error(f"Error finding recent documents in {collection_name}: {e}")
            return []

# Singleton instance
mongodb = MongoDB()

# Helper function to get the singleton instance
def get_database():
    return mongodb

# Example usage
if __name__ == "__main__":
    db = get_database()
    # Test insert
    test_doc = {"test": "data", "timestamp": datetime.now()}
    db.insert_one("test_collection", test_doc)
    # Test query
    result = db.find_one("test_collection", {"test": "data"})
    print(f"Query result: {result}")