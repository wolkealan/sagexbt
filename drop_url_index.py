from pymongo import MongoClient
from bson import json_util
import json

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["walletTracker"]
collection = db["news_data"]

# 1. Drop all existing indexes
collection.drop_indexes()

# 2. Create a new index strategy
collection.create_index([("query_hash", 1)], unique=False)
collection.create_index([("timestamp", -1)], expireAfterSeconds=21600)  # 6 hours TTL

# 3. Verify the new indexes
print("New indexes:")
for index in collection.list_indexes():
    print(json.dumps(index, indent=2, default=json_util.default))

# 4. Optional: Clear the collection
result = collection.delete_many({})
print(f"Deleted {result.deleted_count} documents from news_data collection")