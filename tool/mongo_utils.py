import time

from pymongo import MongoClient

from config.config import MongoConfig
from tool.json_format import json_format
from tool.logger import logger

mongo_client = None


def get_mongo_client():
    global mongo_client
    if mongo_client is None:
        mongo_client = MongoClient(MongoConfig.mongo_url)
    return mongo_client


mongo_tool = None


def get_mongo_tool():
    global mongo_tool
    if mongo_tool is None:
        mongo_client = get_mongo_client()
        db = mongo_client[MongoConfig.mongo_db_name]
        mongo_tool = db["chat_history"]
        mongo_tool.create_index([("session_id", 1), ("ts", -1)])
    return mongo_tool


def clear_history(session_id):
    mongo_tool = get_mongo_tool()
    mongo_tool.delete_many({"session_id": session_id})


def add_or_update_message(session_id, role, text, rewritten_query=None, item_names=None, ts=None, message_id=None):
    mongo_tool = get_mongo_tool()
    if not message_id:
        res = mongo_tool.insert_one({
            "session_id": session_id,
            "role": role,
            "text": text,
            "rewritten_query": rewritten_query,
            "item_names": item_names,
            "ts": ts or int(time.time())
        })
        return res.inserted_id
    else:
        res = mongo_tool.update_one({"_id": message_id}, {"$set": {
            "session_id": session_id,
            "role": role,
            "text": text,
            "rewritten_query": rewritten_query,
            "item_names": item_names,
            "ts": ts or int(time.time())
        }})
        return res.modified_count


def get_recent_messages(session_id, limit=10):
    mongo_tool = get_mongo_tool()
    res = mongo_tool.find({"session_id": session_id}).sort("ts", -1).limit(limit)
    return list(res)


def update_history_messages(ids, item_names, rewritten_query):
    mongo_tool = get_mongo_tool()
    mongo_tool.update_many({"_id": {"$in": ids}},
                           {"$set": {"item_names": item_names, "rewritten_query": rewritten_query}})


if __name__ == '__main__':
    # 模拟会话历史
    clear_history("test_001")
