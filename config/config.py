import os

from dotenv import load_dotenv

# 使用绝对路径加载 .env
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
load_dotenv(dotenv_path=env_path, override=True)


class MineruConfig:
    mineru_token = os.getenv("MINERU_API_TOKEN")


class LLMConfig:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    LLM_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL")
    LLM_DEFAULT_TEMPERATURE = float(os.getenv("LLM_DEFAULT_TEMPERATURE"))
    VL_MODEL = os.getenv("VL_MODEL")
    ITEM_MODEL = os.getenv("ITEM_MODEL")


class MinioConfig:
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
    MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")
    MINIO_IMG_DIR = os.getenv("MINIO_IMG_DIR")


class BGEM3Config:
    bge_m3_path = os.getenv("BGE_M3_PATH")
    bge_m3 = os.getenv("BGE_M3")
    bge_device = os.getenv("BGE_DEVICE")
    bge_fp16 = True if os.getenv("BGE_FP16") in ("True", "true", "1") else False


class MilvusConfig:
    milvus_url = os.getenv("MILVUS_URL")
    chunks_collection = os.getenv("CHUNKS_COLLECTION")
    item_name_collection = os.getenv("ITEM_NAME_COLLECTION")


class MongoConfig:
    mongo_url = os.getenv("MONGO_URL")
    mongo_db_name = os.getenv("MONGO_DB_NAME")


class McpConfig:
    mcp_dashscope_base_url = os.getenv("MCP_DASHSCOPE_BASE_URL")
    mcp_dashscope_api_key = os.getenv("MCP_DASHSCOPE_API_KEY")


class RerankConfig:
    rerank_base_url = os.getenv("RERANK_BASE_URL")
    rerank_api_key = os.getenv("RERANK_API_KEY")
