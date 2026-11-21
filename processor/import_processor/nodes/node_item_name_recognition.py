import json

from langchain.chat_models import init_chat_model
from pymilvus import DataType

from config.config import LLMConfig, MilvusConfig
from tool.bge_m3_tool import get_bge_m3_embeddings
from tool.json_format import json_format

from tool.logger import logger
from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState
from tool.milvus_utils import get_milvus_client

from config.prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取
    """

    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState):
        # 获取chunks
        chunks, file_title = self.get_chunks(state)
        # 主体识别生成item_name
        item_name = self.get_item_name(chunks, file_title)
        # 创建collection
        collection_name, milvus_client = self.create_milvus_collection()
        # 向量化item_name,并且插入item_name到milvus
        self.insert_item_name_milvus(chunks, collection_name, file_title, item_name, milvus_client)

        # with open(r"D:\Study\python\knowledge_base\output\hak180产品安全手册\item_name_chunks.json", "w",
        #           encoding="utf-8") as f:
        #     f.write(json_format(chunks))
        return {
            "chunks": chunks
        }

    def insert_item_name_milvus(self, chunks, collection_name, file_title, item_name, milvus_client):
        result = get_bge_m3_embeddings([item_name])
        dense_vector = result.get("dense")[0]
        sparse_vector = result.get("sparse")[0]
        milvus_client.load_collection(collection_name=collection_name)
        filter_title = file_title.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        filter = f"file_title == '{filter_title}'"
        milvus_client.delete(
            collection_name=collection_name,
            filter=filter
        )
        milvus_client.insert(
            collection_name=collection_name,
            data={
                "item_name": item_name,
                "file_title": file_title,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
        )
        for chunk in chunks:
            chunk["item_name"] = item_name

    def create_milvus_collection(self):
        milvus_client = get_milvus_client()
        collection_name = MilvusConfig.item_name_collection
        if not milvus_client.has_collection(collection_name=collection_name):
            schema = milvus_client.create_schema(
                auto_id=True
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.INT64,
                is_primary=True
            ).add_field(
                field_name="item_name",
                datatype=DataType.VARCHAR,
                max_length=100
            ).add_field(
                field_name="file_title",
                datatype=DataType.VARCHAR,
                max_length=100
            ).add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=1024
            ).add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            index_params = milvus_client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 1024, "nprobe": 10}
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",
                }
            )
            milvus_client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
        return collection_name, milvus_client

    def get_item_name(self, chunks, file_title):
        max_len = 12000
        chunk_k_list = chunks[:20]
        content_str = ""
        for idx, chunk in enumerate(chunk_k_list, start=1):
            title = chunk.get("title")
            content = chunk.get("content")
            chunk_content = f"[切片:{idx}]\n[标题:{title}]\n[内容:{content}]\n"
            if len(content_str) > max_len:
                content_str = content_str[:max_len]
                break
            content_str += chunk_content
        llm = init_chat_model(
            model=LLMConfig.ITEM_MODEL,
            temperature=LLMConfig.LLM_DEFAULT_TEMPERATURE,
            model_provider="openai",
            base_url=LLMConfig.OPENAI_API_BASE,
            api_key=LLMConfig.OPENAI_API_KEY
        )
        messages = [
            {
                "role": "system",
                "content": ITEM_NAME_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=content_str)
            }
        ]
        response = llm.invoke(messages)
        item_name = response.content
        item_name = item_name.replace("\n", "").replace("\t", "").replace(" ", "")
        if not item_name:
            item_name = file_title
        return item_name

    def get_chunks(self, state):
        chunks = state.get("chunks")
        file_title = state.get("file_title")
        if not chunks:
            logger.error("chunks为空,无法进行主体识别")
            raise Exception("chunks为空，无法进行主题识别")
        if not file_title:
            logger.error("file_title为空,无法进行主体识别")
            raise Exception("file_title为空，无法进行主题识别")
        return chunks, file_title


if __name__ == '__main__':
    with open(r"D:\Study\python\knowledge_base\output\hak180产品安全手册\split_chunk.json", "r", encoding="utf-8") as f:
        chunks = f.read()
    init_state = {
        "chunks": json.loads(chunks),
        "file_title": "ak180产品安全手册"
    }
    node = NodeItemNameRecognition()
    response = node(init_state)
    logger.info(response)
