import json

from pymilvus import DataType

from config.config import MilvusConfig
from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState
from tool.logger import logger
from tool.milvus_utils import get_milvus_client


class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """

    name = "node_import_milvus"

    def process(self, state: ImportGraphState):
        chunks = state.get("chunks")
        if not chunks:
            logger.info("chunks不存在")
            raise Exception("chunks不存在")

        milvus_client = get_milvus_client()
        collection_name = MilvusConfig.chunks_collection
        if not milvus_client.has_collection(collection_name):
            schema = milvus_client.create_schema(
                auto_id=True
            )
            (schema.add_field(
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
                field_name="content",
                datatype=DataType.VARCHAR,
                max_length=3000
            ).add_field(
                field_name="title",
                datatype=DataType.VARCHAR,
                max_length=100
            ).add_field(
                field_name="part",
                datatype=DataType.INT64
            ).add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=1024
            ).add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            ))
            index_params = milvus_client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 128, "nprobe": 10}
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE", }
            )
            milvus_client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )

        file_title = chunks[0]["file_title"]
        milvus_client.load_collection(collection_name=collection_name)
        file_title = file_title.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        filter_str = f"file_title == '{file_title}'"
        milvus_client.delete(
            collection_name=collection_name,
            filter=filter_str
        )
        result = milvus_client.insert(
            collection_name=collection_name,
            data=chunks
        )
        ids = result.get("ids")
        for idx, chunk in enumerate(chunks):
            chunk["id"] = ids[idx]

        return {
            "chunks": chunks
        }


if __name__ == '__main__':
    with open(r"D:\Study\python\knowledge_base\output\hak180产品安全手册\embeddings_chunks.json", "r",
              encoding="utf-8") as f:
        chunks = json.load(f)
    init_state = {
        "chunks": chunks
    }
    node_import_milvus = NodeImportMilvus()
    res = node_import_milvus(init_state)
    logger.info(res)
