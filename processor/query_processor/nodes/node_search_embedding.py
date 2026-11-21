import json

from config.config import MilvusConfig
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.bge_m3_tool import get_bge_m3_embeddings
from tool.json_format import json_format
from tool.logger import logger
from tool.milvus_utils import create_hybrid_request, hybrid_search


class NodeSearchEmbedding(NodeBase):
    """
   节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
   """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState):
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        if not rewritten_query:
            logger.error("改写后的用户问题不能为空")
            raise Exception("改写后的用户问题不能为空")
        if not item_names:
            logger.error("已确认主体名不能为空")
            raise Exception("已确认主体名不能为空")

        embeddings = get_bge_m3_embeddings([rewritten_query])
        dense_data = embeddings.get("dense")
        sparse_data = embeddings.get("sparse")
        item_names = [item_name.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"') for item_name in
                      item_names]
        expr = f"item_name in {item_names}"
        reqs = create_hybrid_request(dense_data, sparse_data, "dense_vector", "sparse_vector", limit=10, expr=expr)
        results = hybrid_search(MilvusConfig.chunks_collection, reqs, weight=(0.8, 0.2), limit=10,
                                output_fields=["id", "item_name", "file_title", "content", "title"])
        embeddings_chunks = []
        for result in results[0]:
            embeddings_chunks.append({
                **result.get("entity"),
                "score": result.get("distance"),
                "source": "local"
            })

        return {
            "embedding_chunks": embeddings_chunks
        }


if __name__ == '__main__':
    init_state = {
        "rewritten_query": "成都交通的地铁怎么坐",
        "item_names": ["成都交通指南"]
    }
    node_search_embedding = NodeSearchEmbedding()
    res = node_search_embedding(init_state)
    logger.info(json_format(res))
