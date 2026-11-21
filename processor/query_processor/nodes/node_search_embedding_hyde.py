import json

from langchain.chat_models import init_chat_model

from config import prompt
from config.config import LLMConfig, MilvusConfig
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.bge_m3_tool import get_bge_m3_embeddings
from tool.json_format import json_format
from tool.logger import logger
from tool.milvus_utils import create_hybrid_request, hybrid_search


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState):
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        if not rewritten_query:
            logger.info("没有改写后的用户问题")
            raise Exception("没有改写后的用户问题")
        if not item_names:
            logger.info("没有已确认主体名")
            raise Exception("没有已确认主体名")

        llm = init_chat_model(
            model=LLMConfig.LLM_DEFAULT_MODEL,
            model_provider="openai",
            temperature=LLMConfig.LLM_DEFAULT_TEMPERATURE,
            base_url=LLMConfig.OPENAI_API_BASE,
            api_key=LLMConfig.OPENAI_API_KEY
        )
        messages = [
            {
                "role": "user",
                "content": prompt.HYDE_PROMPT.format(rewritten_query=rewritten_query)
            }
        ]
        response = llm.invoke(messages)
        hyde_answer = response.content
        merges_query = rewritten_query + " " + hyde_answer

        embeddings = get_bge_m3_embeddings([merges_query])
        dense_data = embeddings.get("dense")
        sparse_data = embeddings.get("sparse")
        item_names = [item_name.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"') for item_name in
                      item_names]
        expr = f"item_name in {item_names}"
        reqs = create_hybrid_request(dense_data, sparse_data, "dense_vector", "sparse_vector", limit=10, expr=expr)
        result = hybrid_search(MilvusConfig.chunks_collection, reqs, weight=(0.8, 0.2), limit=10,
                               output_fields=["id", "file_title", "title", "content", "item_name"])
        chunks = []
        for item in result[0]:
            chunks.append({
                **item.get("entity"),
                "score": item.get("distance"),
                "source": "local"
            })

        # return state
        return {"hyde_embedding_chunks": chunks}


if __name__ == '__main__':
    init_state = {
        "rewritten_query": "关于HAK180烫金机如何使用",
        "item_names": ["HAK180烫金机"]
    }
    node_search_embedding_hyde = NodeSearchEmbeddingHyde()
    res = node_search_embedding_hyde(init_state)
    logger.info(json_format(res))
