import json
import logging


from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState
from tool.bge_m3_tool import get_bge_m3_embeddings
from tool.logger import logger


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):
        chunks = self.get_chunks(state)

        self.embedding_chunks(chunks)

        # with open(r"D:\Study\python\knowledge_base\output\hak180产品安全手册\embeddings_chunks.json", 'w',
        #           encoding="utf-8") as f:
        #     f.write(json_format(chunks))
        return state

    def embedding_chunks(self, chunks):
        for i in range(0, len(chunks), 3):
            chunk_k_list = chunks[i:i + 3]
            chunk_content_k_list = [f"{item.get('item_name')}\n{item.get('content')}" for item in chunk_k_list]
            embeddings = get_bge_m3_embeddings(chunk_content_k_list)
            for idx, chunk in enumerate(chunk_k_list):
                chunk["dense_vector"] = embeddings.get("dense")[idx]
                chunk["sparse_vector"] = embeddings.get("sparse")[idx]

    def get_chunks(self, state):
        chunks = state.get("chunks")
        if not chunks:
            logger.error("chunks不存在")
            raise Exception("chunks不存在")
        return chunks


if __name__ == '__main__':
    with open(r"D:\Study\python\knowledge_base\output\hak180产品安全手册\item_name_chunks.json", "r",
              encoding="utf-8") as f:
        chunks = json.load(f)
    init_state = {
        "chunks": chunks
    }
    node_bge_embedding = NodeBGEEmbedding()
    res = node_bge_embedding(init_state)
    logging.info(res)
