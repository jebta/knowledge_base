import dashscope
from http import HTTPStatus

from config.config import RerankConfig
from tool.logger import logger

# 以下为华北2（北京）地域的配置，调用时请将{WorkspaceId}替换为真实的业务空间ID，各地域的配置不同。
dashscope.base_http_api_url = RerankConfig.rerank_base_url
dashscope.api_key = RerankConfig.rerank_api_key

def text_rerank(query,texts,limit=10):
    resp = dashscope.TextReRank.call(
        model="qwen3.7-text-rerank",
        query=query,
        documents=texts,
        top_n=limit,
        instruct="Given a web search query, retrieve relevant passages that answer the query."
    )
    if resp.status_code == HTTPStatus.OK:
        results = resp.output.get("results")
        return [{
            "index": item.index,
            "score": item.relevance_score,
        } for item in results]
    else:
        logger.info("rerank排序错误")
        raise Exception("rerank排序错误")

