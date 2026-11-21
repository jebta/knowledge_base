import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from config.config import McpConfig
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.json_format import json_format
from tool.logger import logger


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState):
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            logger.info("未找到重写后的查询语句")
            raise Exception("未找到重写后的查询语句")

        chunks = []

        async def mcp_run() -> None:
            token = McpConfig.mcp_dashscope_api_key
            async with MCPServerStreamableHttp(
                    name="Streamable HTTP Python Server",
                    params={
                        "url": McpConfig.mcp_dashscope_base_url,
                        "headers": {"Authorization": f"Bearer {token}"},
                        "timeout": 10,
                    },
                    cache_tools_list=True,
                    max_retry_attempts=3,
                    client_session_timeout_seconds=30
            ) as server:
                result = await server.call_tool("bailian_web_search", arguments={"query": rewritten_query, "count": 10})
                datas = result.content[0].text
                datas = json.loads(datas).get("pages")
                for data in datas:
                    chunks.append({
                        "title": data.get("title"),
                        "content": data.get("snippet"),
                        "url": data.get("url"),
                        "source": "web"
                    })

        asyncio.run(mcp_run())

        # return state
        return {"web_search_docs": chunks}


if __name__ == '__main__':
    init_state = {
        "rewritten_query": "关于HAK180烫金机如何使用",
    }
    node_web_search_mcp = NodeWebSearchMcp()
    res = node_web_search_mcp(init_state)
    logger.info(json_format(res))
