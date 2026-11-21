import json

from langchain.chat_models import init_chat_model


from config.config import LLMConfig, MilvusConfig
from config.prompt import ITEM_NAME_EXTRACT_SYSTEM_PROMPT, ITEM_NAME_EXTRACT_TEMPLATE
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.bge_m3_tool import get_bge_m3_embeddings
from tool.json_format import json_format
from tool.logger import logger
from tool.milvus_utils import create_hybrid_request, hybrid_search
from tool.mongo_utils import get_recent_messages, add_or_update_message, update_history_messages


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState):
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        history_text, original_query, session_id = self.get_history_content(state)

        item_names, message_id, rewritten_query = self.get_llm_item_names(history_text, original_query, session_id)

        search_item_names_list = []
        final_item_names = []
        answer = "不好意思，无法理解您的提问，能否再提问一次？"
        if item_names:
            self.hybrid_search_item_names(item_names, search_item_names_list)
            answer, final_item_names = self.align_item_names(answer, final_item_names, search_item_names_list)
        message_id = self.handler_history(answer, final_item_names, message_id, rewritten_query, session_id)

        return {
            "session_id": session_id,
            "original_query": original_query,
            "item_names": final_item_names,
            "rewritten_query": rewritten_query,
            "message_id": message_id,
            "answer": answer,
            "history": get_recent_messages(session_id)
        }

    def handler_history(self, answer, final_item_names, message_id, rewritten_query, session_id):
        if answer:
            message_id = add_or_update_message(session_id, "assistant", answer)
        else:
            history_contents = get_recent_messages(session_id)
            ids = [history_content.get("_id") for history_content in history_contents]
            if ids:
                update_history_messages(ids, final_item_names, rewritten_query)
        return message_id

    def align_item_names(self, answer, final_item_names, search_item_names_list):
        confirm_item_names = [item.get("item_name") for item in search_item_names_list if item.get("score") >= 0.75]
        option_item_names = [item.get("item_name") for item in search_item_names_list if
                             0.6 <= item.get("score") < 0.75]
        if confirm_item_names:
            final_item_names = confirm_item_names
            answer = ""
        elif option_item_names:
            final_item_names = option_item_names
            answer = f"您的问题中可能包含多个目的地/景点，请确认：[{final_item_names}]"
        return answer, final_item_names

    def hybrid_search_item_names(self, item_names, search_item_names_list):
        embeddings = get_bge_m3_embeddings(item_names)
        dense_data = embeddings.get("dense")
        sparse_data = embeddings.get("sparse")
        reqs = create_hybrid_request(dense_data, sparse_data, "dense_vector", "sparse_vector")
        results = hybrid_search(MilvusConfig.item_name_collection, reqs, weight=(0.7, 0.3), limit=10,
                                output_fields=["item_name"])
        for idx, item in enumerate(item_names):
            for result in results[idx]:
                search_item_names_list.append({
                    "item_name": result.get("entity", {}).get("item_name", ""),
                    "score": result.get("distance")
                })

    def get_llm_item_names(self, history_text, original_query, session_id):
        message_id = add_or_update_message(session_id, "user", original_query)
        llm = init_chat_model(
            model=LLMConfig.LLM_DEFAULT_MODEL,
            temperature=LLMConfig.LLM_DEFAULT_TEMPERATURE,
            model_provider="openai",
            base_url=LLMConfig.OPENAI_API_BASE,
            api_key=LLMConfig.OPENAI_API_KEY
        )
        messages = [
            {
                "role": "system",
                "content": ITEM_NAME_EXTRACT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=history_text, original_query=original_query)
            }
        ]
        response = llm.invoke(messages)
        content = response.content
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        json_content = json.loads(content)
        item_names = json_content.get("item_names")
        item_names = [item_name.replace("\n", "").replace(" ", "").replace("\t", "") for item_name in item_names]
        if not item_names:
            item_names = []
        rewritten_query = json_content.get("rewritten_query")
        if not rewritten_query:
            rewritten_query = original_query
        return item_names, message_id, rewritten_query

    def get_history_content(self, state):
        session_id = state.get("session_id")
        original_query = state.get("original_query")
        if not session_id:
            logger.error("session_id不能为空")
            raise Exception("session_id不能为空")
        if not original_query:
            logger.error("original_query不能为空")
            raise Exception("original_query不能为空")
        history = get_recent_messages(session_id)
        history_text = ""
        for item in history:
            history_text += (item.get("role") + "\n" + item.get("text"))
        return history_text, original_query, session_id


if __name__ == '__main__':
    init_state = {
        "session_id": "test_001",
        "original_query": "成都交通指南？"
    }
    node_item_name_confirm = NodeItemNameConfirm()
    res = node_item_name_confirm(init_state)
    logger.info(json_format(res))
