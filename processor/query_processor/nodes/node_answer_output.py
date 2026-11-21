import re

from langchain.chat_models import init_chat_model

from config.config import LLMConfig
from config.prompt import ANSWER_PROMPT
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.mongo_utils import add_or_update_message
from tool.task_utils import put_data


class NodeAnswerOutput(NodeBase):
    """
    节点功能: 答案生成
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_answer_output"

    def process(self, state: QueryGraphState):
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        task_id = state.get("task_id")
        answer = state.get("answer")
        chunks = state.get("reranked_docs")
        history_list = state.get("history")
        item_names = state.get("item_names")
        question = state.get("rewritten_query")
        session_id = state.get("session_id")
        if answer:
            put_data(task_id, "final", {"answer": answer})
        else:
            llm = init_chat_model(
                model=LLMConfig.ITEM_MODEL,
                model_provider="openai",
                temperature=LLMConfig.LLM_DEFAULT_TEMPERATURE,
                base_url=LLMConfig.OPENAI_API_BASE,
                api_key=LLMConfig.OPENAI_API_KEY
            )
            context = ''
            for chunk in chunks:
                title = chunk.get("title")
                content = chunk.get("content")
                url = chunk.get("url")
                source = chunk.get("source")
                chunk_content = f"标题：{title}\n内容：{content}\n来源：{source}\n链接：{url}\n\n"
                context += chunk_content

            history = ""
            for item in history_list:
                h_content = f"{item.get('role')}: {item.get('content')}\n\n"
                history += h_content

            item_names_str = ','.join(item_names)

            prompt = ANSWER_PROMPT.format(context=context, question=question, history=history,
                                          item_names=item_names_str)

            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            res = llm.stream(input=messages)
            for item in res:
                put_data(task_id, "delta", {"delta": item.content})
                answer += item.content

            seen = set()
            md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
            for item in chunks:
                content = item.get("content")
                matches = md_img_pattern.findall(content)
                for image_url in matches:
                    image_url = image_url.strip()
                    seen.add(image_url)
            images = list(seen)
            put_data(task_id, "final", {"image_urls": images})

            if answer:
                add_or_update_message(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    rewritten_query=question,
                    item_names=item_names,
                )

        return {
            "answer": answer
        }
