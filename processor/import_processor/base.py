import time
from abc import ABC, abstractmethod
from processor.import_processor.state import ImportGraphState
from tool.logger import logger
from tool.task_utils import add_running_task, add_done_task, add_node_duration

"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能
"""


class BaseNode(ABC):
    name = "node_base"

    def __init__(self, **kwargs):
        if self.name == "node_base":
            logger.error(f"子类{self.__class__.__name__}必须重写父类的node_name属性")
            raise Exception(f"子类{self.__class__.__name__}必须重写父类的node_name属性")

    def __call__(self, state: ImportGraphState):
        task_id = state.get("task_id")
        try:
            logger.info(f"{self.name}开始处理节点")
            start_time = time.time()
            add_running_task(task_id, self.name)
            result = self.process(state)
            logger.info(f"{self.name}结束处理节点")
            end_time = time.time()
            add_done_task(task_id, self.name)
            add_node_duration(task_id, self.name, end_time - start_time)
            return result
        except Exception as e:
            logger.error(f"{self.name}处理节点发生异常")
            raise e

    @abstractmethod
    def process(self, state):
        pass
