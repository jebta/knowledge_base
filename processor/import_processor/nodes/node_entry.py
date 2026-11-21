from pathlib import Path

from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState
from tool.logger import logger


class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """

    name = 'node_entry'

    def process(self, state: ImportGraphState):
        local_file = state.get("local_file_path")
        if not local_file:
            logger.error(f"{local_file}路径为空")
            raise Exception(f"{local_file}路径为空")
        local_file_path = Path(local_file)
        if not local_file_path.exists():
            logger.error(f"{local_file}文件不存在")
            raise Exception(f"{local_file}文件不存在")

        suffix = local_file_path.suffix
        file_title = local_file_path.stem
        if suffix.lower() == '.pdf':
            return {
                "is_pdf_read_enabled": True,
                "pdf_path": local_file,
                "file_title": file_title
            }
        elif suffix.lower() == '.md':
            return {
                "is_md_read_enabled": True,
                "md_path": local_file,
                "file_title": file_title
            }
        else:
            logger.error(f"{suffix}文件格式不支持")
            raise Exception(f"{suffix}文件格式不支持")


if __name__ == '__main__':
    init_state = {
        "local_file_path": "D:\Study\python\knowledge_base\doc\hak180产品安全手册.pdf",
    }
    node_entry = NodeEntry()
    res = node_entry(init_state)
    logger.info(res)
