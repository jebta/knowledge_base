import json
from pathlib import Path
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState
from tool.json_format import json_format
from tool.logger import logger


class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):
        # 获取md_content内容,替换\r\n,\r为\n
        file_title, md_content, md_path_obj = self.get_md_content(state)
        # 粗切section
        section_list = self.get_section_list(file_title, md_content)
        # 细切为chunk
        chunk_list = self.get_chunk_list(section_list)

        # 保存切分后的json文档
        # split_chunk_json_obj = md_path_obj.parent / "split_chunk.json"
        # with open(split_chunk_json_obj, "w", encoding="utf-8") as f:
        #     f.write(json_format(chunk_list))
        return {
            "chunks": [chunk for chunk in chunk_list if chunk.get("content")]
        }

    def get_chunk_list(self, section_list):
        max_len = 300
        spliter = RecursiveCharacterTextSplitter(
            chunk_size=max_len,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", " "]
        )
        chunk_list = []
        for section in section_list:
            title = section.get("title")
            content = section.get("content")
            real_content = content[len(title) + 2:] if title.startswith("#") else content
            if len(real_content) < max_len:
                chunk_list.append({
                    **section,
                    "part": 0
                })
                continue
            if "<table" in real_content:
                chunk_list.append({
                    **section,
                    "part": 0
                })
                continue
            for idx, spliter_text in enumerate(spliter.split_text(real_content), start=1):
                chunk_list.append({
                    **section,
                    "part": idx,
                    "content": title + "\n\n" + spliter_text
                })
        return chunk_list

    def get_section_list(self, file_title, md_content):
        line_list = md_content.split("\n")
        block_pattern = r"^(`{3,}|~{3,})"
        title_pattern = r"^\s*#{1,6}\s+.+"
        is_block = False
        marker = None
        section_list = []
        current_idx = 0
        for idx, line in enumerate(line_list):
            match = re.match(block_pattern, line)
            if match:
                if not is_block:
                    is_block = True
                    marker = match.group(1)
                else:
                    if marker[0] == match.group(1)[0]:
                        is_block = False
                        marker = None
            if not is_block and re.match(title_pattern, line):
                section_content = "\n".join(line_list[current_idx:idx])
                section_list.append({
                    "file_title": file_title,
                    "title": line_list[current_idx] if line_list[current_idx].startswith("#") else "无标题",
                    "content": section_content
                })
                current_idx = idx
        section_content = "\n".join(line_list[current_idx:])
        section_list.append({
            "file_title": file_title,
            "title": line_list[current_idx] if line_list[current_idx].startswith("#") else "无标题",
            "content": section_content
        })
        return section_list

    def get_md_content(self, state):
        md_path = state.get("md_path")
        file_title = state.get("file_title")
        if not md_path:
            logger.error("md_path为空")
            raise Exception("md_path为空")
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            logger.error(f"md_path文件不存在：{md_path}")
            raise Exception(f"md_path文件不存在：{md_path}")
        if not file_title:
            logger.error("file_title为空")
            raise Exception("file_title为空")
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        return file_title, md_content, md_path_obj


if __name__ == '__main__':
    init_state = {
        "md_path": "D:\Study\python\knowledge_base\output\hak180产品安全手册\hak180产品安全手册_new.md",
        "file_title": "ak180产品安全手册"
    }
    node_document_split = NodeDocumentSplit()
    res = node_document_split(init_state)
    logger.info(res)
