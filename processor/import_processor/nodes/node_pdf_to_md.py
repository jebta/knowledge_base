import shutil
import time
import zipfile
from pathlib import Path

import requests

from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState
from config.config import MineruConfig
from tool.logger import logger


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = 'node_pdf_to_md'

    def process(self, state: ImportGraphState):
        local_dir_obj, pdf_path, pdf_path_obj = self.check_pdf(state)

        batch_id, token = self.upload_pdf_mineru(pdf_path, pdf_path_obj)

        zip_url = self.get_md_zip_url(batch_id, token)

        md_content, new_md_path_obj = self.md_unzip(local_dir_obj, pdf_path_obj, zip_url)
        return {
            "md_content": md_content,
            "md_path": str(new_md_path_obj)
        }

    def md_unzip(self, local_dir_obj, pdf_path_obj, zip_url):
        res_download = requests.get(zip_url)
        if res_download.status_code != 200:
            logger.error(f"压缩文件下载失败，状态码：{res_download.status_code}")
            raise Exception(f"压缩文件下载失败，状态码：{res_download.status_code}")
        zip_path_obj = local_dir_obj / f"{pdf_path_obj.stem}.zip"
        with open(zip_path_obj, 'wb') as f:
            f.write(res_download.content)
        logger.info(f"压缩文件保存成功，路径：{zip_path_obj}")
        unzip_path_obj = local_dir_obj / f"{pdf_path_obj.stem}"
        # 文件夹没有一定要创建，文件可以不创建
        if unzip_path_obj.exists():
            shutil.rmtree(unzip_path_obj)
        unzip_path_obj.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path_obj, 'r') as zip_handle:
            zip_handle.extractall(unzip_path_obj)
        origin_md_path_obj = unzip_path_obj / f"full.md"
        new_md_path_obj = origin_md_path_obj.with_name(f"{pdf_path_obj.stem}.md")
        origin_md_path_obj.rename(new_md_path_obj)
        with open(new_md_path_obj, 'r', encoding='utf-8') as f:
            md_content = f.read()
        return md_content, new_md_path_obj

    def get_md_zip_url(self, batch_id, token):
        url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        max_time = 120
        start_time = 0
        current_time = 0
        while True:
            try:
                start_time = time.time()
                response = requests.get(url, headers=header)
                if response.status_code != 200:
                    logger.error(f"文件下载请求失败，状态码：{response.status_code}")
                    raise Exception(f"文件下载请求失败，状态码：{response.status_code}")
                result = response.json()
                if result["code"] != 0:
                    logger.error(f"文件下载失败，code:{result['code']}")
                    raise Exception(f"文件下载失败，code:{result['code']}")
                data = result["data"]
                if data["extract_result"][0]["state"] != "done":
                    logger.warning(f"文件下载解析未完成，等待几秒后重试")
                    raise Exception(f"文件下载解析未完成，等待几秒后重试")
                zip_url = data["extract_result"][0]["full_zip_url"]
                break
            except Exception as e:
                end_time = time.time()
                current_time += (end_time - start_time)
                if current_time > max_time:
                    logger.error(f"文件下载解析超时，请检查文件上传是否成功")
                    raise Exception(f"文件下载解析超时，请检查文件上传是否成功")
                time.sleep(1)
                continue
        return zip_url

    def upload_pdf_mineru(self, pdf_path, pdf_path_obj):
        token = MineruConfig.mineru_token
        url = "https://mineru.net/api/v4/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name, "data_id": "abcd", "is_ocr": True}
            ],
            "model_version": "vlm"
        }
        file_path = [pdf_path]
        response = requests.post(url, headers=header, json=data)
        if response.status_code != 200:
            logger.error(f"请求失败，状态码：{response.status_code}")
            raise Exception(f"请求失败，状态码：{response.status_code}")
        result = response.json()
        if result["code"] != 0:
            logger.error(f"请求失败，code:{result['code']}，原因：{result['msg']}")
            raise Exception(f"请求失败，code:{result['code']}，原因：{result['msg']}")
        batch_id = result["data"]["batch_id"]
        urls = result["data"]["file_urls"]
        for i in range(0, len(urls)):
            with open(file_path[i], 'rb') as f:
                res_upload = requests.put(urls[i], data=f)
                if res_upload.status_code == 200:
                    logger.info(f"{file_path[i]}上传成功")
                else:
                    logger.error(f"{file_path[i]}上传失败")
                    raise Exception(f"{file_path[i]}上传失败")
        return batch_id, token

    def check_pdf(self, state):
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            logger.error(f"pdf路径{pdf_path}为空")
            raise Exception(f"pdf路径{pdf_path}为空")
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            logger.error(f"{pdf_path}文件不存在")
            raise Exception(f"{pdf_path}文件不存在")
        local_dir = state.get("local_dir")
        if not local_dir:
            logger.error(f"输出路径{local_dir}为空")
            raise Exception(f"输出路径{local_dir}为空")
        local_dir_obj = Path(local_dir)
        if not local_dir_obj.exists():
            local_dir_obj.mkdir(parents=True, exist_ok=True)
        return local_dir_obj, pdf_path, pdf_path_obj


if __name__ == '__main__':
    init_state = {
        "pdf_path": r"D:\Study\python\knowledge_base\doc\hak180产品安全手册.pdf",
        "local_dir": r"D:\Study\python\knowledge_base\output"
    }
    node_pdf_to_md = NodePDFToMD()
    res = node_pdf_to_md(init_state)
    logger.info(res)
