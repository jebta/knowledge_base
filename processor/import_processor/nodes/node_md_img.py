import base64
import os
import re
import time
from datetime import datetime
from collections import deque
from pathlib import Path

from langchain.chat_models import init_chat_model
from minio.deleteobjects import DeleteObject


from config.config import LLMConfig, MinioConfig
from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState
from tool.json_format import json_format
from tool.logger import logger
from tool.minio_utils import get_minio_client


class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):
        # 获取图片所在路径，并检验路径是否存在
        image_path_obj, md_content, md_path_obj = self.get_image_path(state)
        # 如果没有图片路径就不处理，直接返回md_content
        if not image_path_obj.exists():
            return {
                "content": md_content
            }
        # 有文件夹但是图片也直接返回md_content
        image_list = os.listdir(image_path_obj)
        if not image_list:
            return {
                "content": md_content
            }
        # 获取图片上下文列表
        image_context_list = self.get_image_context_list(image_list, image_path_obj, md_content)
        # 根据qwen视觉模型 ，上传图片和上下文获取图片摘要
        image_summary_list = self.get_image_summary_list(image_context_list)
        # 上传图片到minio，并获取图片url
        image_summary_url_list = self.upload_image_minio(image_summary_list, md_path_obj)
        # 替换md中的图片摘要和url，备份到新文件中
        md_content, new_md_path_obj = self.replace_summary_and_url(image_summary_url_list, md_content, md_path_obj)
        return {
            "md_path": str(new_md_path_obj),
            "md_content": md_content
        }

    def replace_summary_and_url(self, image_summary_url_list, md_content, md_path_obj):
        for image_summary_url in image_summary_url_list:
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_summary_url.get("image_name")) + r"\)")
            md_content = pattern.sub(
                lambda _: f"![{image_summary_url.get('summary')}]({image_summary_url.get('image_url')})",
                md_content
            )
        new_md_path_obj = md_path_obj.parent / f"{md_path_obj.stem}_new.md"
        with open(new_md_path_obj, "w", encoding="utf-8") as f:
            f.write(md_content)
        return md_content, new_md_path_obj

    def upload_image_minio(self, image_summary_list, md_path_obj):
        minio_client = get_minio_client()
        upload_dir = MinioConfig.MINIO_IMG_DIR + '/' + datetime.now().strftime("%Y%m%d") + '/' + md_path_obj.stem
        old_image_generator = minio_client.list_objects(
            bucket_name=MinioConfig.MINIO_BUCKET_NAME
        )
        delete_object_list = [DeleteObject(item.object_name) for item in old_image_generator]
        errors = minio_client.remove_objects(
            bucket_name=MinioConfig.MINIO_BUCKET_NAME,
            delete_object_list=delete_object_list
        )
        for error in errors:
            logger.warning("Error occurred while deleting old images: %s", error)

        image_summary_url_list = []
        for image_summary in image_summary_list:
            minio_client.fput_object(
                bucket_name=MinioConfig.MINIO_BUCKET_NAME,
                object_name=upload_dir + "/" + image_summary.get("image_name"),
                file_path=image_summary.get("image_path")
            )
            image_url = f"http://{MinioConfig.MINIO_ENDPOINT}/{MinioConfig.MINIO_BUCKET_NAME}/{upload_dir}/{image_summary.get('image_name')}"
            image_summary_url_list.append({
                **image_summary,
                "image_url": image_url
            })
        return image_summary_url_list

    def get_image_summary_list(self, image_context_list):
        llm = init_chat_model(
            model=LLMConfig.VL_MODEL,
            model_provider="openai",
            base_url=LLMConfig.OPENAI_API_BASE,
            api_key=LLMConfig.OPENAI_API_KEY,
            temperature=LLMConfig.LLM_DEFAULT_TEMPERATURE
        )
        q = deque(maxlen=30)
        image_summary_list = []
        for image_context in image_context_list:
            while q and time.time() - q[0] > 60:
                q.popleft()
            if len(q) >= 30:
                wait_time = time.time() - q[0]
                if wait_time < 60:
                    time.sleep(60 - wait_time)
                while q and time.time() - q[0] > 60:
                    q.popleft()
            q.append(time.time())
            with open(image_context.get("image_path"), "rb") as f:
                image_data = f.read()
            base64_data = base64.b64encode(image_data).decode("utf-8")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": f"""这是一张图片，图片上文部分为"{image_context.get("pre_text")}"，
                                            下文部分为"{image_context.get("post_text")}"，请用中文简要总结这张图片的摘要,字数在50字以内。"""},
                    ],
                },
            ]
            res = llm.invoke(messages)

            image_summary_list.append({
                "image_name": image_context.get("image_name"),
                "image_path": image_context.get("image_path"),
                "summary": res.content
            })
        return image_summary_list

    def get_image_context_list(self, image_list, image_path_obj, md_content):
        image_context_list = []
        for image_name in image_list:
            IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
            suffix = Path(image_name).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r"\)")
            match = pattern.search(md_content)
            if not match:
                logger.warning(f"图片{image_name}在md文件中未找到")
                continue
            start, end = match.span()
            pre_text = md_content[max(start - 250, 0):start]
            post_text = md_content[end:end + 250]
            image_context_list.append({
                "image_name": image_name,
                "image_path": str(image_path_obj / image_name),
                "pre_text": pre_text,
                "post_text": post_text,
            })
        return image_context_list

    def get_image_path(self, state):
        md_path = state.get("md_path")
        if not md_path:
            logger.error(f"md_path路径为空")
            raise Exception(f"md_path路径为空")
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            logger.error(f"{md_path}文件不存在")
            raise Exception(f"{md_path}文件不存在")
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        image_path_obj = md_path_obj.parent / "images"
        return image_path_obj, md_content, md_path_obj


if __name__ == '__main__':
    init_state = {
        "md_path": "D:\Study\python\knowledge_base\output\hak180产品安全手册\hak180产品安全手册.md",
    }
    node_md_img = NodeMDImg()
    res = node_md_img(init_state)
    print(res)
