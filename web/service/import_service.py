import datetime
import shutil
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware

from processor.import_processor.main_graph import ImportMainGraphRunner
from tool.logger import logger, set_trace_id
from tool.minio_utils import get_minio_client
from tool.task_utils import add_running_task, add_done_task, update_task_status, TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, get_task_info

app = FastAPI(
    title="掌柜智库-导入API",
    description="此文档是掌柜智库导入流程的API接口说明",
)
# 2. 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的源
    allow_credentials=True,  # 允许携带cookie
    allow_methods=["*"],  # 允许的请求方法
    allow_headers=["*"],  # 允许的请求头
)


def run_main_graph(task_id, local_dir, local_file_path):
    # 绑定 trace_id（复用 task_id），后续所有日志自动携带，便于链路追踪
    set_trace_id(task_id)
    try:
        init_state = {
            "task_id": task_id,
            "local_dir": local_dir,
            "local_file_path": local_file_path
        }
        logger.info("导入任务开始，文件=%s", local_file_path)
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        ImportMainGraphRunner.create_and_run(init_state)
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        logger.info("导入任务完成")
    except Exception:
        logger.exception("导入任务失败")
        update_task_status(task_id, TASK_STATUS_FAILED)
        raise


@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())

    local_dir = rf"D:\Study\python\knowledge_base\output\{datetime.datetime.now().strftime('%Y%m%d')}"
    local_file_path = f"{local_dir}/{file.filename}"
    local_dir_path_obj = Path(local_dir)
    if not local_dir_path_obj.exists():
        local_dir_path_obj.mkdir(parents=True, exist_ok=True)
    add_running_task(task_id, "upload_file")
    with open(local_file_path, "wb") as fs:
        shutil.copyfileobj(file.file, fs, length=1024 * 1024)
    minio_client = get_minio_client()
    minio_client.fput_object(
        bucket_name="knowledge-base",
        object_name=f"pdf_file/{datetime.datetime.now().strftime('%Y%m%d')}/{file.filename}",
        file_path=local_file_path,
    )
    add_done_task(task_id, "upload_file")
    background_tasks.add_task(run_main_graph, task_id, local_dir, local_file_path)
    return {
        "task_id": task_id,
        "file_name": file.filename,
        "file_size": file.size
    }


@app.get("/status/{task_id}")
def get_task_status(task_id: str):
    return get_task_info(task_id)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
