import json
import uuid

import uvicorn
from fastapi import FastAPI, Path, BackgroundTasks, Body
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from processor.query_processor.main_graph import QueryMainGraphRunner
from tool.logger import logger, set_trace_id
from tool.mongo_utils import get_recent_messages, clear_history
from tool.task_utils import create_queue, update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, \
    add_running_task, put_data, get_task_info, TASK_STATUS_FAILED, get_data

app = FastAPI(
    title="Query Service",
    description="Query Service for the application",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的源
    allow_credentials=True,  # 允许携带cookie
    allow_methods=["*"],  # 允许的请求方法
    allow_headers=["*"],  # 允许的请求头
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/history/{session_id}")
def get_history(session_id: str = Path(..., description="会话id")):
    history = get_recent_messages(session_id)
    for item in history:
        item["_id"] = str(item["_id"])
    history.sort(key=lambda x: x["ts"])
    return {"items": history}


@app.delete("/history/{session_id}")
def delete_history(session_id: str = Path(..., description="会话id")):
    clear_history(session_id)
    return {"status": "ok"}


def run_main_graph(task_id, original_query, session_id):
    # 绑定 trace_id（复用 task_id），后续所有日志自动携带，便于链路追踪
    set_trace_id(task_id)
    try:
        init_state = {
            "task_id": task_id,
            "original_query": original_query,
            "session_id": session_id,
        }
        logger.info("查询任务开始，query=%s", original_query)
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        put_data(task_id, "progress", get_task_info(task_id))

        QueryMainGraphRunner.create_and_run(init_state)

        update_task_status(task_id, TASK_STATUS_COMPLETED)
        put_data(task_id, "progress", get_task_info(task_id))
        logger.info("查询任务完成")
    except Exception:
        logger.exception("查询任务失败")
        update_task_status(task_id, TASK_STATUS_FAILED)
        put_data(task_id, "error", get_task_info(task_id))
        raise


class QueryParams(BaseModel):
    query: str = Field(..., description="用户问题")
    session_id: str = Field(..., description="会话id")


@app.post("/query")
def query(background_tasks: BackgroundTasks, query_params: QueryParams = Body(..., description="用户提问参数")):
    task_id = str(uuid.uuid4())[:8]
    session_id = query_params.session_id
    original_query = query_params.query
    create_queue(task_id)

    background_tasks.add_task(run_main_graph, task_id, original_query, session_id)
    return {
        "task_id": task_id,
        "session_id": session_id,
    }


@app.get("/stream/{task_id}")
def stream(task_id: str = Path(..., description="任务id")):
    def generate():
        while True:
            data = get_data(task_id)
            yield f"event: {data.get('event')}\n"
            yield f"data: {json.dumps(data.get('data'), ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)
