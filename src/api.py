"""FastAPI 入口（演示用）：POST /ask, GET /healthz。

curl 示例：
  curl -X POST http://127.0.0.1:8000/ask \
       -H 'Content-Type: application/json' \
       -d '{"question": "ERR_1042 是什么原因？"}'
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from .cli import build_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动即初始化，失败直接进程退出——比"首请求时才炸"更诚实
    app.state.pipeline = build_pipeline(".env")
    yield


app = FastAPI(title="mini-rag", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: dict):
    question = (payload or {}).get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")
    try:
        ans = app.state.pipeline.ask(question)
    except Exception as e:  # API 边界统一兜底为 500，from e 保留原始异常链
        raise HTTPException(status_code=500, detail=str(e)) from e
    return asdict(ans)
