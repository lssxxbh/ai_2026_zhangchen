# ============================================================
# 强制加载项目根目录 .env
# 必须放在所有业务配置读取之前
# ============================================================
from pathlib import Path
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# app.py
# -*- coding: utf-8 -*-

from contextlib import asynccontextmanager
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api import (
    auth_router,
    chat_router,
    history_router,
    graph_router,
    multimodal_router,
)
from utils import logger


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Medical AI System A...")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOADS_DIR / "images").mkdir(parents=True, exist_ok=True)
    (UPLOADS_DIR / "compressed").mkdir(parents=True, exist_ok=True)

    yield

    logger.info("Shutting down Medical AI System A...")


app = FastAPI(
    title="智能体检报告解析系统 A API",
    description="A 项目：前端、上传、OCR、多模态汇总，并调用 B 项目 MedRAG 报告接口",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# API routers
# =========================

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(history_router)
app.include_router(graph_router)
app.include_router(multimodal_router)


# =========================
# Static frontend / uploads
# =========================

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    logger.warning(f"前端静态目录不存在: {STATIC_DIR}")

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


def page_response(filename: str):
    file_path = FRONTEND_DIR / filename

    if file_path.exists():
        return FileResponse(
            str(file_path),
            media_type="text/html; charset=utf-8",
        )

    return JSONResponse(
        {
            "code": 404,
            "msg": f"frontend/{filename} not found",
            "expected_path": str(file_path),
            "docs": "/docs",
        },
        status_code=404,
    )


@app.get("/")
async def frontend_index():
    return page_response("index.html")


@app.get("/frontend")
async def frontend_alias():
    return page_response("index.html")


@app.get("/index.html")
async def frontend_index_html():
    return page_response("index.html")


@app.get("/intro.html")
async def frontend_intro_html():
    return page_response("intro.html")


@app.get("/login.html")
async def frontend_login_html():
    return page_response("login.html")


@app.get("/upload.html")
async def frontend_upload_html():
    return page_response("upload.html")


@app.get("/report.html")
async def frontend_report_html():
    return page_response("report.html")


@app.get("/kg.html")
async def frontend_kg_html():
    return page_response("kg.html")


@app.get("/stats.html")
async def frontend_stats_html():
    return page_response("stats.html")


@app.get("/advice.html")
async def frontend_advice_html():
    return page_response("advice.html")


@app.get("/image.html")
async def frontend_image_html():
    return page_response("image.html")


@app.get("/blood.html")
async def frontend_blood_html():
    return page_response("blood.html")


# =========================
# Health / API info
# =========================

@app.get("/api")
async def api_root():
    return {
        "message": "智能体检报告解析系统 A API",
        "version": "1.0.0",
        "docs": "/docs",
        "frontend": "/",
        "port": 8001,
    }


@app.get("/api/health")
async def api_health():
    return {
        "status": "healthy",
        "service": "medical-ai-api-A",
        "port": 8001,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "A",
    }


if __name__ == "__main__":
    host = os.getenv("A_BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("A_BACKEND_PORT", "8001"))

    # 关键修复：
    # 1. 这里必须传 app 对象，不能传 "app:app"
    #    否则 python app.py 时会执行当前文件一次，
    #    uvicorn 再 import app 执行第二次，导致路由/初始化日志重复。
    #
    # 2. reload 必须 False
    #    否则会启动 reloader 进程，EasyOCR / LLM 会初始化两遍。
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
