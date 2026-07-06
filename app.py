from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api import auth_router, chat_router, history_router, graph_router
from utils import logger


BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"
STATIC_DIR = FRONTEND_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Medical AI System...")
    yield
    logger.info("Shutting down Medical AI System...")


app = FastAPI(
    title="智能体检报告解析系统 API",
    description="一个用于解析体检报告并提供 AI 对话功能的系统",
    version="1.0.0",
    lifespan=lifespan
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


# =========================
# Static frontend
# =========================
# 如果以后你把 css/js/image 拆到 frontend/static 里，可以直接通过 /static/... 访问
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def frontend_index():
    """
    新前端入口。
    访问 http://localhost:8000/ 会直接打开 frontend/index.html
    """
    if INDEX_FILE.exists():
        return FileResponse(
            str(INDEX_FILE),
            media_type="text/html; charset=utf-8"
        )

    return JSONResponse(
        {
            "code": 404,
            "msg": "frontend/index.html not found",
            "expected_path": str(INDEX_FILE),
            "docs": "/docs"
        },
        status_code=404
    )


@app.get("/frontend")
async def frontend_alias():
    """
    前端备用入口。
    """
    if INDEX_FILE.exists():
        return FileResponse(
            str(INDEX_FILE),
            media_type="text/html; charset=utf-8"
        )

    return JSONResponse(
        {
            "code": 404,
            "msg": "frontend/index.html not found",
            "expected_path": str(INDEX_FILE),
            "docs": "/docs"
        },
        status_code=404
    )


@app.get("/index.html")
async def frontend_index_html():
    """
    兼容直接访问 /index.html。
    """
    if INDEX_FILE.exists():
        return FileResponse(
            str(INDEX_FILE),
            media_type="text/html; charset=utf-8"
        )

    return JSONResponse(
        {
            "code": 404,
            "msg": "frontend/index.html not found",
            "expected_path": str(INDEX_FILE),
            "docs": "/docs"
        },
        status_code=404
    )


# =========================
# Health / API info
# =========================

@app.get("/api")
async def api_root():
    return {
        "message": "智能体检报告解析系统 API",
        "version": "1.0.0",
        "docs": "/docs",
        "frontend": "/"
    }


@app.get("/api/health")
async def api_health():
    return {
        "status": "healthy",
        "service": "medical-ai-api"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
