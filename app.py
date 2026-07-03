from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from api import auth_router, chat_router, history_router
from utils import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Medical AI System...")
    yield
    logger.info("Shutting down Medical AI System...")


app = FastAPI(
    title="智能体检报告解析系统 API",
    description="一个用于解析体检报告并提供AI对话功能的系统",
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

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(history_router)


@app.get("/")
async def root():
    return {
        "message": "智能体检报告解析系统 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
