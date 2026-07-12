# config.py
# -*- coding: utf-8 -*-

from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv
import os


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"

# 强制读取项目根目录 .env，并覆盖系统环境变量中的旧值
load_dotenv(dotenv_path=ENV_PATH, override=True)


class Settings(BaseSettings):
    BASE_DIR: Path = PROJECT_ROOT

    # =========================
    # 数据库配置
    # =========================
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://root:Rsy%40050914@localhost:3306/medical_ai?charset=utf8mb4"
    )

    # =========================
    # JWT 配置
    # =========================
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY",
        "your-super-secret-key-change-in-production"
    )
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", str(60 * 24 * 7))
    )

    # =========================
    # 主 LLM 配置
    # 当前启用：硅基流动 SiliconFlow，OpenAI-compatible 协议
    # =========================
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")

    # 注意：虽然变量名历史上叫 ALIYUN_API_KEY，但这里实际优先读取 OPENAI_API_KEY
    # OPENAI_API_KEY 填的是硅基流动 Key
    ALIYUN_API_KEY: str = os.getenv(
        "OPENAI_API_KEY",
        os.getenv("ALIYUN_API_KEY", "")
    )

    LLM_API_BASE: str = os.getenv(
        "LLM_API_BASE",
        "https://api.siliconflow.cn/v1"
    )

    LLM_MODEL: str = os.getenv(
        "LLM_MODEL",
        "Qwen/Qwen3.5-9B"
    )

    # =========================
    # 外部 B 工程 / MedRAG 报告生成 API
    # =========================
    REPORT_GENERATION_API_URL: str = os.getenv(
        "REPORT_GENERATION_API_URL",
        "http://127.0.0.1:8000/api/v1/reports/from-ocr-json/simple"
    )

    # =========================
    # 硅基流动多模态配置
    # =========================
    SILICONFLOW_API_KEY: str = os.getenv(
        "SILICONFLOW_API_KEY",
        os.getenv("OPENAI_API_KEY", "")
    )

    SILICONFLOW_BASE_URL: str = os.getenv(
        "SILICONFLOW_BASE_URL",
        "https://api.siliconflow.cn/v1"
    )

    # 影像模型
    VL_MODEL: str = os.getenv(
        "VL_MODEL",
        "Qwen/Qwen3-VL-32B-Instruct"
    )

    # 血常规 / 检验文本模型
    BLOOD_MODEL: str = os.getenv(
        "BLOOD_MODEL",
        "Qwen/Qwen3.5-9B"
    )

    # 影像 + 血常规综合模型
    COMBINE_MODEL: str = os.getenv(
        "COMBINE_MODEL",
        "Qwen/Qwen3.5-9B"
    )

    # 健康建议 / 科室 / 检查建议模型
    ADVICE_MODEL: str = os.getenv(
        "ADVICE_MODEL",
        "Qwen/Qwen3.5-9B"
    )

    # =========================
    # Token 与温度
    # =========================
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "3000"))
    BLOOD_MAX_TOKENS: int = int(os.getenv("BLOOD_MAX_TOKENS", "2500"))
    COMBINE_MAX_TOKENS: int = int(os.getenv("COMBINE_MAX_TOKENS", "2500"))
    ADVICE_MAX_TOKENS: int = int(os.getenv("ADVICE_MAX_TOKENS", "1800"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))

    # =========================
    # 图片压缩配置
    # =========================
    IMAGE_MAX_SIDE: int = int(os.getenv("IMAGE_MAX_SIDE", "1280"))
    MAX_IMAGE_COUNT: int = int(os.getenv("MAX_IMAGE_COUNT", "6"))
    JPEG_QUALITY: int = int(os.getenv("JPEG_QUALITY", "85"))

    # =========================
    # 上传目录
    # =========================
    UPLOAD_DIR: Path = BASE_DIR / "uploads"

    class Config:
        env_file = str(ENV_PATH)
        case_sensitive = True
        extra = "ignore"


settings = Settings()

settings.UPLOAD_DIR.mkdir(exist_ok=True)
(settings.UPLOAD_DIR / "images").mkdir(parents=True, exist_ok=True)
(settings.UPLOAD_DIR / "compressed").mkdir(parents=True, exist_ok=True)
