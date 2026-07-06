from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).parent
    
    DATABASE_URL: str = "mysql+pymysql://root:123456@localhost:3306/medical_ai?charset=utf8mb4"
    
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
    
    # 阿里云百联API配置 - 优先从环境变量读取
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "aliyun")
    # 从 OPENAI_API_KEY 环境变量读取，兼容标准 OpenAI 命名
    ALIYUN_API_KEY: str = os.getenv("OPENAI_API_KEY", os.getenv("ALIYUN_API_KEY", ""))
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen-plus")  # 可以使用 qwen-turbo, qwen-plus, qwen-max 等
    
    # 外部报告生成 API 配置
    REPORT_GENERATION_API_URL: str = os.getenv(
        "REPORT_GENERATION_API_URL",
        "http://10.187.79.121:8000/api/v1/reports/from-ocr-json/simple"
    )
    
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    
    class Config:
        env_file = ".env"
        case_sensitive = True  # 区分大小写


settings = Settings()
settings.UPLOAD_DIR.mkdir(exist_ok=True)
