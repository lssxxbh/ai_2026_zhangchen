import os
from pathlib import Path
from config import settings

print("="*60)
print("环境变量诊断工具")
print("="*60)

# 检查 .env 文件
env_path = Path(__file__).parent / ".env"
print(f"\n1. 检查 .env 文件: {env_path}")
print(f"   存在: {env_path.exists()}")
if env_path.exists():
    content = env_path.read_text(encoding='utf-8')
    print(f"   内容:")
    for line in content.strip().split('\n'):
        print(f"      {line}")

# 检查系统环境变量
print("\n2. 检查系统环境变量:")
env_vars = [
    'ALIYUN_API_KEY',
    'LLM_API_BASE',
    'LLM_MODEL',
    'LLM_PROVIDER'
]
for var in env_vars:
    value = os.environ.get(var)
    print(f"   {var}: {'已配置' if value else '未配置'}")
    if value and var != 'ALIYUN_API_KEY':
        print(f"      值: {value}")

# 检查加载的配置
print("\n3. 检查加载的配置:")
print(f"   LLM_PROVIDER: {settings.LLM_PROVIDER}")
print(f"   ALIYUN_API_KEY: {'已配置' if settings.ALIYUN_API_KEY else '未配置'}")
print(f"   LLM_API_BASE: {settings.LLM_API_BASE}")
print(f"   LLM_MODEL: {settings.LLM_MODEL}")

print("\n" + "="*60)
