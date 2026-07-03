import asyncio
from pathlib import Path
from services.ai_service import AIService

# 测试内容（用户提供的）
test_content = """示例体检报告

姓名：张三
性别：男
年龄：28岁
身高：175cm
体重：68kg
血压：118/76mmHg
心率：72次/分
空腹血糖：5.2mmol/L
总胆固醇：4.8mmol/L
血红蛋白：145g/L

体检结论：各项指标基本正常，建议保持规律作息、均衡饮食、适量运动，每年定期体检。
"""

print("="*60)
print("测试本地规则解析")
print("="*60)
print()

print("测试内容：")
print("-" * 40)
print(test_content)
print("-" * 40)
print()

ai_service = AIService()

# 直接调用本地解析
result = ai_service._mock_parse(test_content)

print()
print("="*60)
print("解析结果：")
print("="*60)

import json
print(json.dumps(result, ensure_ascii=False, indent=2))

