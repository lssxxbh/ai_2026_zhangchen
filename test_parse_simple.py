import json
import re

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

result = {}

if test_content:
    info = {}
    
    print(f"解析文本前200字符: {repr(test_content[:200])}")
    
    # 清理文本，去掉多余空格和换行
    clean_text = re.sub(r'\s+', ' ', test_content).strip()
    print(f"清理后文本前200字符: {repr(clean_text[:200])}")
    
    # 姓名 - 更宽松的匹配
    name_match = re.search(r'(?:姓名|Name)[：:\s]*([^\s，。,]{2,8})', clean_text)
    if name_match:
        info["name"] = name_match.group(1).strip()
        print(f"找到姓名: {info['name']}")
    
    # 性别
    gender_match = re.search(r'(?:性别|Gender)[：:\s]*([男女男女malefemale]{1,6})', clean_text, re.IGNORECASE)
    if gender_match:
        info["gender"] = gender_match.group(1).strip()
        print(f"找到性别: {info['gender']}")
    
    # 年龄
    age_match = re.search(r'(?:年龄|Age)[：:\s]*(\d{1,3})', clean_text)
    if age_match:
        info["age"] = int(age_match.group(1))
        print(f"找到年龄: {info['age']}")
    
    # 身高
    height_match = re.search(r'(?:身高|Height)[：:\s]*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?', clean_text, re.IGNORECASE)
    if height_match:
        info["height"] = {"value": float(height_match.group(1)), "unit": "cm"}
        print(f"找到身高: {info['height']}")
    
    # 体重
    weight_match = re.search(r'(?:体重|Weight)[：:\s]*(\d+(?:\.\d+)?)\s*(?:kg|千克)?', clean_text, re.IGNORECASE)
    if weight_match:
        info["weight"] = {"value": float(weight_match.group(1)), "unit": "kg"}
        print(f"找到体重: {info['weight']}")
    
    # 血压
    bp_match = re.search(r'(?:血压|Blood Pressure|BP)[：:\s]*(\d{2,3})[/-](\d{2,3})\s*(?:mmHg)?', clean_text, re.IGNORECASE)
    if bp_match:
        info["blood_pressure"] = {
            "systolic": int(bp_match.group(1)),
            "diastolic": int(bp_match.group(2)),
            "unit": "mmHg"
        }
        print(f"找到血压: {info['blood_pressure']}")
    
    # 心率
    hr_match = re.search(r'(?:心率|Heart Rate|HR)[：:\s]*(\d{2,3})\s*(?:次/分|bpm|beats)?', clean_text, re.IGNORECASE)
    if hr_match:
        info["heart_rate"] = {"value": int(hr_match.group(1)), "unit": "bpm"}
        print(f"找到心率: {info['heart_rate']}")
    
    # 空腹血糖
    glucose_match = re.search(r'(?:空腹)?(?:血糖|Glucose)[：:\s]*(\d+(?:\.\d+)?)\s*(?:mmol/L|mg/dL)?', clean_text, re.IGNORECASE)
    if glucose_match:
        info["fasting_blood_glucose"] = {"value": float(glucose_match.group(1)), "unit": "mmol/L"}
        print(f"找到血糖: {info['fasting_blood_glucose']}")
    
    # 总胆固醇
    chol_match = re.search(r'(?:总胆固醇|Cholesterol|TC)[：:\s]*(\d+(?:\.\d+)?)\s*(?:mmol/L)?', clean_text, re.IGNORECASE)
    if chol_match:
        info["total_cholesterol"] = {"value": float(chol_match.group(1)), "unit": "mmol/L"}
        print(f"找到总胆固醇: {info['total_cholesterol']}")
    
    # 血红蛋白
    hb_match = re.search(r'(?:血红蛋白|Hb|HGB)[：:\s]*(\d{2,3})\s*(?:g/L)?', clean_text, re.IGNORECASE)
    if hb_match:
        info["hemoglobin"] = {"value": int(hb_match.group(1)), "unit": "g/L"}
        print(f"找到血红蛋白: {info['hemoglobin']}")
    
    if info:
        result.update(info)
        print(f"本地规则解析成功，提取到 {len(info)} 个字段")

print()
print("="*60)
print("解析结果：")
print("="*60)

print(json.dumps(result, ensure_ascii=False, indent=2))

