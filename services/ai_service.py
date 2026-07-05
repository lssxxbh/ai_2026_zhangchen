import json
import re
from typing import Optional, Dict, Any
from openai import AsyncOpenAI
from config import settings
from utils.logger import logger


class AIService:
    def __init__(self):
        self.client = None
        
        logger.info("="*60)
        logger.info("LLM 配置信息:")
        logger.info(f"  Provider: {settings.LLM_PROVIDER}")
        logger.info(f"  API Base: {settings.LLM_API_BASE}")
        logger.info(f"  Model: {settings.LLM_MODEL}")
        logger.info(f"  API Key: {'已配置' if settings.ALIYUN_API_KEY else '未配置'}")
        logger.info("="*60)
        
        if settings.ALIYUN_API_KEY:
            self.client = AsyncOpenAI(
                api_key=settings.ALIYUN_API_KEY,
                base_url=settings.LLM_API_BASE
            )
            logger.info(f"LLM初始化成功: provider={settings.LLM_PROVIDER}, model={settings.LLM_MODEL}")
        else:
            logger.warning("未配置ALIYUN_API_KEY环境变量，将使用本地规则解析")

    async def parse_report(self, text: str, user_question: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not self.client or not settings.ALIYUN_API_KEY:
            logger.warning("LLM未配置或无API密钥，使用本地规则解析")
            return self._mock_parse(text, user_question)

        prompt = self._build_prompt(text, user_question)

        try:
            logger.info(f"开始调用LLM: {settings.LLM_MODEL}")
            
            response = await self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """你是一名专业的医学信息抽取专家。请从体检报告文本中抽取信息，输出严格的JSON格式，不包含任何额外说明文字。只输出JSON，不要输出其他任何内容。"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=3000
            )

            content = response.choices[0].message.content
            if content:
                result = self._parse_json_response(content)
                # 只在有文本提取时才添加 ocr 信息
                if text:
                    result["ocr"] = {
                        "source": "llm",
                        "text_extracted": True,
                        "text_length": len(text),
                        "raw_text": text[:1000]
                    }
                logger.info(f"LLM解析成功")
                return result

            # 如果没有结果，只返回 ocr 信息
            result = {}
            if text:
                result["ocr"] = {
                    "source": "llm",
                    "text_extracted": False,
                    "text_length": len(text),
                    "raw_text": text[:1000]
                }
            return result
        except Exception as e:
            logger.error(f"LLM调用错误: {e}", exc_info=True)
            logger.info("降级到本地规则解析")
            return self._mock_parse(text, user_question)

    def _build_prompt(self, text: str, user_question: Optional[str] = None) -> str:
        prompt = f"""你是一名医学信息抽取专家。

请从以下体检报告文本中抽取信息并输出JSON。

重要要求：
1. 只抽取报告中真实存在的信息，没有的字段不要输出
2. 不要编造数据
3. 保持医学语义准确
4. 数值不修改，单位不修改
5. 只输出JSON，不输出任何其他解释文字！

以下是一个参考结构，你可以根据实际内容选择使用哪些字段：
{{
  "patient_info": {{
    "name": "姓名",
    "gender": "性别",
    "age": "年龄",
    "check_date": "检查日期",
    "hospital": "医院"
  }},
  "vital_signs": {{
    "height_cm": 身高数值,
    "weight_kg": 体重数值,
    "bmi": BMI数值,
    "blood_pressure": "血压",
    "pulse": "脉搏"
  }},
  "laboratory": [
    {{
      "name": "检验项目名称",
      "value": "结果值",
      "unit": "单位",
      "reference": "参考范围",
      "flag": "异常标识（H/L/N）"
    }}
  ],
  "summary": {{
    "health_conclusion": "健康总结",
    "recommendation": ["建议1", "建议2"]
  }}
}}

体检报告文本：
{text}
"""
        if user_question:
            prompt += f"\n用户问题：{user_question}\n"

        prompt += "\nJSON 输出："
        return prompt

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        try:
            json_str = content
            json_start = content.find("{")
            json_end = content.rfind("}")
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end + 1]
            result = json.loads(json_str)
            # 移除所有值为 null 或空数组的字段
            result = self._clean_empty_fields(result)
            return result
        except Exception as e:
            logger.error(f"JSON解析错误: {e}", exc_info=True)
            return {"raw_response": content}

    def _clean_empty_fields(self, data: Any) -> Any:
        """递归移除空值字段"""
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                cleaned_value = self._clean_empty_fields(value)
                # 只保留非 null 和非空数组/对象的值
                if cleaned_value is not None:
                    if isinstance(cleaned_value, (list, dict)):
                        if len(cleaned_value) > 0:
                            cleaned[key] = cleaned_value
                    else:
                        cleaned[key] = cleaned_value
            return cleaned
        elif isinstance(data, list):
            cleaned = []
            for item in data:
                cleaned_item = self._clean_empty_fields(item)
                if cleaned_item is not None:
                    if isinstance(cleaned_item, (list, dict)):
                        if len(cleaned_item) > 0:
                            cleaned.append(cleaned_item)
                    else:
                        cleaned.append(cleaned_item)
            return cleaned if cleaned else None
        else:
            return data

    def _mock_parse(self, text: str, user_question: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"开始本地规则解析，文本长度: {len(text)}")
        
        result = {}
        # 始终添加 ocr 信息
        if text:
            result["ocr"] = {
                "source": "local_rule_parser",
                "text_extracted": True,
                "text_length": len(text),
                "raw_text": text[:1000]
            }
        
        patient_info = {}
        vital_signs = {}
        laboratory = []
        
        if text:
            # 清理文本，去掉多余空格和换行
            clean_text = re.sub(r'\s+', ' ', text).strip()
            logger.info(f"清理后文本前200字符: {repr(clean_text[:200])}")
            
            # 姓名
            name_match = re.search(r'(?:姓名|Name)[：:\s]*([^\s，。,]{2,8})', clean_text)
            if name_match:
                patient_info["name"] = name_match.group(1).strip()
            
            # 性别
            gender_match = re.search(r'(?:性别|Gender)[：:\s]*([男女男女malefemale]{1,6})', clean_text, re.IGNORECASE)
            if gender_match:
                patient_info["gender"] = gender_match.group(1).strip()
            
            # 年龄
            age_match = re.search(r'(?:年龄|Age)[：:\s]*(\d{1,3})', clean_text)
            if age_match:
                patient_info["age"] = age_match.group(1)
            
            # 检查日期
            date_match = re.search(r'(?:日期|Date|检查日期)[：:\s]*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})', clean_text)
            if date_match:
                patient_info["check_date"] = date_match.group(1)
            
            # 医院
            hospital_match = re.search(r'(?:医院|Hospital)[：:\s]*([^\s，。,]{2,20})', clean_text)
            if hospital_match:
                patient_info["hospital"] = hospital_match.group(1).strip()
            
            # 身高
            height_match = re.search(r'(?:身高|Height)[：:\s]*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?', clean_text, re.IGNORECASE)
            if height_match:
                vital_signs["height_cm"] = float(height_match.group(1))
            
            # 体重
            weight_match = re.search(r'(?:体重|Weight)[：:\s]*(\d+(?:\.\d+)?)\s*(?:kg|千克)?', clean_text, re.IGNORECASE)
            if weight_match:
                vital_signs["weight_kg"] = float(weight_match.group(1))
                if vital_signs.get("height_cm"):
                    height_m = vital_signs["height_cm"] / 100
                    vital_signs["bmi"] = round(vital_signs["weight_kg"] / (height_m * height_m), 1)
            
            # 血压
            bp_match = re.search(r'(?:血压|Blood Pressure|BP)[：:\s]*(\d{2,3})[/-](\d{2,3})\s*(?:mmHg)?', clean_text, re.IGNORECASE)
            if bp_match:
                vital_signs["blood_pressure"] = f"{bp_match.group(1)}/{bp_match.group(2)}"
            
            # 心率/脉搏
            hr_match = re.search(r'(?:心率|脉搏|Heart Rate|HR|Pulse)[：:\s]*(\d{2,3})\s*(?:次/分|bpm|beats)?', clean_text, re.IGNORECASE)
            if hr_match:
                vital_signs["pulse"] = hr_match.group(1)
            
            # 简单的实验室指标提取
            lab_patterns = [
                (r'(?:空腹)?(?:血糖|Glucose)[：:\s]*(\d+(?:\.\d+)?)\s*(?:mmol/L|mg/dL)?', "空腹血糖", "mmol/L"),
                (r'(?:总胆固醇|Cholesterol|TC)[：:\s]*(\d+(?:\.\d+)?)\s*(?:mmol/L)?', "总胆固醇", "mmol/L"),
                (r'(?:甘油三酯|Triglyceride|TG)[：:\s]*(\d+(?:\.\d+)?)\s*(?:mmol/L)?', "甘油三酯", "mmol/L"),
                (r'(?:血红蛋白|Hb|HGB)[：:\s]*(\d{2,3})\s*(?:g/L)?', "血红蛋白", "g/L"),
                (r'(?:白细胞|WBC)[：:\s]*(\d+(?:\.\d+)?)\s*(?:\*10\^9/L|×10⁹/L)?', "白细胞计数", "×10⁹/L"),
                (r'(?:红细胞|RBC)[：:\s]*(\d+(?:\.\d+)?)\s*(?:\*10\^12/L|×10¹²/L)?', "红细胞计数", "×10¹²/L"),
                (r'(?:血小板|PLT)[：:\s]*(\d{2,3})\s*(?:\*10\^9/L|×10⁹/L)?', "血小板计数", "×10⁹/L")
            ]
            
            for pattern, name, unit in lab_patterns:
                match = re.search(pattern, clean_text, re.IGNORECASE)
                if match:
                    laboratory.append({
                        "name": name,
                        "value": match.group(1),
                        "unit": unit
                    })
            
            # 体检结论
            conclusion_match = re.search(r'(?:体检)?(?:结论|Summary|Conclusion)[：:\s]*(.+?)(?=\s{2,}|\Z)', clean_text, re.DOTALL)
            if conclusion_match:
                result["summary"] = {
                    "health_conclusion": conclusion_match.group(1).strip()
                }
        
        # 更新结果，只添加有内容的字段
        if patient_info:
            result["patient_info"] = patient_info
        
        if vital_signs:
            result["vital_signs"] = vital_signs
        
        if laboratory:
            result["laboratory"] = laboratory
        
        logger.info(f"本地规则解析完成")
        return result
