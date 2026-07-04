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
                result["ocr"] = {
                    "source": "llm",
                    "file_type": None,
                    "text_extracted": True,
                    "text_length": len(text),
                    "parse_confidence": None,
                    "raw_text": text[:1000]
                }
                logger.info(f"LLM解析成功")
                return result

            return self._get_empty_result(text)
        except Exception as e:
            logger.error(f"LLM调用错误: {e}", exc_info=True)
            logger.info("降级到本地规则解析")
            return self._mock_parse(text, user_question)

    def _build_prompt(self, text: str, user_question: Optional[str] = None) -> str:
        prompt = f"""你是一名医学信息抽取专家。

请从以下体检报告文本中抽取信息并输出JSON，使用以下结构：

{{
  "patient_info": {{
    "patient_id": "患者ID",
    "name": "姓名",
    "gender": "性别",
    "age": "年龄",
    "birthday": "生日",
    "phone": "电话",
    "check_date": "检查日期",
    "hospital": "医院"
  }},
  "chief_complaint": "主诉",
  "present_illness": "现病史",
  "symptoms": ["症状1", "症状2"],
  "vital_signs": {{
    "height_cm": 身高数值,
    "weight_kg": 体重数值,
    "bmi": BMI数值,
    "temperature": "体温",
    "blood_pressure": "血压",
    "pulse": "脉搏",
    "respiration": "呼吸",
    "spo2": "血氧饱和度"
  }},
  "medical_history": {{
    "past_history": ["既往病史1", "既往病史2"],
    "family_history": ["家族史1", "家族史2"],
    "allergy_history": ["过敏史1", "过敏史2"],
    "medications": ["用药1", "用药2"],
    "vaccination": ["疫苗接种1", "疫苗接种2"],
    "surgery_history": ["手术史1", "手术史2"],
    "smoking": "吸烟情况",
    "drinking": "饮酒情况"
  }},
  "physical_examination": [
    {{
      "system": "系统名称",
      "finding": "检查发现",
      "normal": true/false
    }}
  ],
  "laboratory": [
    {{
      "item": "检验项目编码",
      "name": "检验项目名称",
      "value": "结果值",
      "raw_value": "原始值",
      "unit": "单位",
      "reference": "参考范围",
      "flag": "异常标识（H/L/N）",
      "valid": true/false
    }}
  ],
  "urine_test": {{
    "protein": "尿蛋白",
    "glucose": "尿糖",
    "occult_blood": "尿潜血",
    "ketone": "尿酮体",
    "leukocyte": "尿白细胞"
  }},
  "imaging": [
    {{
      "type": "检查类型",
      "finding": "影像表现",
      "conclusion": "影像结论"
    }}
  ],
  "ecg": {{
    "finding": "心电图表现",
    "conclusion": "心电图结论"
  }},
  "diagnosis": {{
    "clinical_diagnosis": ["诊断1", "诊断2"],
    "doctor_conclusion": "医生结论"
  }},
  "clinical_features": ["临床特征1", "临床特征2"],
  "kg_query": {{
    "symptoms": ["症状1", "症状2"],
    "laboratory_abnormal": ["异常指标1", "异常指标2"],
    "risk_factors": ["风险因素1", "风险因素2"]
  }},
  "summary": {{
    "health_conclusion": "健康总结",
    "recommendation": ["建议1", "建议2"]
  }}
}}

重要要求：
1. 只抽取报告中真实存在的信息，不存在的字段设置为 null 或空数组 []
2. 不要编造数据
3. 保持医学语义准确
4. 数值不修改，单位不修改
5. 只输出JSON，不输出任何其他解释文字！

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
            return self._ensure_result_structure(result)
        except Exception as e:
            logger.error(f"JSON解析错误: {e}", exc_info=True)
            return {"raw_response": content}

    def _ensure_result_structure(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """确保返回的结果包含所有必需的顶层字段"""
        base_structure = {
            "patient_info": None,
            "chief_complaint": None,
            "present_illness": None,
            "symptoms": [],
            "vital_signs": None,
            "medical_history": None,
            "physical_examination": [],
            "laboratory": [],
            "urine_test": None,
            "imaging": [],
            "ecg": None,
            "diagnosis": None,
            "clinical_features": [],
            "kg_query": None,
            "summary": None
        }
        
        # 合并用户结果到基础结构中
        for key, value in result.items():
            if key in base_structure or key == "ocr":
                base_structure[key] = value
        
        return base_structure

    def _get_empty_result(self, text: str) -> Dict[str, Any]:
        """返回空结果结构"""
        return {
            "patient_info": None,
            "chief_complaint": None,
            "present_illness": None,
            "symptoms": [],
            "vital_signs": None,
            "medical_history": None,
            "physical_examination": [],
            "laboratory": [],
            "urine_test": None,
            "imaging": [],
            "ecg": None,
            "diagnosis": None,
            "clinical_features": [],
            "kg_query": None,
            "summary": None,
            "ocr": {
                "source": "local",
                "file_type": None,
                "text_extracted": len(text) > 0,
                "text_length": len(text),
                "parse_confidence": None,
                "raw_text": text[:1000]
            }
        }

    def _mock_parse(self, text: str, user_question: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"开始本地规则解析，文本长度: {len(text)}")
        
        result = self._get_empty_result(text)
        result["ocr"]["source"] = "local_rule_parser"
        
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
                (r'(?:空腹)?(?:血糖|Glucose)[：:\s]*(\d+(?:\.\d+)?)\s*(?:mmol/L|mg/dL)?', "空腹血糖", "血糖", "mmol/L"),
                (r'(?:总胆固醇|Cholesterol|TC)[：:\s]*(\d+(?:\.\d+)?)\s*(?:mmol/L)?', "总胆固醇", "CHOL", "mmol/L"),
                (r'(?:甘油三酯|Triglyceride|TG)[：:\s]*(\d+(?:\.\d+)?)\s*(?:mmol/L)?', "甘油三酯", "TG", "mmol/L"),
                (r'(?:血红蛋白|Hb|HGB)[：:\s]*(\d{2,3})\s*(?:g/L)?', "血红蛋白", "HGB", "g/L"),
                (r'(?:白细胞|WBC)[：:\s]*(\d+(?:\.\d+)?)\s*(?:\*10\^9/L|×10⁹/L)?', "白细胞计数", "WBC", "×10⁹/L"),
                (r'(?:红细胞|RBC)[：:\s]*(\d+(?:\.\d+)?)\s*(?:\*10\^12/L|×10¹²/L)?', "红细胞计数", "RBC", "×10¹²/L"),
                (r'(?:血小板|PLT)[：:\s]*(\d{2,3})\s*(?:\*10\^9/L|×10⁹/L)?', "血小板计数", "PLT", "×10⁹/L")
            ]
            
            for pattern, name, item, unit in lab_patterns:
                match = re.search(pattern, clean_text, re.IGNORECASE)
                if match:
                    laboratory.append({
                        "item": item,
                        "name": name,
                        "value": match.group(1),
                        "raw_value": match.group(1),
                        "unit": unit,
                        "reference": None,
                        "flag": None,
                        "valid": True
                    })
            
            # 体检结论
            conclusion_match = re.search(r'(?:体检)?(?:结论|Summary|Conclusion)[：:\s]*(.+?)(?=\s{2,}|\Z)', clean_text, re.DOTALL)
            if conclusion_match:
                result["summary"] = {
                    "health_conclusion": conclusion_match.group(1).strip(),
                    "recommendation": []
                }
        
        # 更新结果
        if patient_info:
            result["patient_info"] = patient_info
        
        if vital_signs:
            result["vital_signs"] = vital_signs
        
        if laboratory:
            result["laboratory"] = laboratory
        
        logger.info(f"本地规则解析完成")
        return result
