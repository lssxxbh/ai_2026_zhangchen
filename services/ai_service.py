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
                max_tokens=2000
            )

            content = response.choices[0].message.content
            if content:
                result = self._parse_json_response(content)
                logger.info(f"LLM解析成功")
                return result

            return None
        except Exception as e:
            logger.error(f"LLM调用错误: {e}", exc_info=True)
            logger.info("降级到本地规则解析")
            return self._mock_parse(text, user_question)

    def _build_prompt(self, text: str, user_question: Optional[str] = None) -> str:
        prompt = f"""你是一名医学信息抽取专家。

请从以下体检报告文本中抽取信息并输出JSON。

要求：
1. 只抽取报告中真实存在的信息
2. 不存在的字段不要输出（动态JSON结构）
3. 不要编造数据
4. 保持医学语义准确
5. 数值不修改
6. 单位不修改
7. 如果有用户问题，请提取到 question 字段

重要：只输出JSON，不输出任何其他解释文字！

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
            result["_source"] = "llm"
            return result
        except Exception as e:
            logger.error(f"JSON解析错误: {e}", exc_info=True)
            return {"raw_response": content, "_source": "llm_raw"}

    def _mock_parse(self, text: str, user_question: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"开始本地规则解析，文本长度: {len(text)}")
        result = {}
        
        if user_question:
            result["question"] = user_question

        if text:
            info = {}
            
            logger.info(f"解析文本前200字符: {repr(text[:200])}")
            
            # 清理文本，去掉多余空格和换行
            clean_text = re.sub(r'\s+', ' ', text).strip()
            logger.info(f"清理后文本前200字符: {repr(clean_text[:200])}")
            
            # 姓名 - 更宽松的匹配
            name_match = re.search(r'(?:姓名|Name)[：:\s]*([^\s，。,]{2,8})', clean_text)
            if name_match:
                info["name"] = name_match.group(1).strip()
                logger.info(f"找到姓名: {info['name']}")
            
            # 性别
            gender_match = re.search(r'(?:性别|Gender)[：:\s]*([男女男女malefemale]{1,6})', clean_text, re.IGNORECASE)
            if gender_match:
                info["gender"] = gender_match.group(1).strip()
                logger.info(f"找到性别: {info['gender']}")
            
            # 年龄
            age_match = re.search(r'(?:年龄|Age)[：:\s]*(\d{1,3})', clean_text)
            if age_match:
                info["age"] = int(age_match.group(1))
                logger.info(f"找到年龄: {info['age']}")
            
            # 身高
            height_match = re.search(r'(?:身高|Height)[：:\s]*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?', clean_text, re.IGNORECASE)
            if height_match:
                info["height"] = {"value": float(height_match.group(1)), "unit": "cm"}
                logger.info(f"找到身高: {info['height']}")
            
            # 体重
            weight_match = re.search(r'(?:体重|Weight)[：:\s]*(\d+(?:\.\d+)?)\s*(?:kg|千克)?', clean_text, re.IGNORECASE)
            if weight_match:
                info["weight"] = {"value": float(weight_match.group(1)), "unit": "kg"}
                logger.info(f"找到体重: {info['weight']}")
            
            # 血压
            bp_match = re.search(r'(?:血压|Blood Pressure|BP)[：:\s]*(\d{2,3})[/-](\d{2,3})\s*(?:mmHg)?', clean_text, re.IGNORECASE)
            if bp_match:
                info["blood_pressure"] = {
                    "systolic": int(bp_match.group(1)),
                    "diastolic": int(bp_match.group(2)),
                    "unit": "mmHg"
                }
                logger.info(f"找到血压: {info['blood_pressure']}")
            
            # 心率
            hr_match = re.search(r'(?:心率|Heart Rate|HR)[：:\s]*(\d{2,3})\s*(?:次/分|bpm|beats)?', clean_text, re.IGNORECASE)
            if hr_match:
                info["heart_rate"] = {"value": int(hr_match.group(1)), "unit": "bpm"}
                logger.info(f"找到心率: {info['heart_rate']}")
            
            # 空腹血糖
            glucose_match = re.search(r'(?:空腹)?(?:血糖|Glucose)[：:\s]*(\d+(?:\.\d+)?)\s*(?:mmol/L|mg/dL)?', clean_text, re.IGNORECASE)
            if glucose_match:
                info["fasting_blood_glucose"] = {"value": float(glucose_match.group(1)), "unit": "mmol/L"}
                logger.info(f"找到血糖: {info['fasting_blood_glucose']}")
            
            # 总胆固醇
            chol_match = re.search(r'(?:总胆固醇|Cholesterol|TC)[：:\s]*(\d+(?:\.\d+)?)\s*(?:mmol/L)?', clean_text, re.IGNORECASE)
            if chol_match:
                info["total_cholesterol"] = {"value": float(chol_match.group(1)), "unit": "mmol/L"}
                logger.info(f"找到总胆固醇: {info['total_cholesterol']}")
            
            # 血红蛋白
            hb_match = re.search(r'(?:血红蛋白|Hb|HGB)[：:\s]*(\d{2,3})\s*(?:g/L)?', clean_text, re.IGNORECASE)
            if hb_match:
                info["hemoglobin"] = {"value": int(hb_match.group(1)), "unit": "g/L"}
                logger.info(f"找到血红蛋白: {info['hemoglobin']}")
            
            # 体检结论
            conclusion_match = re.search(r'(?:体检)?(?:结论|Summary|Conclusion)[：:\s]*(.+?)(?=\s{2,}|\Z)', clean_text, re.DOTALL)
            if conclusion_match:
                info["conclusion"] = conclusion_match.group(1).strip()
                logger.info(f"找到结论: {info['conclusion'][:80]}...")
            
            if info:
                result.update(info)
                logger.info(f"本地规则解析成功，提取到 {len(info)} 个字段")
            else:
                logger.warning("本地规则未提取到任何信息，返回文本预览")
                result["text_preview"] = text[:800]
                result["message"] = "使用本地规则解析，未找到结构化数据"
                result["raw_text"] = text[:500]
            
            result["_source"] = "local_rule_parser"
            result["_text_length"] = len(text)

        logger.info(f"最终解析结果: {result}")
        return result
