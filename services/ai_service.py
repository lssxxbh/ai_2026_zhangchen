import json
import re
from typing import Optional, Dict, Any, List, Tuple
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
                result = self._enhance_laboratory_result(result, text)
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
            logger.debug(f"尝试解析LLM原始输出: {content[:1000]}...") # 记录前1000字符
            json_str = content
            json_start = content.find("{")
            json_end = content.rfind("}")
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end + 1]
            
            # 如果截取后发现不是有效的JSON，尝试修复
            if not json_str.strip().startswith("{") or not json_str.strip().endswith("}"):
                logger.warning("LLM输出可能包含非JSON前缀或后缀，尝试直接解析完整内容")
                json_str = content # 恢复完整内容尝试解析

            result = json.loads(json_str)
            # 移除所有值为 null 或空数组的字段
            result = self._clean_empty_fields(result)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}. 原始LLM输出: {content}", exc_info=True)
            return {"raw_response": content}
        except Exception as e:
            logger.error(f"JSON解析发生未知错误: {e}. 原始LLM输出: {content}", exc_info=True)
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
        return self._enhance_laboratory_result(result, text)

    def _enhance_laboratory_result(self, result: Optional[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """用确定性表格解析修正血常规这类高频检验单，降低 LLM/OCR 串列错误。"""
        if not isinstance(result, dict):
            result = {}

        parsed_labs = self._parse_blood_routine_table(text)
        if not parsed_labs:
            return result

        existing = result.get("laboratory")
        if not isinstance(existing, list) or len(parsed_labs) >= max(3, len(existing) // 2):
            result["laboratory"] = parsed_labs
            return result

        by_name = {item.get("name"): item for item in existing if isinstance(item, dict)}
        for lab in parsed_labs:
            by_name[lab["name"]] = lab
        result["laboratory"] = list(by_name.values())
        return result

    def _parse_blood_routine_table(self, text: str) -> List[Dict[str, str]]:
        if not text:
            return []

        specs = [
            ("白细胞数目", ["白细胞数目", "白细胞计数", "WBC"], "10*9/L", "4-10"),
            ("淋巴细胞数目", ["淋巴细胞数目", "#LYM"], "10*9/L", "0.8-4"),
            ("中间细胞数目", ["中间细胞数目", "#MON"], "10*9/L", "0.1-1.2"),
            ("中性细胞数目", ["中性细胞数目", "中性粒细胞数目", "#GRAN"], "10*9/L", "2-7"),
            ("淋巴细胞百分比", ["淋巴细胞百分比", "%LYM"], "%", "20-40"),
            ("中间细胞百分比", ["中间细胞百分比", "%MON"], "%", "3-14"),
            ("中性粒细胞百分比", ["中性粒细胞百分比", "%GRAN"], "%", "50-70"),
            ("红细胞数目", ["红细胞数目", "红细胞计数", "RBC"], "10*12/L", "3.5-5.5"),
            ("血红蛋白", ["血红蛋白", "HGB"], "g/L", "110-160"),
            ("红细胞压积", ["红细胞压积", "HCT"], "%", "37-54"),
            ("平均红细胞体积", ["平均红细胞体积", "MCV"], "fL", "80-100"),
            ("平均红细胞血红蛋白量", ["平均红细胞血红蛋白量", "平均红细胞血红蛋白里", "MCH"], "pg", "27-34"),
            ("平均红细胞血红蛋白浓度", ["平均红细胞血红蛋白浓度", "MCHC"], "g/L", "320-360"),
            ("红细胞分布宽度变异系数", ["红细胞分布宽度变异系数", "红细胞分布宽度娈异系数", "RDW-CV", "RDW-CT"], "%", "11-16"),
            ("红细胞分布宽度标准差", ["红细胞分布宽度标准差", "RDW-SV", "RDN-ST"], "fL", "35-56"),
            ("血小板", ["血小板", "PLT"], "10*9/L", "100-300"),
            ("平均血小板体积", ["平均血小板体积", "MPV"], "fL", "6.5-12"),
            ("血小板分布宽度", ["血小板分布宽度", "PDW", "PDV"], "fL", "9-17"),
            ("血小板压积", ["血小板压积", "PCT"], "%", "0.108-0.282"),
        ]

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        parsed: List[Dict[str, str]] = []
        used_names = set()

        for name, aliases, default_unit, default_ref in specs:
            row_index, row_text = self._find_lab_row(lines, aliases)
            if row_index < 0 or name in used_names:
                continue

            value, reference = self._extract_lab_value_and_reference(row_text, aliases)
            if not value:
                window = " ".join(lines[row_index + 1: row_index + 5])
                value, reference = self._extract_lab_value_and_reference(window, aliases)
                if not value:
                    tokens = self._number_like_tokens(window)
                    value = tokens[0] if tokens else ""
                    reference = tokens[1] if len(tokens) > 1 else ""
            if not value:
                continue

            value = self._normalize_lab_number(value)
            reference = self._normalize_lab_reference(reference or default_ref, default_ref)
            flag = self._calc_lab_flag(value, reference)
            parsed.append({
                "name": name,
                "value": value,
                "unit": default_unit,
                "reference": reference,
                "flag": flag,
            })
            used_names.add(name)

        return parsed if len(parsed) >= 3 else []

    @staticmethod
    def _find_lab_row(lines: List[str], aliases: List[str]) -> Tuple[int, str]:
        chinese_aliases = [alias for alias in aliases if re.search(r"[\u4e00-\u9fff]", alias)]
        ordered_alias_groups = [chinese_aliases, aliases]
        seen = set()
        for alias_group in ordered_alias_groups:
            for index, line in enumerate(lines):
                compact = re.sub(r"\s+", "", line).upper()
                for alias in alias_group:
                    key = (index, alias)
                    if key in seen:
                        continue
                    seen.add(key)
                    if alias.upper().replace(" ", "") in compact:
                        return index, line
        return -1, ""

    def _extract_lab_value_and_reference(self, text: str, aliases: List[str]) -> Tuple[str, str]:
        for alias in aliases:
            pattern = re.escape(alias)
            match = re.search(pattern + r"(?P<trailing>.*)", text, re.IGNORECASE)
            if match:
                tokens = self._number_like_tokens(match.group("trailing"))
                if tokens:
                    value = tokens[0]
                    reference = tokens[1] if len(tokens) > 1 else ""
                    return value, reference
        return "", ""

    @staticmethod
    def _number_like_tokens(text: str) -> List[str]:
        normalized = text.replace("—", "-").replace("－", "-").replace("~", "-")
        pattern = (
            r"(?<![A-Za-z])(?:"
            r"\d+(?:\.\d+)?\s*-+\s*\d+(?:\.\d+)?"
            r"|[\dTtOoIl吕GgB]+(?:\.\s*[\dTtOoIl吕GgB]+)?"
            r")"
        )
        return re.findall(pattern, normalized)

    @staticmethod
    def _normalize_lab_number(value: str) -> str:
        value = re.sub(r"\s+", "", value)
        trans = str.maketrans({
            "T": "7", "t": "7", "O": "0", "o": "0", "I": "1", "l": "1",
            "吕": "8", "G": "6", "g": "6", "B": "8",
        })
        value = value.translate(trans)
        return value

    def _normalize_lab_reference(self, reference: str, default_ref: str) -> str:
        reference = self._normalize_lab_number(reference).replace("--", "-")
        reference = re.sub(r"-{2,}", "-", reference)
        reference = reference.strip("-")
        if not re.fullmatch(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?", reference):
            return default_ref
        low, high = [float(x) for x in reference.split("-", 1)]
        if low >= high:
            return default_ref

        try:
            default_low, default_high = [float(x) for x in default_ref.split("-", 1)]
        except Exception:
            return reference

        if high < default_high * 0.55 or high > default_high * 2.2:
            return default_ref
        if default_low >= 1 and (low < default_low * 0.45 or low > default_low * 2.2):
            return default_ref
        return reference

    @staticmethod
    def _calc_lab_flag(value: str, reference: str) -> str:
        try:
            number = float(value)
            low_text, high_text = reference.split("-", 1)
            low = float(low_text)
            high = float(high_text)
        except Exception:
            return "N"
        if number < low:
            return "L"
        if number > high:
            return "H"
        return "N"
