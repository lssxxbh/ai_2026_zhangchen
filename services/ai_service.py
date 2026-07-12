# services/ai_service.py
# -*- coding: utf-8 -*-

import json
import re
from typing import Optional, Dict, Any, List, Tuple

from openai import AsyncOpenAI

from config import settings
from utils.logger import logger


class AIService:
    def __init__(self):
        self.client = None

        # 兼容不同配置字段名：
        # 你的项目里日志显示 OPENAI_API_KEY 已配置，但代码里用的是 ALIYUN_API_KEY。
        # 这里两个都兼容，优先取 ALIYUN_API_KEY，没有再取 OPENAI_API_KEY。
        api_key = (
            getattr(settings, "ALIYUN_API_KEY", None)
            or getattr(settings, "OPENAI_API_KEY", None)
        )

        self.api_key = api_key

        logger.info("=" * 60)
        logger.info("LLM 配置信息:")
        logger.info(f"  Provider: {settings.LLM_PROVIDER}")
        logger.info(f"  API Base: {settings.LLM_API_BASE}")
        logger.info(f"  Model: {settings.LLM_MODEL}")
        logger.info(f"  API Key: {'已配置' if api_key else '未配置'}")
        logger.info("=" * 60)

        if api_key and "your_api_key_here" not in api_key:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=settings.LLM_API_BASE,
                timeout=float(getattr(settings, "LLM_TIMEOUT", 180)),
                max_retries=1,
            )
            logger.info(
                f"LLM 初始化成功: provider={settings.LLM_PROVIDER}, model={settings.LLM_MODEL}"
            )
        else:
            logger.warning("未配置有效 OPENAI_API_KEY / ALIYUN_API_KEY，将使用本地规则解析")

    async def parse_report(
        self,
        text: str,
        user_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        text = text or ""

        if not self.client:
            logger.warning("LLM 未配置，使用本地规则解析")
            return self._mock_parse(text, user_question)

        prompt = self._build_prompt(text, user_question)

        try:
            logger.info(f"开始调用 LLM: {settings.LLM_MODEL}")

            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一名专业的医学 OCR JSON 结构化专家。"
                        "你必须只输出严格 JSON 对象，不要输出 Markdown，不要输出 ```json，不要解释。"
                        "不要编造报告中不存在的信息。"
                        "如果某些字段无法识别，可以省略该字段。"
                        "如果完全无法识别，也必须返回一个合法 JSON 对象。"
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            # SiliconFlow API Reference 里 enable_thinking 是请求体字段。
            # OpenAI SDK 传第三方扩展字段时，应放在 extra_body 里。
            # response_format 用 JSON mode，要求模型返回合法 JSON。
            response = await self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=int(getattr(settings, "MAX_TOKENS", 3000)),
                response_format={"type": "json_object"},
                extra_body={
                    "enable_thinking": False
                },
                timeout=float(getattr(settings, "LLM_TIMEOUT", 180)),
            )

            content = self._extract_message_content(response)

            logger.info(f"LLM 原始输出长度: {len(content)}")
            logger.info(f"LLM 原始输出前500字符: {content[:500]}")

            parsed = self._parse_json_response(content)

            if not isinstance(parsed, dict):
                parsed = {"raw_response": content}

            parsed = self._clean_empty_fields(parsed)
            parsed = self._enhance_laboratory_result(parsed, text)
            parsed = self._standardize_ocr_json(parsed, text, source="llm")

            logger.info("LLM 解析成功")
            return parsed

        except Exception as e:
            logger.error(f"LLM 调用错误: {e}", exc_info=True)
            logger.info("降级到本地规则解析")
            return self._mock_parse(text, user_question)

    async def answer_followup_question(
        self,
        question: str,
        context_json: Dict[str, Any],
    ) -> str:
        """
        追问模式：基于上一轮报告 JSON 回答，不重新解析报告。
        """
        question = (question or "").strip()

        if not question:
            return "请先输入你想追问的问题。"

        if not self.client:
            return "当前未配置大模型，无法进行追问回答。请查看上一轮结构化报告结果。"

        compact_context = json.dumps(context_json or {}, ensure_ascii=False)[:12000]

        prompt = f"""
你是一名谨慎的医疗报告解读助手。请基于上一轮报告 JSON 回答用户追问。

要求：
1. 只能基于已有报告、影像、血常规、病历文本和结构化指标回答；
2. 不要编造不存在的数据；
3. 不要做确定诊断；
4. 如果用户问“看什么科 / 做什么检查 / 严重吗 / 下一步怎么办”，要结合风险等级、异常指标和影像证据回答；
5. 使用简体中文；
6. 不要输出 Markdown 标题符号；
7. 用清楚的分点说明；
8. 最后必须提醒“不能替代医生诊断”。

上一轮报告 JSON：
{compact_context}

用户追问：
{question}

请回答：
"""

        try:
            response = await self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "你是谨慎的医疗报告解读助手，不替代医生诊断。",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.1,
                max_tokens=1600,
                extra_body={
                    "enable_thinking": False
                },
                timeout=float(getattr(settings, "LLM_TIMEOUT", 180)),
            )

            answer = self._extract_message_content(response).strip()

            return answer or "没有生成有效回答。"

        except Exception as e:
            logger.error(f"追问回答失败: {e}", exc_info=True)
            return f"追问回答失败：{e}"

    def _extract_message_content(self, response: Any) -> str:
        """
        兼容 SiliconFlow / OpenAI-compatible 返回格式。

        SiliconFlow 示例：
        message: {
            "role": "assistant",
            "content": "...",
            "reasoning_content": "..."
        }

        某些推理模型如果 thinking 没关好，content 可能为空。
        这里优先取 content，content 为空时再取 reasoning_content。
        """
        try:
            message = response.choices[0].message
        except Exception as e:
            logger.error(f"无法读取 LLM response.choices[0].message: {e}", exc_info=True)
            return ""

        content = getattr(message, "content", None) or ""

        if not content:
            reasoning_content = getattr(message, "reasoning_content", None) or ""
            if reasoning_content:
                logger.warning("LLM content 为空，已使用 reasoning_content 兜底")
                content = reasoning_content

        return str(content or "").strip()

    def _build_prompt(self, text: str, user_question: Optional[str] = None) -> str:
        prompt = f"""
你是一名医学 OCR JSON 结构化专家。

请从以下体检报告、检验报告、病历文本中抽取信息，输出严格 JSON。

重要要求：
1. 只抽取文本中真实存在的信息；
2. 不要编造数据；
3. 数值、单位、参考范围必须保持原样；
4. 没有的字段不要输出；
5. 只输出 JSON，不要输出任何额外文字；
6. laboratory 中每个指标尽量包含 name、value、unit、reference、flag/status；
7. 输出必须是一个合法 JSON 对象，不能是数组，不能是空字符串。

推荐 JSON 结构：
{{
  "patient_info": {{
    "name": "姓名",
    "gender": "性别",
    "age": "年龄",
    "check_date": "检查日期",
    "hospital": "医院"
  }},
  "vital_signs": {{
    "height_cm": "身高",
    "weight_kg": "体重",
    "bmi": "BMI",
    "blood_pressure": "血压",
    "pulse": "脉搏"
  }},
  "laboratory": [
    {{
      "name": "检验项目名称",
      "value": "结果值",
      "unit": "单位",
      "reference": "参考范围",
      "flag": "H/L/N 或 偏高/偏低/正常"
    }}
  ],
  "summary": {{
    "health_conclusion": "报告原文中的总结或根据异常指标概括的简短说明",
    "recommendation": ["建议1", "建议2"]
  }}
}}

报告文本：
{text}
"""

        if user_question:
            prompt += f"\n用户补充问题：{user_question}\n"

        prompt += "\n请只输出 JSON 对象："
        return prompt

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        解析 LLM JSON 输出。

        兼容：
        1. 空字符串；
        2. ```json ... ```；
        3. 前后带解释文字；
        4. 模型返回数组；
        5. JSON 解析失败时返回 raw_response，避免流程中断。
        """
        value = str(content or "").strip()

        if not value:
            logger.error("JSON 解析失败: LLM 返回空字符串")
            return {
                "raw_response": "",
                "parse_error": "empty_llm_response",
            }

        # 去除 Markdown 代码块
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE).strip()
        value = re.sub(r"\s*```$", "", value).strip()

        # 先直接解析
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"items": parsed}
            return {"value": parsed}
        except Exception:
            pass

        # 截取第一个 { 到最后一个 }
        start_obj = value.find("{")
        end_obj = value.rfind("}")

        if start_obj >= 0 and end_obj >= start_obj:
            json_text = value[start_obj:end_obj + 1]
            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    return {"items": parsed}
                return {"value": parsed}
            except Exception as e:
                logger.error(
                    f"截取 JSON 对象后仍解析失败: {e}. JSON片段前500字符: {json_text[:500]}",
                    exc_info=True,
                )

        # 如果是数组形式，截取第一个 [ 到最后一个 ]
        start_arr = value.find("[")
        end_arr = value.rfind("]")

        if start_arr >= 0 and end_arr >= start_arr:
            json_text = value[start_arr:end_arr + 1]
            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, list):
                    return {"items": parsed}
                if isinstance(parsed, dict):
                    return parsed
                return {"value": parsed}
            except Exception as e:
                logger.error(
                    f"截取 JSON 数组后仍解析失败: {e}. JSON片段前500字符: {json_text[:500]}",
                    exc_info=True,
                )

        logger.error(f"JSON 解析失败，原始输出前1000字符: {value[:1000]}")

        return {
            "raw_response": content,
            "parse_error": "json_decode_failed",
        }

    def _clean_empty_fields(self, data: Any) -> Any:
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                cleaned_value = self._clean_empty_fields(value)

                if cleaned_value is None:
                    continue

                if isinstance(cleaned_value, (list, dict)) and len(cleaned_value) == 0:
                    continue

                cleaned[key] = cleaned_value

            return cleaned

        if isinstance(data, list):
            cleaned = []
            for item in data:
                cleaned_item = self._clean_empty_fields(item)

                if cleaned_item is None:
                    continue

                if isinstance(cleaned_item, (list, dict)) and len(cleaned_item) == 0:
                    continue

                cleaned.append(cleaned_item)

            return cleaned

        return data

    def _standardize_ocr_json(
        self,
        result: Dict[str, Any],
        text: str,
        source: str,
    ) -> Dict[str, Any]:
        if not isinstance(result, dict):
            result = {}

        result.setdefault("ocr", {})
        result["ocr"].update({
            "source": source,
            "text_extracted": bool(text),
            "text_length": len(text or ""),
            "raw_text": (text or "")[:2000],
        })

        laboratory = result.get("laboratory") or result.get("indicators") or result.get("metrics") or []

        if isinstance(laboratory, dict):
            labs = []
            for key, value in laboratory.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("name", key)
                    labs.append(item)
                else:
                    labs.append({
                        "name": key,
                        "value": value,
                    })
            laboratory = labs

        if not isinstance(laboratory, list):
            laboratory = []

        result["laboratory"] = [self._normalize_lab_item(item) for item in laboratory]

        if "summary" not in result:
            abnormal = sum(
                1
                for item in result["laboratory"]
                if str(item.get("flag") or item.get("status") or "").upper() in {"H", "L"}
                or any(k in str(item.get("flag") or item.get("status") or "") for k in ["高", "低", "异常", "升高", "降低"])
            )

            result["summary"] = {
                "health_conclusion": f"共识别到 {len(result['laboratory'])} 项检验指标，其中疑似异常 {abnormal} 项。",
                "recommendation": ["建议结合原始报告和医生意见复核异常指标。"],
            }

        return result

    def _normalize_lab_item(self, item: Any) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return {
                "name": str(item),
                "value": "",
                "unit": "",
                "reference": "",
                "flag": "",
                "status": "未标注",
            }

        flag = item.get("flag") or item.get("status") or item.get("state") or item.get("提示") or ""

        return {
            "name": (
                item.get("name")
                or item.get("item")
                or item.get("indicator")
                or item.get("项目")
                or item.get("指标名称")
                or "未知指标"
            ),
            "value": (
                item.get("value")
                or item.get("result")
                or item.get("检测值")
                or item.get("结果")
                or ""
            ),
            "unit": item.get("unit") or item.get("单位") or "",
            "reference": (
                item.get("reference")
                or item.get("range")
                or item.get("ref")
                or item.get("参考范围")
                or item.get("正常范围")
                or ""
            ),
            "flag": flag,
            "status": self._flag_to_status(flag),
        }

    @staticmethod
    def _flag_to_status(flag: Any) -> str:
        text = str(flag or "").strip()
        upper = text.upper()

        if upper == "H":
            return "升高"
        if upper == "L":
            return "降低"
        if upper == "N":
            return "正常"

        return text or "未标注"

    def _mock_parse(
        self,
        text: str,
        user_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info(f"开始本地规则解析，文本长度: {len(text or '')}")

        result: Dict[str, Any] = {
            "ocr": {
                "source": "local_rule_parser",
                "text_extracted": bool(text),
                "text_length": len(text or ""),
                "raw_text": (text or "")[:2000],
            }
        }

        patient_info = {}
        vital_signs = {}
        laboratory = []

        clean_text = re.sub(r"\s+", " ", text or "").strip()

        if clean_text:
            name_match = re.search(r"(?:姓名|Name)[：:\s]*([^\s，。,]{2,8})", clean_text)
            if name_match:
                patient_info["name"] = name_match.group(1).strip()

            gender_match = re.search(r"(?:性别|Gender)[：:\s]*([男女]|male|female)", clean_text, re.IGNORECASE)
            if gender_match:
                patient_info["gender"] = gender_match.group(1).strip()

            age_match = re.search(r"(?:年龄|Age)[：:\s]*(\d{1,3})", clean_text)
            if age_match:
                patient_info["age"] = age_match.group(1)

            date_match = re.search(r"(?:日期|Date|检查日期)[：:\s]*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})", clean_text)
            if date_match:
                patient_info["check_date"] = date_match.group(1)

            hospital_match = re.search(r"(?:医院|Hospital)[：:\s]*([^\s，。,]{2,30})", clean_text)
            if hospital_match:
                patient_info["hospital"] = hospital_match.group(1).strip()

            height_match = re.search(r"(?:身高|Height)[：:\s]*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?", clean_text, re.IGNORECASE)
            if height_match:
                vital_signs["height_cm"] = height_match.group(1)

            weight_match = re.search(r"(?:体重|Weight)[：:\s]*(\d+(?:\.\d+)?)\s*(?:kg|千克)?", clean_text, re.IGNORECASE)
            if weight_match:
                vital_signs["weight_kg"] = weight_match.group(1)

            bp_match = re.search(r"(?:血压|Blood Pressure|BP)[：:\s]*(\d{2,3})[/-](\d{2,3})\s*(?:mmHg)?", clean_text, re.IGNORECASE)
            if bp_match:
                vital_signs["blood_pressure"] = f"{bp_match.group(1)}/{bp_match.group(2)}"

            hr_match = re.search(r"(?:心率|脉搏|Heart Rate|HR|Pulse)[：:\s]*(\d{2,3})\s*(?:次/分|bpm|beats)?", clean_text, re.IGNORECASE)
            if hr_match:
                vital_signs["pulse"] = hr_match.group(1)

            laboratory.extend(self._parse_common_labs(clean_text))
            laboratory.extend(self._parse_blood_routine_table(text))

        if patient_info:
            result["patient_info"] = patient_info

        if vital_signs:
            result["vital_signs"] = vital_signs

        seen = set()
        dedup_labs = []

        for lab in laboratory:
            key = (lab.get("name"), lab.get("value"), lab.get("unit"))
            if key not in seen:
                seen.add(key)
                dedup_labs.append(lab)

        result["laboratory"] = dedup_labs

        result["summary"] = {
            "health_conclusion": f"本地规则共识别到 {len(dedup_labs)} 项检验指标。",
            "recommendation": ["建议结合完整报告和医生意见进一步判断。"],
        }

        logger.info("本地规则解析完成")
        return self._standardize_ocr_json(result, text, source="local_rule_parser")

    def _parse_common_labs(self, clean_text: str) -> List[Dict[str, str]]:
        lab_specs = [
            ("空腹血糖", ["空腹血糖", "血糖", "GLU", "Glucose"], "mmol/L", "3.9-6.1"),
            ("总胆固醇", ["总胆固醇", "TC", "CHOL", "Cholesterol"], "mmol/L", "0-5.2"),
            ("甘油三酯", ["甘油三酯", "TG", "Triglyceride"], "mmol/L", "0-1.7"),
            ("低密度脂蛋白胆固醇", ["LDL-C", "LDL", "低密度脂蛋白胆固醇", "低密度脂蛋白"], "mmol/L", "0-3.4"),
            ("高密度脂蛋白胆固醇", ["HDL-C", "HDL", "高密度脂蛋白胆固醇", "高密度脂蛋白"], "mmol/L", "1.0-2.1"),
            ("谷丙转氨酶", ["ALT", "谷丙转氨酶", "丙氨酸氨基转移酶"], "U/L", "7-40"),
            ("谷草转氨酶", ["AST", "谷草转氨酶", "天门冬氨酸氨基转移酶"], "U/L", "13-35"),
            ("γ-谷氨酰转肽酶", ["GGT", "γ-GT", "谷氨酰转肽酶"], "U/L", "7-45"),
            ("血红蛋白", ["HGB", "Hb", "血红蛋白"], "g/L", "115-150"),
            ("白细胞计数", ["WBC", "白细胞", "白细胞计数"], "10^9/L", "3.5-9.5"),
            ("红细胞计数", ["RBC", "红细胞", "红细胞计数"], "10^12/L", "3.8-5.8"),
            ("血小板计数", ["PLT", "血小板", "血小板计数"], "10^9/L", "125-350"),
        ]

        labs = []

        for name, aliases, default_unit, default_ref in lab_specs:
            for alias in aliases:
                pattern = rf"{re.escape(alias)}\s*[:：]?\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z0-9^*/.%μµ\u4e00-\u9fff]*)?"
                match = re.search(pattern, clean_text, re.IGNORECASE)
                if not match:
                    continue

                value = match.group(1)
                unit = match.group(2) or default_unit
                flag = self._calc_lab_flag(value, default_ref)

                labs.append({
                    "name": name,
                    "value": value,
                    "unit": unit,
                    "reference": default_ref,
                    "flag": flag,
                })
                break

        return labs

    def _enhance_laboratory_result(self, result: Optional[Dict[str, Any]], text: str) -> Dict[str, Any]:
        if not isinstance(result, dict):
            result = {}

        parsed_labs = self._parse_blood_routine_table(text)
        if not parsed_labs:
            return result

        existing = result.get("laboratory")
        if not isinstance(existing, list):
            result["laboratory"] = parsed_labs
            return result

        if len(parsed_labs) >= max(3, len(existing) // 2):
            result["laboratory"] = parsed_labs
            return result

        by_name = {
            item.get("name"): item
            for item in existing
            if isinstance(item, dict)
        }

        for lab in parsed_labs:
            by_name[lab["name"]] = lab

        result["laboratory"] = list(by_name.values())
        return result

    def _parse_blood_routine_table(self, text: str) -> List[Dict[str, str]]:
        if not text:
            return []

        specs = [
            ("白细胞数目", ["白细胞数目", "白细胞计数", "WBC"], "10^9/L", "4-10"),
            ("淋巴细胞数目", ["淋巴细胞数目", "LYMPH#", "#LYM"], "10^9/L", "0.8-4"),
            ("中性粒细胞数目", ["中性粒细胞数目", "NEUT#", "#GRAN"], "10^9/L", "2-7"),
            ("淋巴细胞百分比", ["淋巴细胞百分比", "LYMPH%", "%LYM"], "%", "20-40"),
            ("中性粒细胞百分比", ["中性粒细胞百分比", "NEUT%", "%GRAN"], "%", "50-70"),
            ("红细胞数目", ["红细胞数目", "红细胞计数", "RBC"], "10^12/L", "3.5-5.5"),
            ("血红蛋白", ["血红蛋白", "HGB", "Hb"], "g/L", "110-160"),
            ("红细胞压积", ["红细胞压积", "HCT"], "%", "37-54"),
            ("平均红细胞体积", ["平均红细胞体积", "MCV"], "fL", "80-100"),
            ("平均红细胞血红蛋白量", ["平均红细胞血红蛋白量", "MCH"], "pg", "27-34"),
            ("平均红细胞血红蛋白浓度", ["平均红细胞血红蛋白浓度", "MCHC"], "g/L", "320-360"),
            ("血小板", ["血小板", "PLT"], "10^9/L", "100-300"),
            ("平均血小板体积", ["平均血小板体积", "MPV"], "fL", "6.5-12"),
            ("血小板分布宽度", ["血小板分布宽度", "PDW"], "fL", "9-17"),
            ("血小板压积", ["血小板压积", "PCT"], "%", "0.108-0.282"),
            ("C反应蛋白", ["C反应蛋白", "CRP"], "mg/L", "0-10"),
        ]

        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        parsed: List[Dict[str, str]] = []
        used_names = set()

        for name, aliases, default_unit, default_ref in specs:
            row_index, row_text = self._find_lab_row(lines, aliases)
            if row_index < 0 or name in used_names:
                continue

            value, reference = self._extract_lab_value_and_reference(row_text, aliases)

            if not value:
                window = " ".join(lines[row_index: row_index + 4])
                value, reference = self._extract_lab_value_and_reference(window, aliases)

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

        return parsed if len(parsed) >= 2 else []

    @staticmethod
    def _find_lab_row(lines: List[str], aliases: List[str]) -> Tuple[int, str]:
        for index, line in enumerate(lines):
            compact = re.sub(r"\s+", "", line).upper()
            for alias in aliases:
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
        normalized = str(text or "").replace("—", "-").replace("－", "-").replace("~", "-")
        pattern = (
            r"(?<![A-Za-z])(?:"
            r"\d+(?:\.\d+)?\s*-+\s*\d+(?:\.\d+)?"
            r"|\d+(?:\.\d+)?"
            r")"
        )
        return re.findall(pattern, normalized)

    @staticmethod
    def _normalize_lab_number(value: str) -> str:
        value = re.sub(r"\s+", "", str(value or ""))
        trans = str.maketrans({
            "T": "7",
            "t": "7",
            "O": "0",
            "o": "0",
            "I": "1",
            "l": "1",
            "吕": "8",
            "G": "6",
            "g": "6",
            "B": "8",
        })
        return value.translate(trans)

    def _normalize_lab_reference(self, reference: str, default_ref: str) -> str:
        reference = self._normalize_lab_number(reference).replace("--", "-")
        reference = re.sub(r"-{2,}", "-", reference)
        reference = reference.strip("-")

        if not re.fullmatch(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?", reference):
            return default_ref

        try:
            low, high = [float(x) for x in reference.split("-", 1)]
            if low >= high:
                return default_ref
        except Exception:
            return default_ref

        return reference

    @staticmethod
    def _calc_lab_flag(value: str, reference: str) -> str:
        try:
            number = float(value)
            low_text, high_text = str(reference).split("-", 1)
            low = float(low_text)
            high = float(high_text)
        except Exception:
            return "N"

        if number < low:
            return "L"
        if number > high:
            return "H"
        return "N"
