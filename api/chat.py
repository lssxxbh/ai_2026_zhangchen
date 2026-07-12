# api/chat.py
# -*- coding: utf-8 -*-

import json
import re
import uuid
from pathlib import Path
from typing import Optional, Any, Dict, List
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from database import get_db
from models import User
from models.chat_message import MessageRole, ChatMessage
from schemas.report import (
    Visualization,
    Summary,
    Indicator,
    Agent,
    KnowledgeGraphItem,
    KnowledgeItem,
    Recommendation,
)
from api.deps import get_current_user
from services import (
    OCRService,
    PDFService,
    TextCleanService,
    AIService,
    ParserService,
    ConversationService,
)
from services.multimodal_service import (
    save_uploaded_images,
    analyze_images_sync,
    analyze_blood_sync,
    combine_reports_sync,
    parse_blood_indicators,
)
from services.advice_service import (
    generate_ai_advice,
    infer_risk_and_score,
    build_rule_based_departments_and_tests,
)
from utils import success_response, error_response, logger
from config import settings


router = APIRouter(prefix="/api", tags=["chat"])

ocr_service = OCRService()
pdf_service = PDFService(ocr_service)
text_clean_service = TextCleanService()
ai_service = AIService()
parser_service = ParserService(
    ocr_service,
    pdf_service,
    text_clean_service,
    ai_service,
)


# =========================
# 基础工具
# =========================

def get_file_type(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    suffix = Path(filename).suffix.lower().replace(".", "")
    if suffix in ["jpg", "jpeg", "png", "bmp", "webp"]:
        return suffix
    if suffix == "pdf":
        return "pdf"
    if suffix == "txt":
        return "txt"
    return suffix or "unknown"


def safe_filename(filename: str) -> str:
    filename = Path(filename).name
    filename = re.sub(r"[^\w.\-\u4e00-\u9fa5]", "_", filename)
    return filename or "uploaded_file"


def parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_json_form(value: str, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except Exception:
        return default


def build_conversation_title(text: str = "", file_name: Optional[str] = None) -> str:
    if file_name:
        return f"文件解析：{file_name[:80]}"
    clean_text = " ".join((text or "").strip().split())
    if clean_text:
        return clean_text[:80]
    return f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"


def build_user_message(
    text: str = "",
    file_name: Optional[str] = None,
    image_count: int = 0,
    has_blood: bool = False,
    quoted_evidence: bool = False,
) -> str:
    parts = []
    if text and text.strip():
        parts.append(text.strip())
    if file_name:
        parts.append(f"[附件: {file_name}]")
    if image_count:
        parts.append(f"[医学影像: {image_count} 张]")
    if has_blood:
        parts.append("[血常规文本: 已输入]")
    if quoted_evidence:
        parts.append("[引用上次多模态资料]")
    return "\n".join(parts) if parts else "空消息"


def strip_markdown_for_plain_text(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"```[\s\S]*?```", "", value)
    value = re.sub(r"^\s{0,3}#{1,6}\s*", "", value, flags=re.MULTILINE)
    value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
    value = re.sub(r"__(.*?)__", r"\1", value)
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"^\s*[-*+]\s+", "• ", value, flags=re.MULTILINE)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def clean_markdown_report_title(text: str) -> str:
    value = str(text or "")
    value = value.replace("####", "###")
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def is_followup_question(
    text: str,
    file: Optional[UploadFile],
    image_files: Optional[List[UploadFile]],
    blood_text: str,
    image_report: str = "",
    blood_report: str = "",
    combined_report: str = "",
) -> bool:
    if file and file.filename:
        return False
    valid_image_count = len([x for x in (image_files or []) if x and x.filename])
    if valid_image_count:
        return False
    if blood_text and blood_text.strip():
        return False
    if image_report or blood_report or combined_report:
        return False

    clean = (text or "").strip()
    if not clean:
        return False

    followup_keywords = [
        "为什么", "怎么办", "严重吗", "需要", "建议", "什么意思", "解释",
        "风险", "下一步", "看什么科", "做什么检查", "这个", "上述",
        "刚才", "报告里", "能不能", "要不要", "多久", "复查",
    ]
    return any(k in clean for k in followup_keywords) or len(clean) < 80


# =========================
# 标准化
# =========================

def normalize_indicator_item(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return {
            "name": item.get("name") or item.get("display_name") or item.get("indicator") or item.get("item") or item.get("metric") or item.get("指标名称") or item.get("项目") or "未知指标",
            "abbreviation": item.get("abbreviation") or item.get("abbr") or "",
            "value": item.get("value") or item.get("result") or item.get("detected_value") or item.get("检测值") or item.get("结果") or "--",
            "unit": item.get("unit") or "",
            "reference": item.get("reference") or item.get("range") or item.get("ref") or item.get("参考范围") or item.get("正常范围") or "--",
            "status": item.get("status") or item.get("flag") or item.get("state") or item.get("状态") or item.get("提示") or "未标注",
            "ref_low": item.get("ref_low"),
            "ref_high": item.get("ref_high"),
            "is_abnormal": item.get("is_abnormal"),
            "original_text": item.get("original_text") or "",
        }

    return {
        "name": str(item),
        "abbreviation": "",
        "value": "--",
        "unit": "",
        "reference": "--",
        "status": "未标注",
    }


def normalize_indicators(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    indicators = (
        result.get("indicators")
        or result.get("laboratory")
        or result.get("metrics")
        or result.get("items")
        or result.get("structured_data")
        or result.get("extracted_indicators")
        or []
    )

    if isinstance(indicators, dict):
        normalized = []
        for key, value in indicators.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("name", key)
                normalized.append(normalize_indicator_item(item))
            else:
                normalized.append({
                    "name": key,
                    "abbreviation": "",
                    "value": value,
                    "unit": "",
                    "reference": "--",
                    "status": "未标注",
                })
        return normalized

    if isinstance(indicators, list):
        return [normalize_indicator_item(item) for item in indicators]

    return []


def is_abnormal_status(status: Any) -> bool:
    text = str(status or "")
    keywords = ["高", "低", "异常", "临界", "阳性", "风险", "偏高", "偏低", "升高", "降低", "不正常", "H", "L"]
    return any(k in text for k in keywords)


def infer_risk_level(indicators: List[Dict[str, Any]]) -> str:
    abnormal_count = sum(1 for item in indicators if is_abnormal_status(item.get("status")))
    if abnormal_count >= 5:
        return "高风险"
    if abnormal_count >= 3:
        return "中高风险"
    if abnormal_count >= 1:
        return "中等风险"
    if indicators:
        return "低风险"
    return "未知"


def build_agents_from_indicators(indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text = " ".join([
        f"{item.get('name', '')} {item.get('abbreviation', '')} {item.get('status', '')}"
        for item in indicators
    ])

    agents = []

    if any(k in text for k in ["LDL", "HDL", "胆固醇", "血脂", "脂蛋白", "心", "血压"]):
        agents.append({
            "name": "心血管 Agent",
            "conclusion": "检测到血脂、血压或心血管相关指标，建议结合 BMI、吸烟史、家族史等因素综合评估心血管风险。",
            "confidence": 82,
            "risk_level": "需结合指标判断",
        })

    if any(k in text for k in ["ALT", "AST", "GGT", "转氨酶", "胆红素", "肝"]):
        agents.append({
            "name": "肝脏 Agent",
            "conclusion": "检测到肝功能相关指标，建议结合 AST、GGT、胆红素及肝胆超声进一步判断。",
            "confidence": 76,
            "risk_level": "需结合指标判断",
        })

    if any(k in text for k in ["血糖", "GLU", "糖化", "HbA1c", "胰岛", "糖代谢"]):
        agents.append({
            "name": "内分泌 Agent",
            "conclusion": "检测到糖代谢相关指标，建议关注空腹血糖、餐后血糖和糖化血红蛋白变化。",
            "confidence": 79,
            "risk_level": "需结合指标判断",
        })

    if any(k in text for k in ["血红蛋白", "HGB", "Hb", "白细胞", "红细胞", "血小板", "贫血", "WBC", "RBC", "PLT", "CRP", "PCT", "ESR"]):
        agents.append({
            "name": "血液 / 检验 Agent",
            "conclusion": "检测到血常规或炎症相关指标，建议结合白细胞、血红蛋白、血小板、CRP、PCT、ESR 等项目综合判断。",
            "confidence": 74,
            "risk_level": "需结合指标判断",
        })

    if any(k in text for k in ["结核", "空洞", "树芽征", "盗汗", "低热", "咯血"]):
        agents.append({
            "name": "呼吸 / 感染 Agent",
            "conclusion": "检测到呼吸系统感染或结核相关线索，建议结合影像原片、痰检、IGRA/PPD 等进一步评估。",
            "confidence": 84,
            "risk_level": "需尽快专科评估",
        })

    if not agents:
        agents.append({
            "name": "综合分析 Agent",
            "conclusion": "系统已完成体检报告综合解析，建议结合异常指标、既往病史和临床症状进一步复核。",
            "confidence": 70,
            "risk_level": "未标注",
        })

    return agents


def build_advice_from_indicators(indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    advice = []
    added_keys = set()

    for item in indicators:
        name = str(item.get("name", ""))
        abbr = str(item.get("abbreviation", ""))
        status = str(item.get("status", ""))
        text = f"{name} {abbr} {status}"

        if any(k in text for k in ["LDL", "HDL", "胆固醇", "血脂", "脂蛋白"]) and "lipid" not in added_keys:
            added_keys.add("lipid")
            advice.append({
                "title": "血脂相关复查建议",
                "content": f"{name} 当前状态为“{status}”。建议结合血压、BMI、吸烟史、家族史综合评估心血管风险。",
                "priority": "建议",
                "tags": [name, status],
            })

        elif any(k in text for k in ["ALT", "AST", "GGT", "转氨酶", "胆红素", "肝"]) and "liver" not in added_keys:
            added_keys.add("liver")
            advice.append({
                "title": "肝功能复查建议",
                "content": f"{name} 当前状态为“{status}”。建议结合 AST、GGT、胆红素、肝胆超声及用药饮酒史进一步判断。",
                "priority": "建议",
                "tags": [name, status],
            })

        elif any(k in text for k in ["血糖", "GLU", "糖化", "HbA1c", "糖代谢"]) and "glucose" not in added_keys:
            added_keys.add("glucose")
            advice.append({
                "title": "糖代谢复查建议",
                "content": f"{name} 当前状态为“{status}”。建议结合空腹血糖、餐后2小时血糖和糖化血红蛋白判断。",
                "priority": "建议",
                "tags": [name, status],
            })

        elif any(k in text for k in ["血红蛋白", "HGB", "Hb", "贫血"]) and "blood" not in added_keys:
            added_keys.add("blood")
            advice.append({
                "title": "贫血相关评估建议",
                "content": f"{name} 当前状态为“{status}”。建议结合红细胞、血红蛋白、铁蛋白、维生素B12、叶酸等进一步判断。",
                "priority": "建议",
                "tags": [name, status],
            })

        elif any(k in text for k in ["白细胞", "WBC", "中性粒", "CRP", "PCT", "ESR", "血沉", "感染", "炎症"]) and "infection" not in added_keys:
            added_keys.add("infection")
            advice.append({
                "title": "感染和炎症指标复查建议",
                "content": f"{name} 当前状态为“{status}”。建议结合体温、症状、CRP、PCT、ESR、影像表现和病原学检查综合判断。",
                "priority": "建议",
                "tags": [name, status],
            })

    return advice


def normalize_agents(result: Dict[str, Any], indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agents = (
        result.get("agents")
        or result.get("agent_outputs")
        or result.get("multi_agent")
        or result.get("agent_results")
        or result.get("specialist_analysis")
        or result.get("specialists")
        or []
    )

    if isinstance(agents, list) and agents:
        normalized = []
        for index, agent in enumerate(agents):
            if isinstance(agent, dict):
                try:
                    confidence = float(agent.get("confidence") or agent.get("score") or agent.get("probability") or 0)
                    if confidence <= 1:
                        confidence *= 100
                    confidence = int(confidence)
                except Exception:
                    confidence = 0

                normalized.append({
                    "name": agent.get("name") or agent.get("agent") or agent.get("role") or agent.get("department") or agent.get("agent_name") or f"Agent {index + 1}",
                    "conclusion": agent.get("conclusion") or agent.get("summary") or agent.get("result") or agent.get("message") or agent.get("suggestion") or "暂无结论。",
                    "confidence": confidence,
                    "risk_level": agent.get("risk_level") or result.get("risk_level") or "未标注",
                    "recommendation": agent.get("recommendation") or [],
                })
            else:
                normalized.append({
                    "name": f"Agent {index + 1}",
                    "conclusion": str(agent),
                    "confidence": 0,
                    "risk_level": result.get("risk_level") or "未标注",
                    "recommendation": [],
                })
        return normalized

    return build_agents_from_indicators(indicators)


def normalize_critique(result: Dict[str, Any]) -> List[str]:
    critique = result.get("critique") or result.get("critiques") or result.get("validation") or result.get("checks") or result.get("evidence_verification") or []
    if isinstance(critique, list):
        normalized = []
        for item in critique:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(item.get("message") or item.get("content") or item.get("result") or json.dumps(item, ensure_ascii=False))
            else:
                normalized.append(str(item))
        return normalized
    if isinstance(critique, str):
        return [critique]
    if isinstance(critique, dict):
        return [
            f"{key}：{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
            for key, value in critique.items()
        ]
    return [
        "结构校验：已返回标准 JSON，可用于前端可视化。",
        "证据校验：建议结合原始报告文本和医生意见复核异常指标。",
        "安全提示：系统结果仅用于健康管理与辅助初筛，不替代医生诊断。",
    ]


def normalize_advice(result: Dict[str, Any], indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    advice = result.get("advice") or result.get("recommendations") or result.get("health_advice") or result.get("suggestions") or []

    if isinstance(advice, dict):
        merged = []
        for key in ["diet", "exercise", "follow_up"]:
            for item in advice.get(key, []) or []:
                merged.append({
                    "title": {"diet": "饮食建议", "exercise": "运动建议", "follow_up": "复查建议"}.get(key, "建议"),
                    "content": str(item),
                    "priority": "建议",
                    "tags": [key],
                })
        if merged:
            return merged

    if isinstance(advice, list) and advice:
        normalized = []
        for index, item in enumerate(advice):
            if isinstance(item, str):
                normalized.append({
                    "title": f"建议 {index + 1}",
                    "content": item,
                    "priority": "建议",
                    "tags": [],
                })
            elif isinstance(item, dict):
                normalized.append({
                    "title": item.get("title") or item.get("name") or item.get("category") or f"建议 {index + 1}",
                    "content": item.get("content") or item.get("text") or item.get("message") or item.get("suggestion") or json.dumps(item, ensure_ascii=False),
                    "priority": item.get("priority") or "建议",
                    "tags": item.get("tags") or item.get("related") or [],
                })
            else:
                normalized.append({
                    "title": f"建议 {index + 1}",
                    "content": str(item),
                    "priority": "建议",
                    "tags": [],
                })
        return normalized

    return build_advice_from_indicators(indicators)


def normalize_knowledge_graph_items(items: Any) -> List[Dict[str, Any]]:
    if not items:
        return []
    if isinstance(items, dict):
        items = [items]
    if isinstance(items, str):
        return [{"disease": "知识图谱", "relation": "相关证据", "entity": items, "value": ""}]
    if not isinstance(items, list):
        return []

    normalized = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"disease": "知识图谱", "relation": "相关证据", "entity": item, "value": ""})
            continue
        if not isinstance(item, dict):
            continue
        normalized.append({
            "disease": str(item.get("disease") or item.get("head") or item.get("source") or item.get("subject") or item.get("from") or item.get("疾病") or item.get("头实体") or "N/A"),
            "relation": str(item.get("relation") or item.get("predicate") or item.get("type") or item.get("关系") or "N/A"),
            "entity": str(item.get("entity") or item.get("tail") or item.get("target") or item.get("object") or item.get("实体") or item.get("尾实体") or "N/A"),
            "value": str(item.get("value") or item.get("score") or item.get("weight") or item.get("confidence") or item.get("证据值") or ""),
        })
    return normalized


def normalize_knowledge_items(items: Any) -> List[Dict[str, Any]]:
    if not items:
        return []
    if isinstance(items, dict):
        items = [items]
    if isinstance(items, str):
        return [{"source": "医学依据", "content": items}]
    if not isinstance(items, list):
        return []

    normalized = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"source": "医学依据", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        normalized.append({
            "source": str(item.get("source") or item.get("title") or item.get("name") or item.get("来源") or "医学依据"),
            "content": str(item.get("content") or item.get("text") or item.get("summary") or item.get("evidence") or item.get("description") or item.get("内容") or json.dumps(item, ensure_ascii=False)),
        })
    return normalized


# =========================
# 前端增强结构
# =========================

def enhance_result_for_frontend(parsed_json: Dict[str, Any], extracted_text: str = "", user_text: str = "") -> Dict[str, Any]:
    if not isinstance(parsed_json, dict):
        parsed_json = {"raw_result": parsed_json}

    result = dict(parsed_json)

    indicators = normalize_indicators(result)
    result["indicators"] = indicators

    risk_level = result.get("risk_level") or result.get("risk") or result.get("overall_risk") or infer_risk_level(indicators)
    result["risk_level"] = risk_level

    if "summary" not in result:
        summary = result.get("diagnosis_summary") or result.get("report_summary") or result.get("final_answer") or result.get("answer") or result.get("message")
        if not summary:
            abnormal_count_value = sum(1 for item in indicators if is_abnormal_status(item.get("status")))
            if indicators:
                summary = f"系统共识别到 {len(indicators)} 项体检指标，其中疑似异常或需关注项目 {abnormal_count_value} 项。综合风险等级为：{risk_level}。"
            else:
                summary = "系统已完成报告解析，但未识别到标准指标列表，请查看 JSON 原始结果。"
        result["summary"] = summary

    result["agents"] = normalize_agents(result, indicators)
    result["critique"] = normalize_critique(result)
    result["advice"] = normalize_advice(result, indicators)

    if extracted_text and "extracted_text" not in result:
        result["extracted_text"] = extracted_text[:5000]

    if user_text and "user_text" not in result:
        result["user_text"] = user_text

    result.setdefault("_frontend_ready", True)
    return result


def build_assistant_message(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return "解析完成，但结果格式异常，请查看 JSON 原始结果。"

    if result.get("report_text"):
        return str(result["report_text"])

    summary = result.get("summary") or result.get("report_summary") or result.get("diagnosis_summary") or result.get("final_report") or result.get("conclusion") or "系统已完成报告解析。"
    risk_level = result.get("risk_level") or "未标注"
    health_score = result.get("health_score")

    indicators = result.get("indicators") or []
    agents = result.get("agents") or []
    advice = result.get("advice") or []
    critique = result.get("critique") or []
    multimodal = result.get("multimodal_evidence") or {}
    departments = result.get("departments") or []
    recommended_tests = result.get("recommended_tests") or []
    follow_up_plan = result.get("follow_up_plan") or []
    red_flags = result.get("red_flags") or []

    lines = []
    lines.append("📋 智能体检报告辅助诊断书")
    lines.append("")
    lines.append("一、综合结论")
    if isinstance(summary, dict):
        lines.append(str(summary.get("diagnosis") or summary.get("title") or json.dumps(summary, ensure_ascii=False)))
    else:
        lines.append(str(summary))
    lines.append("")

    lines.append("二、风险等级")
    lines.append(f"综合风险等级：{risk_level}")
    if health_score is not None:
        lines.append(f"健康评分：{health_score}")
    lines.append("")

    lines.append("三、关键体检指标")
    if isinstance(indicators, list) and indicators:
        for index, item in enumerate(indicators[:12], start=1):
            if isinstance(item, dict):
                lines.append(f"{index}. {item.get('name') or '未知指标'}：{item.get('value') or '--'} {item.get('unit') or ''}，参考范围：{item.get('reference') or '--'}，状态：{item.get('status') or '未标注'}")
            else:
                lines.append(f"{index}. {item}")
        if len(indicators) > 12:
            lines.append(f"其余 {len(indicators) - 12} 项指标请查看结构化 JSON 或报告可视化页面。")
    else:
        lines.append("暂未识别到标准化体检指标。")
    lines.append("")

    lines.append("四、多模态补充证据")
    if multimodal:
        if multimodal.get("image_report"):
            lines.append("1. 已纳入医学影像 AI 报告。")
        if multimodal.get("blood_report"):
            lines.append("2. 已纳入血常规 AI 报告。")
        if multimodal.get("combined_report"):
            lines.append("3. 已纳入影像与血常规初步综合报告。")
        if multimodal.get("blood_indicators"):
            lines.append(f"4. 已纳入血常规结构化指标 {len(multimodal.get('blood_indicators') or [])} 项。")
    else:
        lines.append("暂无影像或血常规补充证据。")
    lines.append("")

    lines.append("五、建议就诊科室")
    if departments:
        for index, item in enumerate(departments, start=1):
            if isinstance(item, dict):
                lines.append(f"{index}. {item.get('name', '相关专科')}：{item.get('reason', '')}")
            else:
                lines.append(f"{index}. {item}")
    else:
        lines.append("暂无明确科室建议。")
    lines.append("")

    lines.append("六、进一步检查建议")
    if recommended_tests:
        for index, item in enumerate(recommended_tests, start=1):
            if isinstance(item, dict):
                lines.append(f"{index}. [{item.get('priority', '建议')}] {item.get('name', '检查项目')}：{item.get('reason', '')}")
            else:
                lines.append(f"{index}. {item}")
    else:
        lines.append("暂无明确进一步检查建议。")
    lines.append("")

    lines.append("七、多 Agent 辅助分析")
    if isinstance(agents, list) and agents:
        for index, agent in enumerate(agents, start=1):
            if isinstance(agent, dict):
                name = agent.get("name") or agent.get("department") or f"Agent {index}"
                conclusion = agent.get("conclusion") or agent.get("summary") or "暂无结论。"
                confidence = agent.get("confidence")
                if confidence is not None and confidence != "":
                    lines.append(f"{index}. {name}：{conclusion} 置信度：{confidence}%")
                else:
                    lines.append(f"{index}. {name}：{conclusion}")
            else:
                lines.append(f"{index}. {agent}")
    else:
        lines.append("后端未返回独立 Agent 结果。")
    lines.append("")

    lines.append("八、健康管理建议")
    if isinstance(advice, list) and advice:
        for index, item in enumerate(advice, start=1):
            if isinstance(item, dict):
                title = item.get("title") or f"建议 {index}"
                content = item.get("content") or item.get("text") or item.get("message") or item.get("suggestion") or ""
                priority = item.get("priority") or "建议"
                lines.append(f"{index}. [{priority}] {title}：{content}")
            else:
                lines.append(f"{index}. {item}")
    else:
        lines.append("暂无明确健康建议。")
    lines.append("")

    lines.append("九、复查计划")
    if follow_up_plan:
        for index, item in enumerate(follow_up_plan, start=1):
            if isinstance(item, dict):
                lines.append(f"{index}. {item.get('time', '随访')}：{item.get('action', '')}")
            else:
                lines.append(f"{index}. {item}")
    else:
        lines.append("暂无明确复查计划。")
    lines.append("")

    lines.append("十、需要及时就医的信号")
    if red_flags:
        for index, item in enumerate(red_flags, start=1):
            lines.append(f"{index}. {item}")
    else:
        lines.append("暂无特别危险信号提示。")
    lines.append("")

    lines.append("十一、校验说明")
    if isinstance(critique, list) and critique:
        for index, item in enumerate(critique[:5], start=1):
            lines.append(f"{index}. {item}")
    else:
        lines.append("当前结果仅作为健康管理与辅助初筛参考，不替代医生诊断。")
    lines.append("")
    lines.append("⚠️ 免责声明：本报告由 AI 根据体检文本、OCR 结果及可选多模态证据自动生成，仅供健康管理参考，不能替代临床医生诊断。")

    return "\n".join(lines)


# =========================
# 文件解析 / 多模态分析
# =========================

async def run_multimodal_analysis(
    image_files: Optional[List[UploadFile]],
    blood_text: str,
    extra_note: str,
    analysis_mode: str,
    use_compression: bool,
) -> Dict[str, Any]:
    image_report = ""
    blood_report = ""
    combined_report = ""
    blood_indicators = []
    image_result = None

    valid_files = [f for f in (image_files or []) if f and f.filename]

    if valid_files:
        try:
            image_paths = await save_uploaded_images(valid_files)
            image_result = await run_in_threadpool(
                analyze_images_sync,
                image_paths,
                extra_note,
                analysis_mode,
                use_compression,
            )
            image_report = image_result.get("image_report", "") if isinstance(image_result, dict) else ""
        except Exception as e:
            logger.error(f"统一 chat 中影像分析失败: {e}", exc_info=True)
            image_report = f"影像分析失败：{e}"

    if blood_text and blood_text.strip():
        try:
            blood_indicators = parse_blood_indicators(blood_text)
            blood_report = await run_in_threadpool(analyze_blood_sync, blood_text)
        except Exception as e:
            logger.error(f"统一 chat 中血常规分析失败: {e}", exc_info=True)
            blood_report = f"血常规分析失败：{e}"
            blood_indicators = parse_blood_indicators(blood_text)

    if image_report and blood_report:
        try:
            combined_report = await run_in_threadpool(
                combine_reports_sync,
                image_report,
                blood_report,
            )
        except Exception as e:
            logger.error(f"统一 chat 中影像血常规综合失败: {e}", exc_info=True)
            combined_report = f"影像与血常规初步综合失败：{e}"

    return {
        "image_report": image_report or "",
        "blood_report": blood_report or "",
        "combined_report": combined_report or "",
        "blood_indicators": blood_indicators or [],
        "image_result": image_result or {},
    }


def merge_multimodal_evidence(final_json: Dict[str, Any], multimodal_payload: Dict[str, Any]) -> Dict[str, Any]:
    image_report = multimodal_payload.get("image_report") or ""
    blood_report = multimodal_payload.get("blood_report") or ""
    combined_report = multimodal_payload.get("combined_report") or ""
    blood_indicator_list = multimodal_payload.get("blood_indicators") or []
    image_result = multimodal_payload.get("image_result") or {}

    has_multimodal = bool(image_report or blood_report or combined_report or blood_indicator_list or image_result)
    if not has_multimodal:
        return final_json

    clean_payload = {
        "image_report": image_report,
        "blood_report": blood_report,
        "combined_report": combined_report,
        "blood_indicators": blood_indicator_list,
        "image_result": image_result,
    }

    final_json["multimodal_evidence"] = clean_payload
    final_json["image_report"] = image_report
    final_json["blood_report"] = blood_report
    final_json["combined_report"] = combined_report
    final_json["blood_indicators"] = blood_indicator_list

    old_indicators = final_json.get("indicators") or []
    if not isinstance(old_indicators, list):
        old_indicators = []

    converted_blood_indicators = []
    if isinstance(blood_indicator_list, list):
        for item in blood_indicator_list:
            if isinstance(item, dict):
                converted_blood_indicators.append(normalize_indicator_item(item))

    merged = []
    seen = set()

    for item in old_indicators + converted_blood_indicators:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("name", "")), str(item.get("abbreviation", "")), str(item.get("value", "")))
        if key not in seen:
            seen.add(key)
            merged.append(item)

    final_json["indicators"] = merged
    final_json["risk_level"] = final_json.get("risk_level") or infer_risk_level(merged)
    final_json["agents"] = normalize_agents(final_json, merged)
    final_json["advice"] = normalize_advice(final_json, merged)

    return final_json


# =========================
# 外部 B 项目 MedRAG 调用
# =========================

async def call_report_api(
    ocr_json: Dict[str, Any],
    raw_text: str = "",
    user_text: str = "",
    multimodal_evidence: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        if not settings.REPORT_GENERATION_API_URL:
            logger.warning("REPORT_GENERATION_API_URL 未配置，跳过报告接口调用")
            return None

        payload = {
            "ocr_json": ocr_json or {},
            "raw_text": raw_text or "",
            "user_text": user_text or "",
            "multimodal_evidence": multimodal_evidence or {},
            "source": "AI_2026_ZHANGCHEN",
        }

        logger.info(f"开始调用 B 项目报告接口: {settings.REPORT_GENERATION_API_URL}")

        timeout = httpx.Timeout(
            connect=10.0,
            read=600.0,
            write=60.0,
            pool=10.0,
        )

        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(
                settings.REPORT_GENERATION_API_URL,
                json=payload,
            )

            logger.info(f"B 项目报告接口 wrapped 响应状态码: {response.status_code}")

            if response.status_code == 422:
                logger.warning("B 项目报告接口 wrapped 模式 422，尝试 direct ocr_json 模式")
                response = await client.post(
                    settings.REPORT_GENERATION_API_URL,
                    json=ocr_json,
                )
                logger.info(f"B 项目报告接口 direct 响应状态码: {response.status_code}")

        if response.status_code >= 400:
            logger.error(f"B 项目报告接口调用失败: {response.status_code}, {response.text}")
            return {
                "success": False,
                "error": f"REPORT_API_HTTP_{response.status_code}",
                "detail": response.text,
            }

        data = response.json()
        logger.info("B 项目报告接口调用成功")
        return data

    except httpx.ReadTimeout as e:
        logger.error("调用 B 项目报告接口读取超时：B 工程已连接，但长时间未返回结果", exc_info=True)
        return {
            "success": False,
            "error": "B_REPORT_API_READ_TIMEOUT",
            "detail": str(e),
        }

    except httpx.ConnectTimeout as e:
        logger.error("连接 B 项目报告接口超时：请确认 B 工程是否启动、端口是否正确", exc_info=True)
        return {
            "success": False,
            "error": "B_REPORT_API_CONNECT_TIMEOUT",
            "detail": str(e),
        }

    except httpx.ConnectError as e:
        logger.error("连接 B 项目报告接口失败：请确认 http://127.0.0.1:8000 是否可访问", exc_info=True)
        return {
            "success": False,
            "error": "B_REPORT_API_CONNECT_ERROR",
            "detail": str(e),
        }

    except Exception as e:
        logger.error(f"调用 B 项目报告接口异常: {e}", exc_info=True)
        return {
            "success": False,
            "error": type(e).__name__,
            "detail": str(e),
        }


def merge_report_api_result(final_json: Dict[str, Any], report_result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(report_result, dict):
        return final_json

    final_json["report_api_result"] = report_result

    data = report_result.get("data") if isinstance(report_result.get("data"), dict) else report_result
    if not isinstance(data, dict):
        data = {}

    api_report_text = (
        report_result.get("report_text")
        or data.get("report_text")
        or data.get("report")
        or data.get("markdown")
        or data.get("final_report")
        or data.get("final_answer")
    )

    if api_report_text:
        final_json["report_text"] = api_report_text

    api_visualization = (
        report_result.get("visualization")
        or data.get("visualization")
        or data.get("visualization_from_api")
    )

    if api_visualization and isinstance(api_visualization, dict):
        final_json["visualization_from_api"] = api_visualization

        vis_summary = api_visualization.get("summary")
        if vis_summary:
            final_json["summary"] = vis_summary
            if isinstance(vis_summary, dict):
                if "risk_level" in vis_summary:
                    final_json["risk_level"] = vis_summary["risk_level"]
                if "health_score" in vis_summary:
                    final_json["health_score"] = vis_summary["health_score"]

        vis_indicators = api_visualization.get("indicators")
        if vis_indicators:
            final_json["indicators"] = normalize_indicators({"indicators": vis_indicators})

        vis_agents = (
            api_visualization.get("agents")
            or api_visualization.get("agent_outputs")
            or api_visualization.get("multi_agent")
            or api_visualization.get("agent_results")
            or api_visualization.get("specialist_analysis")
            or api_visualization.get("specialists")
        )
        if vis_agents:
            final_json["agents"] = normalize_agents({"agents": vis_agents}, final_json.get("indicators", []))

        vis_advice = (
            api_visualization.get("recommendation")
            or api_visualization.get("advice")
            or api_visualization.get("recommendations")
            or api_visualization.get("health_advice")
        )
        if vis_advice:
            final_json["advice"] = normalize_advice({"advice": vis_advice}, final_json.get("indicators", []))

        vis_critique = (
            api_visualization.get("critique")
            or api_visualization.get("validation")
            or api_visualization.get("checks")
            or api_visualization.get("evidence_verification")
        )
        if vis_critique:
            final_json["critique"] = normalize_critique({"critique": vis_critique})

        vis_kg = api_visualization.get("knowledge_graph") or api_visualization.get("kg_evidence") or api_visualization.get("knowledgeGraph")
        if vis_kg:
            final_json["knowledge_graph"] = normalize_knowledge_graph_items(vis_kg)

        vis_knowledge = (
            api_visualization.get("knowledge")
            or api_visualization.get("medical_evidence")
            or api_visualization.get("medicalEvidence")
            or api_visualization.get("references")
            or api_visualization.get("evidence")
        )
        if vis_knowledge:
            final_json["knowledge"] = normalize_knowledge_items(vis_knowledge)

        for key in ["departments", "recommended_tests", "follow_up_plan", "red_flags"]:
            value = api_visualization.get(key)
            if value:
                final_json[key] = value

    report_summary = data.get("summary") or data.get("report_summary") or data.get("diagnosis_summary") or data.get("conclusion") or data.get("final_report") or data.get("report")
    if report_summary:
        final_json["summary"] = report_summary
        final_json["report_summary"] = report_summary

    report_risk = data.get("risk_level") or data.get("risk") or data.get("overall_risk") or data.get("risk_assessment")
    if report_risk:
        final_json["risk_level"] = report_risk

    report_indicators = data.get("indicators") or data.get("metrics") or data.get("items") or data.get("structured_data") or data.get("extracted_indicators")
    if report_indicators:
        final_json["indicators"] = normalize_indicators({"indicators": report_indicators})

    report_kg = (
        data.get("knowledge_graph")
        or data.get("kg_evidence")
        or data.get("knowledgeGraph")
        or report_result.get("knowledge_graph")
        or report_result.get("kg_evidence")
        or report_result.get("knowledgeGraph")
    )
    if report_kg:
        final_json["knowledge_graph"] = normalize_knowledge_graph_items(report_kg)

    report_knowledge = (
        data.get("knowledge")
        or data.get("medical_evidence")
        or data.get("medicalEvidence")
        or data.get("references")
        or data.get("evidence")
        or report_result.get("knowledge")
        or report_result.get("medical_evidence")
        or report_result.get("medicalEvidence")
        or report_result.get("references")
        or report_result.get("evidence")
    )
    if report_knowledge:
        final_json["knowledge"] = normalize_knowledge_items(report_knowledge)

    report_agents = (
        data.get("agents")
        or data.get("agent_outputs")
        or data.get("multi_agent")
        or data.get("agent_results")
        or data.get("specialist_analysis")
        or data.get("specialists")
        or report_result.get("agents")
        or report_result.get("agent_outputs")
        or report_result.get("multi_agent")
        or report_result.get("agent_results")
        or report_result.get("specialist_analysis")
        or report_result.get("specialists")
    )
    if report_agents:
        final_json["agents"] = normalize_agents({"agents": report_agents}, final_json.get("indicators", []))

    report_advice = (
        data.get("advice")
        or data.get("recommendations")
        or data.get("suggestions")
        or data.get("health_advice")
        or report_result.get("advice")
        or report_result.get("recommendations")
        or report_result.get("suggestions")
        or report_result.get("health_advice")
    )
    if report_advice:
        final_json["advice"] = normalize_advice({"advice": report_advice}, final_json.get("indicators", []))

    report_critique = (
        data.get("critique")
        or data.get("validation")
        or data.get("checks")
        or data.get("evidence_verification")
        or report_result.get("critique")
        or report_result.get("validation")
        or report_result.get("checks")
        or report_result.get("evidence_verification")
    )
    if report_critique:
        final_json["critique"] = normalize_critique({"critique": report_critique})

    for key in ["departments", "recommended_tests", "follow_up_plan", "red_flags"]:
        value = data.get(key) or report_result.get(key)
        if value:
            final_json[key] = value

    final_json["_report_api_merged"] = True
    return final_json


# =========================
# 最终清洗 / 风险 / 建议
# =========================

def remove_noisy_medrag_fields(final_json: Dict[str, Any]) -> Dict[str, Any]:
    noisy_keys = ["similar_cases", "similarCases", "case_retrieval", "retrieved_cases", "rag_cases"]
    for key in noisy_keys:
        final_json.pop(key, None)

    vis = final_json.get("visualization_from_api")
    if isinstance(vis, dict):
        vis["similar_cases"] = []
        final_json["visualization_from_api"] = vis

    report_api = final_json.get("report_api_result")
    if isinstance(report_api, dict):
        data = report_api.get("data")
        if isinstance(data, dict):
            visualization = data.get("visualization")
            if isinstance(visualization, dict):
                visualization["similar_cases"] = []

    return final_json


def enrich_final_medical_result(final_json: Dict[str, Any]) -> Dict[str, Any]:
    final_json = remove_noisy_medrag_fields(final_json)

    risk = infer_risk_and_score(final_json)
    final_json["risk_level"] = risk["risk_level"]
    final_json["health_score"] = risk["health_score"]
    final_json["risk_debug"] = {
        "abnormal_count": risk["abnormal_count"],
        "danger_hit": risk["danger_hit"],
        "tb_hit": risk.get("tb_hit", 0),
    }

    rule_extra = build_rule_based_departments_and_tests(final_json)
    advice_payload = generate_ai_advice(final_json)

    if advice_payload.get("advice"):
        final_json["advice"] = advice_payload["advice"]
    elif not final_json.get("advice"):
        final_json["advice"] = build_advice_from_indicators(final_json.get("indicators", []))

    if not final_json.get("departments"):
        final_json["departments"] = advice_payload.get("departments") or rule_extra["departments"]
    if not final_json.get("recommended_tests"):
        final_json["recommended_tests"] = advice_payload.get("recommended_tests") or rule_extra["recommended_tests"]
    if not final_json.get("follow_up_plan"):
        final_json["follow_up_plan"] = advice_payload.get("follow_up_plan") or rule_extra["follow_up_plan"]
    if not final_json.get("red_flags"):
        final_json["red_flags"] = advice_payload.get("red_flags") or rule_extra["red_flags"]

    vis = final_json.get("visualization_from_api")
    if isinstance(vis, dict):
        summary = vis.get("summary") if isinstance(vis.get("summary"), dict) else {}
        summary["risk_level"] = final_json["risk_level"]
        summary["health_score"] = final_json["health_score"]

        if not summary.get("title"):
            summary["title"] = "医疗检测报告"

        if not summary.get("diagnosis") or summary.get("diagnosis") in ["体检指标异常，需进一步评估", "N/A", "暂无综合诊断"]:
            raw_summary = final_json.get("summary")
            if isinstance(raw_summary, dict):
                summary["diagnosis"] = raw_summary.get("diagnosis") or raw_summary.get("health_conclusion") or raw_summary.get("summary") or "系统已完成综合分析，建议结合异常指标和专科检查进一步评估。"
            else:
                summary["diagnosis"] = str(raw_summary or "系统已完成综合分析，建议结合异常指标和专科检查进一步评估。")

        summary["reason"] = f"系统识别到 {risk['abnormal_count']} 项异常/需关注指标，并结合影像、血常规、病历文本中的危险信号进行综合分级。"

        vis["summary"] = summary
        vis["similar_cases"] = []
        vis["departments"] = final_json.get("departments") or []
        vis["recommended_tests"] = final_json.get("recommended_tests") or []
        vis["follow_up_plan"] = final_json.get("follow_up_plan") or []
        vis["red_flags"] = final_json.get("red_flags") or []

        if final_json.get("knowledge_graph"):
            vis["knowledge_graph"] = final_json.get("knowledge_graph")
        if final_json.get("knowledge"):
            vis["knowledge"] = final_json.get("knowledge")
        if final_json.get("agents"):
            vis["agents"] = final_json.get("agents")

        vis["recommendation"] = {
            "diet": [
                x.get("content", "")
                for x in final_json.get("advice", [])
                if isinstance(x, dict) and x.get("content")
            ],
            "exercise": [],
            "follow_up": [
                f"{x.get('time', '')}：{x.get('action', '')}"
                for x in final_json.get("follow_up_plan", [])
                if isinstance(x, dict)
            ] + [
                f"{x.get('name', '')}：{x.get('reason', '')}"
                for x in final_json.get("recommended_tests", [])
                if isinstance(x, dict)
            ],
        }

        final_json["visualization_from_api"] = vis

    return final_json


# =========================
# 可视化构建
# =========================

def build_visualization(final_json: Dict[str, Any]) -> Optional[Visualization]:
    try:
        summary_data = final_json.get("summary", {})
        risk_level = final_json.get("risk_level") or "待评估"

        if not isinstance(summary_data, dict):
            summary_data = {
                "diagnosis": str(summary_data),
                "title": "医疗检测报告",
                "risk_level": risk_level,
                "confidence": 0.0,
                "reason": "根据结构化指标和多模态证据生成。",
                "health_score": final_json.get("health_score", 0),
            }

        health_score = summary_data.get("health_score") or final_json.get("health_score") or 0

        if not health_score:
            if risk_level == "高风险":
                health_score = 45
            elif risk_level == "中高风险":
                health_score = 58
            elif risk_level == "中等风险":
                health_score = 70
            elif risk_level == "低-中风险":
                health_score = 82
            elif risk_level == "低风险":
                health_score = 90
            else:
                health_score = 0

        diagnosis = summary_data.get("diagnosis") or summary_data.get("health_conclusion") or summary_data.get("summary") or "系统已完成报告解析。"

        summary = Summary(
            title=summary_data.get("title", "医疗检测报告"),
            diagnosis=str(diagnosis),
            risk_level=summary_data.get("risk_level", risk_level),
            confidence=float(summary_data.get("confidence", 0.0)),
            reason=summary_data.get("reason", "根据报告文本、OCR 结构化结果及多模态证据生成。"),
            health_score=int(health_score),
        )

        indicators = []
        for item in final_json.get("indicators", []):
            if isinstance(item, dict):
                raw_value = item.get("value", 0)
                try:
                    value = float(raw_value)
                except Exception:
                    value = 0.0

                indicators.append(
                    Indicator(
                        name=item.get("name", "N/A"),
                        abbreviation=item.get("abbreviation", ""),
                        value=value,
                        unit=item.get("unit", ""),
                        reference=item.get("reference", "N/A"),
                        status=item.get("status", "N/A"),
                    )
                )

        agents = []
        for item in final_json.get("agents", []):
            if isinstance(item, dict):
                try:
                    conf = float(item.get("confidence", 0))
                    if conf > 1:
                        conf = conf / 100
                except Exception:
                    conf = 0.0

                rec = item.get("recommendation") or []
                if isinstance(rec, str):
                    rec = [rec]

                agents.append(
                    Agent(
                        department=item.get("name", item.get("department", "N/A")),
                        risk_level=item.get("risk_level", final_json.get("risk_level", "N/A")),
                        confidence=conf,
                        summary=item.get("conclusion", item.get("summary", "N/A")),
                        recommendation=rec,
                    )
                )

        similar_cases = []

        knowledge_graph = []
        for item in final_json.get("knowledge_graph", []):
            if isinstance(item, dict):
                knowledge_graph.append(
                    KnowledgeGraphItem(
                        disease=str(item.get("disease", item.get("head", "N/A"))),
                        relation=str(item.get("relation", "N/A")),
                        entity=str(item.get("entity", item.get("tail", "N/A"))),
                        value=str(item.get("value", item.get("score", "N/A"))),
                    )
                )

        knowledge = []
        for item in final_json.get("knowledge", []):
            if isinstance(item, dict):
                knowledge.append(
                    KnowledgeItem(
                        source=item.get("source", "N/A"),
                        content=item.get("content", item.get("text", "N/A")),
                    )
                )

        if not knowledge:
            for item in final_json.get("critique", []) or []:
                knowledge.append(KnowledgeItem(source="Critique 校验", content=str(item)))

        advice = final_json.get("advice") or []
        departments = final_json.get("departments") or []
        recommended_tests = final_json.get("recommended_tests") or []
        follow_up_plan = final_json.get("follow_up_plan") or []

        diet_list = []
        exercise_list = []
        follow_up_list = []

        for item in advice:
            if isinstance(item, dict):
                text = item.get("content") or item.get("title") or ""
            else:
                text = str(item)
            if text:
                diet_list.append(text)

        for dep in departments:
            if isinstance(dep, dict):
                follow_up_list.append(f"建议就诊科室：{dep.get('name', '')}。原因：{dep.get('reason', '')}")

        for test in recommended_tests:
            if isinstance(test, dict):
                follow_up_list.append(f"建议检查：{test.get('name', '')}。原因：{test.get('reason', '')}")

        for plan in follow_up_plan:
            if isinstance(plan, dict):
                follow_up_list.append(f"{plan.get('time', '')}：{plan.get('action', '')}")

        recommendation = Recommendation(
            diet=diet_list,
            exercise=exercise_list,
            follow_up=follow_up_list,
        )

        warning = final_json.get("warning", "本报告仅供健康管理参考，不替代医生诊断。")

        return Visualization(
            summary=summary,
            indicators=indicators,
            agents=agents,
            similar_cases=similar_cases,
            knowledge_graph=knowledge_graph,
            knowledge=knowledge,
            recommendation=recommendation,
            warning=warning,
            departments=final_json.get("departments") or [],
            recommended_tests=final_json.get("recommended_tests") or [],
            follow_up_plan=final_json.get("follow_up_plan") or [],
            red_flags=final_json.get("red_flags") or [],
        )

    except Exception as e:
        logger.error(f"Error building visualization: {e}", exc_info=True)
        return None


# =========================
# 主接口：统一上传识别
# =========================

@router.post("/chat")
async def chat(
    text: str = Form(""),
    conversation_id: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None),

    image_files: Optional[List[UploadFile]] = File(None),
    blood_text: str = Form(""),
    extra_note: str = Form(""),
    analysis_mode: str = Form("fast"),
    use_compression: str = Form("true"),

    image_report: str = Form(""),
    blood_report: str = Form(""),
    combined_report: str = Form(""),
    blood_indicators: str = Form("[]"),

    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        conv_service = ConversationService(db)

        file_path = None
        file_name = None
        file_type = None

        if file and file.filename:
            logger.info(f"收到主报告文件: filename={file.filename}")

            file_name = file.filename
            file_type = get_file_type(file_name)

            settings.UPLOAD_DIR.mkdir(exist_ok=True)

            save_name = f"{datetime.now().strftime('%Y%m%d%H%M%S_%f')}_{safe_filename(file_name)}"
            file_path = settings.UPLOAD_DIR / save_name

            content = await file.read()
            file_path.write_bytes(content)

            logger.info(f"主报告文件已保存: {file_path}")

        if conversation_id:
            conv = await conv_service.get_conversation(conversation_id, current_user.id)
            if not conv:
                return error_response(msg="Conversation not found")
        else:
            title = build_conversation_title(text=text, file_name=file_name)
            conv = await conv_service.create_conversation(current_user.id, title)

        valid_image_count = len([x for x in (image_files or []) if x and x.filename])
        quoted_evidence = bool(image_report or blood_report or combined_report)

        user_msg_text = build_user_message(
            text=text,
            file_name=file_name,
            image_count=valid_image_count,
            has_blood=bool(blood_text and blood_text.strip()),
            quoted_evidence=quoted_evidence,
        )

        await conv_service.add_message(
            conv.id,
            MessageRole.USER,
            user_msg_text,
            file_name=file_name,
            file_type=file_type,
        )

        if conversation_id and is_followup_question(
            text=text,
            file=file,
            image_files=image_files,
            blood_text=blood_text,
            image_report=image_report,
            blood_report=blood_report,
            combined_report=combined_report,
        ):
            result = await db.execute(
                select(ChatMessage)
                .where(
                    ChatMessage.conversation_id == conv.id,
                    ChatMessage.role == MessageRole.ASSISTANT,
                    ChatMessage.json_result.isnot(None),
                )
                .order_by(ChatMessage.created_at.desc())
            )
            last_assistant = result.scalars().first()

            if last_assistant and last_assistant.json_result:
                try:
                    last_json = json.loads(last_assistant.json_result)
                except Exception:
                    last_json = {}

                answer = await ai_service.answer_followup_question(
                    question=text,
                    context_json=last_json,
                )

                await conv_service.add_message(
                    conv.id,
                    MessageRole.ASSISTANT,
                    answer,
                    json_result=json.dumps(last_json, ensure_ascii=False),
                    report_id=last_assistant.report_id,
                    visualization_json=last_assistant.visualization_json,
                )

                return success_response(
                    data={
                        "conversation_id": conv.id,
                        "message": answer,
                        "report_id": last_assistant.report_id,
                        "has_report": bool(last_assistant.report_id),
                        "result": last_json,
                        "followup": True,
                    }
                )

        parsed_json = None
        extracted_text = ""

        if file_path:
            # 关键修复：不再先 extract_text_from_file，避免图片 OCR 重复跑两遍。
            parsed_json = await parser_service.parse_file(
                file_path=file_path,
                file_type=file_type or "unknown",
                user_text=text,
            )

            if isinstance(parsed_json, dict):
                ocr_info = parsed_json.get("ocr") or {}
                extracted_text = (
                    ocr_info.get("raw_text")
                    or parsed_json.get("extracted_text")
                    or parsed_json.get("raw_text")
                    or ""
                )

            if parsed_json and "ocr" in parsed_json and "source" in parsed_json["ocr"]:
                logger.info(f"当前使用的解析器来源: {parsed_json['ocr']['source']}")

        elif text and text.strip():
            extracted_text = text
            parsed_json = await parser_service.parse_text(text)

            if parsed_json and "ocr" in parsed_json and "source" in parsed_json["ocr"]:
                logger.info(f"当前使用的解析器来源: {parsed_json['ocr']['source']}")

        else:
            extracted_text = blood_text or extra_note or ""
            parsed_json = {
                "ocr": {
                    "source": "multimodal_only",
                    "text_extracted": bool(extracted_text),
                    "text_length": len(extracted_text),
                    "raw_text": extracted_text[:1000],
                },
                "message": "本次未提供主报告文本或文件，仅基于可选多模态证据进行辅助分析。",
                "laboratory": [],
            }

        if not parsed_json:
            parsed_json = {
                "ocr": {
                    "source": "empty_fallback",
                    "text_extracted": False,
                    "text_length": len(extracted_text or ""),
                    "raw_text": (extracted_text or "")[:1000],
                },
                "message": "处理完成，但未解析出结构化结果。",
                "extracted_text": extracted_text[:1500] if extracted_text else "",
                "indicators": [],
                "risk_level": "未知",
                "advice": [{
                    "title": "重新上传或补充信息",
                    "content": "请确认文件内容清晰，或直接输入体检指标文本后再次解析。",
                    "priority": "建议",
                    "tags": ["解析失败"],
                }],
            }

        final_json = enhance_result_for_frontend(
            parsed_json=parsed_json,
            extracted_text=extracted_text,
            user_text=text,
        )

        final_json["extracted_raw"] = parsed_json

        multimodal_payload = await run_multimodal_analysis(
            image_files=image_files,
            blood_text=blood_text,
            extra_note=extra_note,
            analysis_mode=analysis_mode or "fast",
            use_compression=parse_bool(use_compression, True),
        )

        if image_report and not multimodal_payload.get("image_report"):
            multimodal_payload["image_report"] = image_report
        if blood_report and not multimodal_payload.get("blood_report"):
            multimodal_payload["blood_report"] = blood_report
        if combined_report and not multimodal_payload.get("combined_report"):
            multimodal_payload["combined_report"] = combined_report

        cached_blood_indicators = parse_json_form(blood_indicators, [])
        if cached_blood_indicators and not multimodal_payload.get("blood_indicators"):
            multimodal_payload["blood_indicators"] = cached_blood_indicators

        final_json = merge_multimodal_evidence(
            final_json=final_json,
            multimodal_payload=multimodal_payload,
        )

        logger.info(
            f"调用 B 项目报告 API 前 final_json 部分: "
            f"{json.dumps(final_json, ensure_ascii=False)[:800]}..."
        )

        report_result = await call_report_api(
            ocr_json=final_json,
            raw_text=extracted_text or text or blood_text or "",
            user_text=text,
            multimodal_evidence=final_json.get("multimodal_evidence") or {},
        )

        logger.info(
            f"B 项目报告 API 返回部分: "
            f"{json.dumps(report_result, ensure_ascii=False)[:800] if report_result else 'None'}..."
        )

        if report_result is not None:
            final_json = merge_report_api_result(
                final_json=final_json,
                report_result=report_result,
            )

        final_json = enrich_final_medical_result(final_json)

        visualization_obj = build_visualization(final_json)

        report_id = None
        has_report = False

        if visualization_obj:
            report_id = str(uuid.uuid4())
            has_report = True

        raw_assistant_text = final_json.get("report_text") or build_assistant_message(final_json)

        if final_json.get("report_text"):
            final_json["raw_report_text"] = clean_markdown_report_title(final_json.get("report_text"))
            assistant_msg_content = strip_markdown_for_plain_text(final_json.get("report_text"))
        else:
            assistant_msg_content = strip_markdown_for_plain_text(raw_assistant_text)

        json_str = json.dumps(final_json, ensure_ascii=False)

        await conv_service.add_message(
            conv.id,
            MessageRole.ASSISTANT,
            assistant_msg_content,
            json_result=json_str,
            file_name=file_name,
            file_type=file_type,
            report_id=report_id,
            visualization_json=json.dumps(
                visualization_obj.model_dump(),
                ensure_ascii=False,
            ) if visualization_obj else None,
        )

        response_data = {
            "conversation_id": conv.id,
            "message": assistant_msg_content,
            "report_id": report_id,
            "has_report": has_report,
            "result": final_json,
        }

        return success_response(data=response_data)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return error_response(msg=str(e))


# =========================
# 报告查询接口
# =========================

@router.get("/report/{report_id}")
async def get_report(
    report_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        result = await db.execute(
            select(ChatMessage).where(ChatMessage.report_id == report_id)
        )
        chat_message = result.scalars().first()

        if not chat_message:
            raise HTTPException(status_code=404, detail="Report not found")

        visualization_data = None
        if chat_message.visualization_json:
            visualization_data = json.loads(chat_message.visualization_json)

        raw_json = None
        if chat_message.json_result:
            try:
                raw_json = json.loads(chat_message.json_result)
            except Exception:
                raw_json = None

        report_text = chat_message.message
        if isinstance(raw_json, dict):
            report_text = raw_json.get("raw_report_text") or raw_json.get("report_text") or chat_message.message

        return success_response(
            data={
                "status": "done",
                "report_text": report_text,
                "visualization": visualization_data,
                "raw_json": raw_json,
            }
        )

    except HTTPException as http_e:
        raise http_e

    except Exception as e:
        logger.error(f"Error fetching report {report_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/chat/{chat_id}/reports")
async def get_chat_reports(
    chat_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        result = await db.execute(
            select(ChatMessage).where(
                ChatMessage.conversation_id == chat_id,
                ChatMessage.role == MessageRole.ASSISTANT,
                ChatMessage.report_id.isnot(None),
            ).order_by(ChatMessage.created_at)
        )
        chat_messages = result.scalars().all()

        reports = []
        for msg in chat_messages:
            title = "医疗检测报告"
            if msg.visualization_json:
                try:
                    vis_data = json.loads(msg.visualization_json)
                    if vis_data and "summary" in vis_data and "title" in vis_data["summary"]:
                        title = vis_data["summary"]["title"]
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode visualization_json for message {msg.id}")

            reports.append({
                "report_id": msg.report_id,
                "title": title,
                "created_at": msg.created_at.isoformat(),
            })

        return success_response(data=reports)

    except Exception as e:
        logger.error(f"Error fetching reports for chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
