import json
import re
import uuid
from pathlib import Path
from typing import Optional, Any, Dict, List
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from database import get_db
from models import User
from models.chat_message import MessageRole, ChatMessage
from schemas.chat import ChatResponse
from schemas.report import (
    Visualization,
    Summary,
    Indicator,
    Agent,
    SimilarCase,
    KnowledgeGraphItem,
    KnowledgeItem,
    Recommendation
)
from api.deps import get_current_user
from services import (
    OCRService,
    PDFService,
    TextCleanService,
    AIService,
    ParserService,
    ConversationService
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
    ai_service
)


def get_file_type(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None

    suffix = Path(filename).suffix.lower().replace(".", "")

    if suffix in ["jpg", "jpeg", "png", "bmp"]:
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


def build_conversation_title(text: str = "", file_name: Optional[str] = None) -> str:
    if file_name:
        return f"文件解析：{file_name[:80]}"

    clean_text = " ".join((text or "").strip().split())
    if clean_text:
        return clean_text[:80]

    return f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"


def build_user_message(text: str = "", file_name: Optional[str] = None) -> str:
    parts = []

    if text and text.strip():
        parts.append(text.strip())

    if file_name:
        parts.append(f"[附件: {file_name}]")

    return "\n".join(parts) if parts else "空消息"


def normalize_indicator_item(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return {
            "name": (
                item.get("name")
                or item.get("indicator")
                or item.get("item")
                or item.get("metric")
                or item.get("指标名称")
                or item.get("项目")
                or "未知指标"
            ),
            "value": (
                item.get("value")
                or item.get("result")
                or item.get("detected_value")
                or item.get("检测值")
                or item.get("结果")
                or "--"
            ),
            "reference": (
                item.get("reference")
                or item.get("range")
                or item.get("ref")
                or item.get("参考范围")
                or item.get("正常范围")
                or "--"
            ),
            "status": (
                item.get("status")
                or item.get("flag")
                or item.get("state")
                or item.get("状态")
                or item.get("提示")
                or "未标注"
            )
        }

    return {
        "name": str(item),
        "value": "--",
        "reference": "--",
        "status": "未标注"
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
                normalized.append(
                    {
                        "name": key,
                        "value": value,
                        "reference": "--",
                        "status": "未标注"
                    }
                )
        return normalized

    if isinstance(indicators, list):
        return [normalize_indicator_item(item) for item in indicators]

    return []


def is_abnormal_status(status: Any) -> bool:
    text = str(status or "")
    keywords = [
        "高", "低", "异常", "临界", "阳性", "风险",
        "偏高", "偏低", "升高", "降低", "不正常"
    ]
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
    text = " ".join(
        [
            f"{item.get('name', '')} {item.get('status', '')}"
            for item in indicators
        ]
    )

    agents = []

    if any(k in text for k in ["LDL", "HDL", "胆固醇", "血脂", "脂蛋白", "心", "血压"]):
        agents.append(
            {
                "name": "心血管 Agent",
                "conclusion": "检测到血脂、血压或心血管相关指标，建议结合 BMI、吸烟史、家族史等因素综合评估心血管风险。",
                "confidence": 82
            }
        )

    if any(k in text for k in ["ALT", "AST", "GGT", "转氨酶", "胆红素", "肝"]):
        agents.append(
            {
                "name": "肝脏 Agent",
                "conclusion": "检测到肝功能相关指标，建议结合 AST、GGT、胆红素及肝胆超声进一步判断。",
                "confidence": 76
            }
        )

    if any(k in text for k in ["血糖", "GLU", "糖化", "HbA1c", "胰岛", "糖代谢"]):
        agents.append(
            {
                "name": "内分泌 Agent",
                "conclusion": "检测到糖代谢相关指标，建议关注空腹血糖、餐后血糖和糖化血红蛋白变化。",
                "confidence": 79
            }
        )

    if any(k in text for k in ["血红蛋白", "HGB", "Hb", "白细胞", "红细胞", "血小板", "贫血"]):
        agents.append(
            {
                "name": "血液 Agent",
                "conclusion": "检测到血常规相关指标，建议结合血红蛋白、白细胞、血小板等项目综合判断。",
                "confidence": 74
            }
        )

    if not agents:
        agents.append(
            {
                "name": "综合分析 Agent",
                "conclusion": "系统已完成体检报告综合解析，建议结合异常指标、既往病史和临床症状进一步复核。",
                "confidence": 70
            }
        )

    return agents


def build_advice_from_indicators(indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    advice = []
    added_keys = set()

    for item in indicators:
        name = str(item.get("name", ""))
        status = str(item.get("status", ""))
        text = f"{name} {status}"

        if any(k in text for k in ["LDL", "HDL", "胆固醇", "血脂", "脂蛋白"]) and "lipid" not in added_keys:
            added_keys.add("lipid")
            advice.append(
                {
                    "title": "血脂管理建议",
                    "content": f"{name} 当前状态为“{status}”。建议关注饮食脂肪结构、规律运动、体重控制，并结合血压、BMI、吸烟史和家族史评估心血管风险。",
                    "tags": [name, status]
                }
            )

        elif any(k in text for k in ["ALT", "AST", "GGT", "转氨酶", "胆红素", "肝"]) and "liver" not in added_keys:
            added_keys.add("liver")
            advice.append(
                {
                    "title": "肝功能复查建议",
                    "content": f"{name} 当前状态为“{status}”。建议近期避免饮酒、熬夜和自行用药，并结合 AST、GGT、胆红素及肝胆超声进一步判断。",
                    "tags": [name, status]
                }
            )

        elif any(k in text for k in ["血糖", "GLU", "糖化", "HbA1c", "糖代谢"]) and "glucose" not in added_keys:
            added_keys.add("glucose")
            advice.append(
                {
                    "title": "糖代谢干预建议",
                    "content": f"{name} 当前状态为“{status}”。建议关注精制碳水摄入、餐后血糖波动，并结合糖化血红蛋白进行周期性复查。",
                    "tags": [name, status]
                }
            )

        elif any(k in text for k in ["血红蛋白", "HGB", "Hb", "贫血"]) and "blood" not in added_keys:
            added_keys.add("blood")
            advice.append(
                {
                    "title": "血常规复查建议",
                    "content": f"{name} 当前状态为“{status}”。建议结合红细胞、血红蛋白、铁蛋白、维生素 B12 等指标进一步判断。",
                    "tags": [name, status]
                }
            )

    if not advice:
        advice.append(
            {
                "title": "综合健康建议",
                "content": "请结合体检指标、既往病史和临床症状综合判断。若存在持续异常或明显不适，应及时前往医疗机构就诊。",
                "tags": ["综合建议"]
            }
        )

    return advice


def normalize_agents(result: Dict[str, Any], indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agents = (
        result.get("agents")
        or result.get("agent_outputs")
        or result.get("multi_agent")
        or result.get("agent_results")
        or []
    )

    if isinstance(agents, list) and agents:
        normalized = []
        for index, agent in enumerate(agents):
            if isinstance(agent, dict):
                try:
                    confidence = int(float(agent.get("confidence") or agent.get("score") or 0))
                except Exception:
                    confidence = 0

                normalized.append(
                    {
                        "name": (
                            agent.get("name")
                            or agent.get("agent")
                            or agent.get("role")
                            or f"Agent {index + 1}"
                        ),
                        "conclusion": (
                            agent.get("conclusion")
                            or agent.get("summary")
                            or agent.get("result")
                            or agent.get("message")
                            or "暂无结论。"
                        ),
                        "confidence": confidence
                    }
                )
            else:
                normalized.append(
                    {
                        "name": f"Agent {index + 1}",
                        "conclusion": str(agent),
                        "confidence": 0
                    }
                )
        return normalized

    return build_agents_from_indicators(indicators)


def normalize_critique(result: Dict[str, Any]) -> List[str]:
    critique = (
        result.get("critique")
        or result.get("critiques")
        or result.get("validation")
        or result.get("checks")
        or []
    )

    if isinstance(critique, list):
        normalized = []
        for item in critique:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(
                    item.get("message")
                    or item.get("content")
                    or item.get("result")
                    or json.dumps(item, ensure_ascii=False)
                )
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
        "安全提示：系统结果仅用于健康管理与辅助初筛，不替代医生诊断。"
    ]


def normalize_advice(result: Dict[str, Any], indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    advice = (
        result.get("advice")
        or result.get("recommendations")
        or result.get("health_advice")
        or result.get("suggestions")
        or []
    )

    if isinstance(advice, list) and advice:
        normalized = []
        for index, item in enumerate(advice):
            if isinstance(item, str):
                normalized.append(
                    {
                        "title": f"建议 {index + 1}",
                        "content": item,
                        "tags": []
                    }
                )
            elif isinstance(item, dict):
                normalized.append(
                    {
                        "title": (
                            item.get("title")
                            or item.get("name")
                            or item.get("category")
                            or f"建议 {index + 1}"
                        ),
                        "content": (
                            item.get("content")
                            or item.get("text")
                            or item.get("message")
                            or item.get("suggestion")
                            or json.dumps(item, ensure_ascii=False)
                        ),
                        "tags": item.get("tags") or item.get("related") or []
                    }
                )
            else:
                normalized.append(
                    {
                        "title": f"建议 {index + 1}",
                        "content": str(item),
                        "tags": []
                    }
                )
        return normalized

    return build_advice_from_indicators(indicators)


def build_visualization(final_json: Dict[str, Any]) -> Optional[Visualization]:
    try:
        # Extract summary
        summary_data = final_json.get("summary", {})
        
        risk_level = final_json.get("risk_level")
        
        # Ensure summary_data is a dictionary
        if not isinstance(summary_data, dict):
            # If summary is a string, treat it as the diagnosis text and provide defaults for other fields
            summary_data = {"diagnosis": summary_data, "title": "医疗检测报告", "risk_level": risk_level or "N/A", "confidence": 0.0, "reason": "N/A", "health_score": 0}
            
        health_score = summary_data.get("health_score") or 0
        if not health_score:
            if risk_level == "高风险":
                health_score = 60
            elif risk_level == "中高风险":
                health_score = 70
            elif risk_level == "中等风险":
                health_score = 80
            elif risk_level == "低风险":
                health_score = 90
            else:
                health_score = 85 # Default if no risk level or cannot infer

        summary = Summary(
            title=summary_data.get("title", "医疗检测报告"),
            diagnosis=summary_data.get("diagnosis", final_json.get("summary", "N/A")),
            risk_level=summary_data.get("risk_level", risk_level or "N/A"),
            confidence=float(summary_data.get("confidence", 0.0)),
            reason=summary_data.get("reason", "N/A"),
            health_score=health_score
        )

        # Extract indicators
        indicators = []
        for item in final_json.get("indicators", []):
            if isinstance(item, dict):
                indicators.append(Indicator(
                    name=item.get("name", "N/A"),
                    abbreviation=item.get("abbreviation", "N/A"),
                    value=float(item.get("value", 0)) if str(item.get("value", "0")).replace('.', '', 1).isdigit() else 0.0,
                    unit=item.get("unit", "N/A"),
                    reference=item.get("reference", "N/A"),
                    status=item.get("status", "N/A")
                ))

        # Extract agents
        agents = []
        for item in final_json.get("agents", []):
            if isinstance(item, dict):
                agents.append(Agent(
                    department=item.get("name", "N/A"),
                    risk_level=item.get("risk_level", "N/A"),
                    confidence=float(item.get("confidence", 0)),
                    summary=item.get("conclusion", "N/A"),
                    recommendation=item.get("recommendation", [])
                ))

        # Extract similar cases
        similar_cases = []
        for item in final_json.get("similar_cases", []):
            if isinstance(item, dict):
                similar_cases.append(SimilarCase(
                    case_id=item.get("case_id", "N/A"),
                    disease=item.get("disease", "N/A"),
                    similarity=float(item.get("similarity", 0))
                ))

        # Extract knowledge graph
        knowledge_graph = []
        for item in final_json.get("knowledge_graph", []):
            if isinstance(item, dict):
                knowledge_graph.append(KnowledgeGraphItem(
                    disease=item.get("disease", "N/A"),
                    relation=item.get("relation", "N/A"),
                    entity=item.get("entity", "N/A"),
                    value=item.get("value", "N/A")
                ))

        # Extract knowledge
        knowledge = []
        for item in final_json.get("knowledge", []):
            if isinstance(item, dict):
                knowledge.append(KnowledgeItem(
                    source=item.get("source", "N/A"),
                    content=item.get("content", "N/A")
                ))

        # Extract recommendation
        recommendation_data = final_json.get("recommendation", {})
        recommendation = Recommendation(
            diet=recommendation_data.get("diet", []),
            exercise=recommendation_data.get("exercise", []),
            follow_up=recommendation_data.get("follow_up", [])
        )

        # Extract warning
        warning = final_json.get("warning", "N/A")

        return Visualization(
            summary=summary,
            indicators=indicators,
            agents=agents,
            similar_cases=similar_cases,
            knowledge_graph=knowledge_graph,
            knowledge=knowledge,
            recommendation=recommendation,
            warning=warning
        )
    except Exception as e:
        logger.error(f"Error building visualization: {e}", exc_info=True)
        return None

def enhance_result_for_frontend(
    parsed_json: Dict[str, Any],
    extracted_text: str = "",
    user_text: str = ""
) -> Dict[str, Any]:
    if not isinstance(parsed_json, dict):
        parsed_json = {
            "raw_result": parsed_json
        }

    result = dict(parsed_json)

    indicators = normalize_indicators(result)
    result["indicators"] = indicators

    risk_level = (
        result.get("risk_level")
        or result.get("risk")
        or result.get("overall_risk")
        or infer_risk_level(indicators)
    )
    result["risk_level"] = risk_level

    if "summary" not in result:
        summary = (
            result.get("diagnosis_summary")
            or result.get("report_summary")
            or result.get("final_answer")
            or result.get("answer")
            or result.get("message")
        )

        if not summary:
            abnormal_count = sum(1 for item in indicators if is_abnormal_status(item.get("status")))
            if indicators:
                summary = (
                    f"系统共识别到 {len(indicators)} 项体检指标，"
                    f"其中疑似异常或需关注项目 {abnormal_count} 项。"
                    f"综合风险等级为：{risk_level}。"
                )
            else:
                summary = "系统已完成报告解析，但未识别到标准指标列表，请查看 JSON 原始结果。"

        result["summary"] = summary

    result["agents"] = normalize_agents(result, indicators)
    result["critique"] = normalize_critique(result)
    result["advice"] = normalize_advice(result, indicators)

    if extracted_text and "extracted_text" not in result:
        result["extracted_text"] = extracted_text[:1500]

    if user_text and "user_text" not in result:
        result["user_text"] = user_text

    result.setdefault("_frontend_ready", True)

    return result


def build_assistant_message(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return "解析完成，但结果格式异常，请查看 JSON 原始结果。"

    summary = (
        result.get("summary")
        or result.get("report_summary")
        or result.get("diagnosis_summary")
        or result.get("final_report")
        or result.get("conclusion")
        or "系统已完成报告解析。"
    )

    risk_level = (
        result.get("risk_level")
        or result.get("risk")
        or result.get("overall_risk")
        or "未标注"
    )

    indicators = result.get("indicators") or []
    agents = result.get("agents") or []
    advice = result.get("advice") or []
    critique = result.get("critique") or []

    lines = []

    lines.append("📋 智能体检报告辅助诊断书")
    lines.append("")
    lines.append("一、综合结论")
    lines.append(str(summary))
    lines.append("")

    lines.append("二、风险等级")
    lines.append(f"综合风险等级：{risk_level}")
    lines.append("")

    lines.append("三、关键体检指标")
    if isinstance(indicators, list) and indicators:
        for index, item in enumerate(indicators[:12], start=1):
            if isinstance(item, dict):
                name = item.get("name") or "未知指标"
                value = item.get("value") or "--"
                reference = item.get("reference") or "--"
                status = item.get("status") or "未标注"
                lines.append(
                    f"{index}. {name}：{value}，参考范围：{reference}，状态：{status}"
                )
            else:
                lines.append(f"{index}. {item}")

        if len(indicators) > 12:
            lines.append(f"其余 {len(indicators) - 12} 项指标请查看 JSON 结果。")
    else:
        lines.append("暂未识别到标准化体检指标。")
    lines.append("")

    lines.append("四、多 Agent 辅助分析")
    if isinstance(agents, list) and agents:
        for index, agent in enumerate(agents, start=1):
            if isinstance(agent, dict):
                name = agent.get("name") or f"Agent {index}"
                conclusion = agent.get("conclusion") or "暂无结论。"
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

    lines.append("五、健康管理建议")
    if isinstance(advice, list) and advice:
        for index, item in enumerate(advice, start=1):
            if isinstance(item, dict):
                title = item.get("title") or f"建议 {index}"
                content = (
                    item.get("content")
                    or item.get("text")
                    or item.get("message")
                    or item.get("suggestion")
                    or ""
                )
                lines.append(f"{index}. {title}：{content}")
            else:
                lines.append(f"{index}. {item}")
    else:
        lines.append("暂无明确健康建议。")
    lines.append("")

    lines.append("六、校验说明")
    if isinstance(critique, list) and critique:
        for index, item in enumerate(critique, start=1):
            lines.append(f"{index}. {item}")
    else:
        lines.append("当前结果仅作为健康管理与辅助初筛参考，不替代医生诊断。")
    lines.append("")

    lines.append("⚠️ 免责声明：本报告由 AI 根据体检文本或 OCR 结果自动生成，仅供健康管理参考，不能替代临床医生诊断。")

    return "\n".join(lines)


async def extract_text_from_file(file_path: Path, file_type: Optional[str]) -> str:
    extracted_text = ""

    try:
        if file_type == "pdf":
            extracted_text = await pdf_service.extract_text(file_path) or ""

        elif file_type in ["png", "jpg", "jpeg", "bmp", "image"]:
            extracted_text = await ocr_service.extract_text(file_path) or ""

        elif file_type == "txt":
            extracted_text = file_path.read_text(encoding="utf-8", errors="ignore")

    except Exception as e:
        logger.error(f"文件文本提取失败: {e}", exc_info=True)

    return extracted_text


async def call_report_api(ocr_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    调用外部报告生成接口。

    已去掉所有 httpx 时间限制：
    - 不设置 connect timeout
    - 不设置 read timeout
    - 不设置 write timeout
    - 不设置 pool timeout

    注意：
    timeout=None 表示请求会一直等待外部接口返回。
    如果外部接口永久不返回，当前 /api/chat 请求也会一直等待。
    """
    try:
        if not settings.REPORT_GENERATION_API_URL:
            logger.warning("REPORT_GENERATION_API_URL 未配置，跳过报告接口调用")
            return None

        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            payload = {
                "ocr_json": ocr_json
            }

            logger.info(
                f"开始调用报告接口，URL={settings.REPORT_GENERATION_API_URL}，模式=wrapped，无超时限制"
            )

            response = await client.post(
                settings.REPORT_GENERATION_API_URL,
                json=payload
            )

            logger.info(f"报告接口 wrapped 响应状态码: {response.status_code}")

            if response.status_code == 422:
                logger.warning("报告接口 wrapped 模式返回 422，尝试 direct 模式，无超时限制")

                response = await client.post(
                    settings.REPORT_GENERATION_API_URL,
                    json=ocr_json
                )

                logger.info(f"报告接口 direct 响应状态码: {response.status_code}")

        if response.status_code >= 400:
            logger.error(f"报告接口调用失败: {response.status_code}, {response.text}")
            return {
                "success": False,
                "error": f"报告接口调用失败，状态码：{response.status_code}",
                "detail": response.text
            }

        data = response.json()
        logger.info("报告接口调用成功")
        return data

    except Exception as e:
        logger.error(f"调用报告接口异常: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def merge_report_api_result(
    final_json: Dict[str, Any],
    report_result: Dict[str, Any]
) -> Dict[str, Any]:
    if not isinstance(report_result, dict):
        return final_json

    final_json["report_api_result"] = report_result

    # 优先从外部API结果中提取report_text和visualization对象
    api_report_text = report_result.get("report_text")
    api_visualization = report_result.get("visualization")

    if api_report_text:
        final_json["report_text"] = api_report_text # 保存外部API返回的report_text

    if api_visualization and isinstance(api_visualization, dict):
        final_json["visualization_from_api"] = api_visualization # 保存外部API返回的完整visualization对象

        # 从visualization对象中提取字段并更新final_json
        vis_summary = api_visualization.get("summary")
        if vis_summary:
            final_json["summary"] = vis_summary
            # 从visualization.summary中提取risk_level和health_score
            if isinstance(vis_summary, dict):
                if "risk_level" in vis_summary:
                    final_json["risk_level"] = vis_summary["risk_level"]
                if "health_score" in vis_summary:
                    final_json["health_score"] = vis_summary["health_score"]

        vis_indicators = api_visualization.get("indicators")
        if vis_indicators:
            final_json["indicators"] = normalize_indicators({"indicators": vis_indicators}) # 确保经过normalize

        vis_agents = api_visualization.get("agents")
        if vis_agents:
            # normalize_agents需要原始final_json的indicators，这里提供一个空的或已有的
            final_json["agents"] = normalize_agents({"agents": vis_agents}, final_json.get("indicators", []))

        vis_advice = api_visualization.get("recommendation") # 外部API返回的推荐可能在"recommendation"
        if vis_advice:
            final_json["advice"] = normalize_advice({"advice": vis_advice}, final_json.get("indicators", []))
            
        vis_critique = api_visualization.get("critique")
        if vis_critique:
            final_json["critique"] = normalize_critique({"critique": vis_critique})

    # Fallback to top-level data if visualization not used or incomplete (for older API versions)
    data = report_result.get("data") if isinstance(report_result.get("data"), dict) else report_result

    if not isinstance(data, dict):
        return final_json

    # 现有逻辑保持不变，作为补充或兼容老版本
    report_summary = (
        data.get("summary")
        or data.get("report_summary")
        or data.get("diagnosis_summary")
        or data.get("conclusion")
        or data.get("final_report")
        or data.get("report")
    )
    if report_summary and "summary" not in final_json: # 仅当final_json中尚未更新时才使用
        final_json["summary"] = report_summary
        final_json["report_summary"] = report_summary

    report_risk = (
        data.get("risk_level")
        or data.get("risk")
        or data.get("overall_risk")
        or data.get("risk_assessment")
    )
    if report_risk and "risk_level" not in final_json: # 仅当final_json中尚未更新时才使用
        final_json["risk_level"] = report_risk

    report_indicators = (
        data.get("indicators")
        or data.get("metrics")
        or data.get("items")
        or data.get("structured_data")
        or data.get("extracted_indicators")
    )
    if report_indicators and "indicators" not in final_json: # 仅当final_json中尚未更新时才使用
        temp = {"indicators": report_indicators}
        final_json["indicators"] = normalize_indicators(temp)

    report_advice = (
        data.get("advice")
        or data.get("recommendations")
        or data.get("suggestions")
        or data.get("health_advice")
    )
    if report_advice and "advice" not in final_json: # 仅当final_json中尚未更新时才使用
        final_json["advice"] = normalize_advice(
            {"advice": report_advice},
            final_json.get("indicators", [])
        )

    report_agents = (
        data.get("agents")
        or data.get("agent_outputs")
        or data.get("multi_agent")
        or data.get("agent_results")
    )
    if report_agents and "agents" not in final_json: # 仅当final_json中尚未更新时才使用
        final_json["agents"] = normalize_agents(
            {"agents": report_agents},
            final_json.get("indicators", [])
        )

    report_critique = (
        data.get("critique")
        or data.get("validation")
        or data.get("checks")
        or data.get("evidence")
    )
    if report_critique and "critique" not in final_json: # 仅当final_json中尚未更新时才使用
        final_json["critique"] = normalize_critique(
            {"critique": report_critique}
        )

    final_json["_report_api_merged"] = True

    return final_json


@router.post("/chat")
async def chat(
    text: str = Form(""),
    conversation_id: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        conv_service = ConversationService(db)

        file_path = None
        file_name = None
        file_type = None

        if file and file.filename:
            logger.info(f"收到文件: filename={file.filename}")

            file_name = file.filename
            file_type = get_file_type(file_name)

            settings.UPLOAD_DIR.mkdir(exist_ok=True)

            save_name = (
                f"{datetime.now().strftime('%Y%m%d%H%M%S_%f')}_"
                f"{safe_filename(file_name)}"
            )
            file_path = settings.UPLOAD_DIR / save_name

            content = await file.read()
            file_path.write_bytes(content)

            logger.info(f"文件已保存: {file_path}")

        if conversation_id:
            conv = await conv_service.get_conversation(conversation_id, current_user.id)
            if not conv:
                return error_response(msg="Conversation not found")
        else:
            title = build_conversation_title(text=text, file_name=file_name)
            conv = await conv_service.create_conversation(current_user.id, title)

        user_msg_text = build_user_message(text=text, file_name=file_name)

        await conv_service.add_message(
            conv.id,
            MessageRole.USER,
            user_msg_text,
            file_name=file_name,
            file_type=file_type
        )

        parsed_json = None
        extracted_text = ""

        if file_path:
            extracted_text = await extract_text_from_file(file_path, file_type)

            parsed_json = await parser_service.parse_file(
                file_path=file_path,
                file_type=file_type or "unknown",
                user_text=text
            )

            if parsed_json and "ocr" in parsed_json and "source" in parsed_json["ocr"]:
                logger.info(f"当前使用的解析器来源: {parsed_json['ocr']['source']}")

        elif text and text.strip():
            extracted_text = text
            parsed_json = await parser_service.parse_text(text)
            if parsed_json and "ocr" in parsed_json and "source" in parsed_json["ocr"]:
                logger.info(f"当前使用的解析器来源: {parsed_json['ocr']['source']}")

        if not parsed_json:
            parsed_json = {
                "message": "处理完成，但未解析出结构化结果。",
                "extracted_text": extracted_text[:1500] if extracted_text else "",
                "indicators": [],
                "risk_level": "未知",
                "advice": [
                    {
                        "title": "重新上传或补充信息",
                        "content": "请确认文件内容清晰，或直接输入体检指标文本后再次解析。",
                        "tags": ["解析失败"]
                    }
                ]
            }

        final_json = enhance_result_for_frontend(
            parsed_json=parsed_json,
            extracted_text=extracted_text,
            user_text=text
        )

        # 保存原始识别结果用于前端展示“结构化 JSON”
        final_json["extracted_raw"] = parsed_json

        logger.info(f"调用外部报告API前的final_json (部分): {json.dumps(final_json, ensure_ascii=False)[:500]}...")
        report_result = await call_report_api(final_json)
        logger.info(f"外部报告API返回结果 (部分): {json.dumps(report_result, ensure_ascii=False)[:500]}...")

        if report_result is not None:
            final_json = merge_report_api_result(
                final_json=final_json,
                report_result=report_result
            )
            logger.info(f"合并外部报告API结果后的final_json (部分): {json.dumps(final_json, ensure_ascii=False)[:500]}...")

        report_id = None
        has_report = False
        visualization_obj = None

        if final_json:
            visualization_obj = build_visualization(final_json)
            if visualization_obj:
                report_id = str(uuid.uuid4())
                has_report = True
                
        json_str = json.dumps(final_json, ensure_ascii=False)
        
        # 优先使用外部API返回的report_text作为聊天消息
        assistant_msg_content = final_json.get("report_text") or build_assistant_message(final_json)

        # 优先使用外部API返回的visualization对象
        api_visualization_data = final_json.get("visualization_from_api")
        if api_visualization_data:
            try:
                visualization_obj = Visualization(**api_visualization_data) # 直接使用Pydantic模型解析
            except Exception as e:
                logger.error(f"从外部API visualization_data构建visualization_obj失败: {e}", exc_info=True)
                visualization_obj = build_visualization(final_json) # 回退到旧逻辑
        else:
            visualization_obj = build_visualization(final_json)

        assistant_message = await conv_service.add_message(
            conv.id,
            MessageRole.ASSISTANT,
            assistant_msg_content,
            json_result=json_str,
            file_name=file_name,
            file_type=file_type,
            report_id=report_id,
            visualization_json=json.dumps(visualization_obj.model_dump(), ensure_ascii=False) if visualization_obj else None
        )

        response_data = {
            "message": assistant_msg_content,
            "report_id": report_id,
            "has_report": has_report,
            "result": final_json # 添加完整的 final_json
        }

        return success_response(data=response_data)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return error_response(msg=str(e))


@router.get("/report/{report_id}")
async def get_report(
    report_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # 查询数据库获取 ChatMessage 记录
        result = await db.execute(
            select(ChatMessage).where(
                ChatMessage.report_id == report_id
            )
        )
        chat_message = result.scalars().first()

        if not chat_message:
            raise HTTPException(status_code=404, detail="Report not found")

        # 确保是 ASSISTANT 角色且属于当前用户
        # if chat_message.role != MessageRole.ASSISTANT or chat_message.conversation.user_id != current_user.id:
        #     raise HTTPException(status_code=403, detail="Not authorized to access this report")

        visualization_data = None
        if chat_message.visualization_json:
            visualization_data = json.loads(chat_message.visualization_json)

        return success_response(
            data={
                "status": "done",
                "report_text": chat_message.message,
                "visualization": visualization_data
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
    current_user: User = Depends(get_current_user)
):
    try:
        # 查询数据库获取当前聊天中所有带有 report_id 的 ASSISTANT 消息
        result = await db.execute(
            select(ChatMessage).where(
                ChatMessage.conversation_id == chat_id,
                ChatMessage.role == MessageRole.ASSISTANT,
                ChatMessage.report_id.isnot(None)
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
                "created_at": msg.created_at.isoformat()
            })

        return success_response(data=reports)

    except Exception as e:
        logger.error(f"Error fetching reports for chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")