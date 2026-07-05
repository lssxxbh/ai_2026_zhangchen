import json
import httpx
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from database import get_db
from models import User
from models.chat_message import MessageRole
from schemas.chat import ChatResponse
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
parser_service = ParserService(ocr_service, pdf_service, text_clean_service, ai_service)


async def call_report_generation_api(parsed_data: dict) -> dict:
    """调用外部报告生成 API"""
    if not settings.REPORT_GENERATION_API_URL:
        logger.warning("REPORT_GENERATION_API_URL 未配置")
        return None
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                settings.REPORT_GENERATION_API_URL,
                json=parsed_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"报告生成 API 调用成功")
            return result
    except Exception as e:
        logger.error(f"报告生成 API 调用失败: {e}", exc_info=True)
        return None


@router.post("/chat")
async def chat(
    text: str = Form(""),
    conversation_id: int = Form(None),
    file: UploadFile = File(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        conv_service = ConversationService(db)

        if conversation_id:
            conv = await conv_service.get_conversation(conversation_id, current_user.id)
            if not conv:
                return error_response(msg="Conversation not found")
        else:
            title = text[:30] if text else f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            conv = await conv_service.create_conversation(current_user.id, title)

        file_path = None
        file_name = None
        file_type = None

        if file:
            logger.info(f"收到文件: filename={file.filename}")
            file_ext = Path(file.filename).suffix.lower()[1:] if file.filename else ""
            file_type = file_ext
            file_name = file.filename

            save_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
            file_path = settings.UPLOAD_DIR / save_name

            content = await file.read()
            file_path.write_bytes(content)

        user_msg_text = text
        if file:
            user_msg_text = f"{text}\n[附件: {file_name}]" if text else f"[附件: {file_name}]"

        await conv_service.add_message(
            conv.id,
            MessageRole.USER,
            user_msg_text,
            file_name=file_name,
            file_type=file_type
        )

        parsed_json = None
        extracted_text = ""
        
        if file and file_path:
            # 先尝试提取文本
            try:
                if file_type == "pdf":
                    extracted_text = await pdf_service.extract_text(file_path) or ""
                elif file_type in ["png", "jpg", "jpeg", "bmp", "image"]:
                    extracted_text = await ocr_service.extract_text(file_path) or ""
                elif file_type == "txt":
                    extracted_text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.error(f"文件文本提取失败: {e}", exc_info=True)
            
            # 再尝试完整解析
            parsed_json = await parser_service.parse_file(
                file_path,
                file_type,
                text
            )
        elif text:
            extracted_text = text
            parsed_json = await parser_service.parse_text(text)

        if not parsed_json:
            parsed_json = {
                "message": "处理完成",
                "extracted_text": extracted_text[:1000] if extracted_text else ""
            }
        else:
            # 确保结果中包含提取到的原始文本
            if extracted_text and "extracted_text" not in parsed_json:
                parsed_json["extracted_text"] = extracted_text[:1000]

        # 调用外部报告生成 API - 传入完整的识别结果
        assistant_msg = "已解析完成，查看下方JSON结果"
        report_api_result = None
        
        if parsed_json:
            report_api_result = await call_report_generation_api(parsed_json)
            if report_api_result and report_api_result.get("status") == "done":
                report_text = report_api_result.get("report_text", "")
                if report_text:
                    assistant_msg = report_text
                    logger.info("使用外部 API 生成的报告文本")
        
        json_str = json.dumps(parsed_json, ensure_ascii=False)

        assistant_message = await conv_service.add_message(
            conv.id,
            MessageRole.ASSISTANT,
            assistant_msg,
            json_result=json_str
        )

        return success_response(
            data=ChatResponse(
                conversation_id=conv.id,
                message_id=assistant_message.id,
                result=parsed_json
            )
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return error_response(msg=str(e))

