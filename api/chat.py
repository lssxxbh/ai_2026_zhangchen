import json
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, Form, Request
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


@router.post("/chat")
async def chat(
    request: Request,
    text: str = Form(""),
    conversation_id: int = Form(None),
    file: UploadFile = File(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        logger.info("="*50)
        logger.info("开始处理 chat 请求")
        
        # 超级详细的调试日志
        logger.info(f"DEBUG: text = {repr(text)}")
        logger.info(f"DEBUG: conversation_id = {conversation_id}")
        logger.info(f"DEBUG: file = {file}")
        if file:
            logger.info(f"DEBUG: file.filename = {file.filename}")
            logger.info(f"DEBUG: file.content_type = {file.content_type}")
        
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
            logger.info(f"收到文件: filename={file.filename}, content_type={file.content_type}")
            file_ext = Path(file.filename).suffix.lower()[1:] if file.filename else ""
            file_type = file_ext
            file_name = file.filename
            logger.info(f"文件类型: {file_type}")

            save_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
            file_path = settings.UPLOAD_DIR / save_name
            logger.info(f"保存文件到: {file_path}")

            content = await file.read()
            logger.info(f"文件大小: {len(content)} bytes")
            file_path.write_bytes(content)
            logger.info("文件保存成功")

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
            logger.info(f"开始解析文件，类型: {file_type}")
            # 先尝试提取文本，即使后面解析失败也能返回原始文本
            try:
                if file_type == "pdf":
                    logger.info("DEBUG: 处理PDF文件")
                    extracted_text = await pdf_service.extract_text(file_path) or ""
                elif file_type in ["png", "jpg", "jpeg", "bmp", "image"]:
                    logger.info("DEBUG: 处理图片文件")
                    extracted_text = await ocr_service.extract_text(file_path) or ""
                elif file_type == "txt":
                    logger.info("DEBUG: 处理TXT文件")
                    extracted_text = file_path.read_text(encoding="utf-8", errors="ignore")
                
                logger.info(f"文件提取到 {len(extracted_text)} 字符的文本")
                if extracted_text:
                    logger.info(f"文本预览: {repr(extracted_text[:200])}")
            except Exception as e:
                logger.error(f"文件文本提取失败: {e}", exc_info=True)
            
            # 再尝试完整解析
            parsed_json = await parser_service.parse_file(
                file_path,
                file_type,
                text
            )
            logger.info(f"完整解析结果: {parsed_json}")
        elif text:
            logger.info(f"开始解析文本: {text[:100]}")
            extracted_text = text
            parsed_json = await parser_service.parse_text(text)
            logger.info(f"解析结果: {parsed_json}")
        else:
            logger.warning("没有收到文件，也没有收到文本！")

        if not parsed_json:
            logger.warning("解析结果为空，使用默认值")
            parsed_json = {
                "message": "处理完成", 
                "_debug": "解析结果为空",
                "extracted_text": extracted_text[:1000] if extracted_text else ""
            }
        else:
            # 确保结果中包含提取到的原始文本，方便调试
            if extracted_text and "extracted_text" not in parsed_json:
                parsed_json["extracted_text"] = extracted_text[:1000]

        json_str = json.dumps(parsed_json, ensure_ascii=False)

        assistant_msg = "已解析完成，查看下方JSON结果"
        if "question" in parsed_json and parsed_json["question"]:
            assistant_msg = f"已处理您的问题: {parsed_json['question']}"

        assistant_message = await conv_service.add_message(
            conv.id,
            MessageRole.ASSISTANT,
            assistant_msg,
            json_result=json_str
        )

        logger.info("请求处理完成")
        logger.info("="*50)

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

