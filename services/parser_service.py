from pathlib import Path
from typing import Optional, Dict, Any
from services.ocr_service import OCRService
from services.pdf_service import PDFService
from services.text_clean_service import TextCleanService
from services.ai_service import AIService
from utils.logger import logger


class ParserService:
    def __init__(
        self,
        ocr_service: OCRService,
        pdf_service: PDFService,
        text_clean_service: TextCleanService,
        ai_service: AIService
    ):
        self.ocr_service = ocr_service
        self.pdf_service = pdf_service
        self.text_clean_service = text_clean_service
        self.ai_service = ai_service

    async def parse_file(
        self,
        file_path: Path,
        file_type: str,
        user_text: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            logger.info(f"开始解析文件: {file_path}, 类型: {file_type}")
            raw_text = ""

            if file_type in ["image", "png", "jpg", "jpeg", "bmp"]:
                logger.info("使用 OCR 解析图片")
                raw_text = await self.ocr_service.extract_text(file_path) or ""
                logger.info(f"OCR 提取到 {len(raw_text)} 字符")
            elif file_type == "pdf":
                logger.info("使用 PDF 服务解析")
                raw_text = await self.pdf_service.extract_text(file_path) or ""
                logger.info(f"PDF 提取到 {len(raw_text)} 字符")
            elif file_type == "txt":
                logger.info("直接读取文本文件")
                raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
                logger.info(f"文本文件读取到 {len(raw_text)} 字符")

            if not raw_text and user_text:
                logger.info("文件未提取到文本，使用用户输入的文本")
                raw_text = user_text

            if not raw_text:
                logger.warning("没有可解析的文本")
                return None

            logger.info(f"开始清理文本，原始长度: {len(raw_text)}")
            cleaned_text = self.text_clean_service.clean_text(raw_text)
            logger.info(f"文本清理完成，长度: {len(cleaned_text)}")

            if not cleaned_text:
                logger.warning("清理后没有文本")
                return None

            logger.info("开始 AI 解析")
            result = await self.ai_service.parse_report(cleaned_text, user_text)
            logger.info(f"AI 解析完成，结果: {result}")
            return result

        except Exception as e:
            logger.error(f"Parser error: {e}", exc_info=True)
            return None

    async def parse_text(
        self,
        text: str
    ) -> Optional[Dict[str, Any]]:
        try:
            logger.info(f"开始解析文本: {text[:100]}...")
            cleaned_text = self.text_clean_service.clean_text(text)
            logger.info(f"文本清理完成")
            result = await self.ai_service.parse_report(cleaned_text)
            logger.info(f"解析完成")
            return result
        except Exception as e:
            logger.error(f"Text parse error: {e}", exc_info=True)
            return None
