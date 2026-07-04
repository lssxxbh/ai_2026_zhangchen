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
            parse_status = {
                "file_type": file_type,
                "text_extracted": False,
                "text_length": 0
            }

            if file_type in ["image", "png", "jpg", "jpeg", "bmp"]:
                logger.info("使用 OCR 解析图片")
                raw_text = await self.ocr_service.extract_text(file_path) or ""
                logger.info(f"OCR 提取到 {len(raw_text)} 字符")
                parse_status["text_extracted"] = len(raw_text) > 0
            elif file_type == "pdf":
                logger.info("使用 PDF 服务解析")
                raw_text = await self.pdf_service.extract_text(file_path) or ""
                logger.info(f"PDF 提取到 {len(raw_text)} 字符")
                parse_status["text_extracted"] = len(raw_text) > 0
            elif file_type == "txt":
                logger.info("直接读取文本文件")
                try:
                    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
                    logger.info(f"文本文件读取到 {len(raw_text)} 字符")
                    parse_status["text_extracted"] = len(raw_text) > 0
                except Exception as e:
                    logger.error(f"读取文本文件失败: {e}")

            parse_status["text_length"] = len(raw_text)

            # 如果没有提取到文本但有用户输入，使用用户输入
            if not raw_text and user_text:
                logger.info("文件未提取到文本，使用用户输入的文本")
                raw_text = user_text
                parse_status["used_user_text"] = True

            # 无论如何都尝试解析，即使文本为空
            logger.info(f"开始清理文本，原始长度: {len(raw_text)}")
            cleaned_text = self.text_clean_service.clean_text(raw_text)
            logger.info(f"文本清理完成，长度: {len(cleaned_text)}")

            # 构建结果基础信息
            base_result = {
                "_parse_status": parse_status,
                "raw_text_preview": raw_text[:500] if raw_text else ""
            }

            if not cleaned_text:
                logger.warning("清理后没有文本，返回基础信息")
                base_result["message"] = "未能从文件中提取到有效文本内容"
                return base_result

            logger.info("开始 AI 解析")
            result = await self.ai_service.parse_report(cleaned_text, user_text)
            
            if result:
                # 合并基础信息到 AI 解析结果
                if isinstance(result, dict):
                    result.update(base_result)
                logger.info(f"AI 解析完成")
                return result
            else:
                logger.warning("AI 解析返回空结果，返回基础信息")
                base_result["message"] = "AI 解析未能生成结构化数据"
                return base_result

        except Exception as e:
            logger.error(f"Parser error: {e}", exc_info=True)
            return {
                "error": str(e),
                "message": "文件解析过程中发生错误",
                "raw_text_preview": ""
            }

    async def parse_text(
        self,
        text: str
    ) -> Optional[Dict[str, Any]]:
        try:
            logger.info(f"开始解析文本: {text[:100]}...")
            cleaned_text = self.text_clean_service.clean_text(text)
            logger.info(f"文本清理完成")
            
            base_result = {
                "raw_text_preview": text[:500]
            }
            
            if not cleaned_text:
                base_result["message"] = "文本内容为空"
                return base_result
            
            result = await self.ai_service.parse_report(cleaned_text)
            
            if result and isinstance(result, dict):
                result.update(base_result)
                logger.info(f"解析完成")
                return result
            else:
                base_result["message"] = "AI 解析未能生成结构化数据"
                return base_result
                
        except Exception as e:
            logger.error(f"Text parse error: {e}", exc_info=True)
            return {
                "error": str(e),
                "message": "文本解析过程中发生错误"
            }
