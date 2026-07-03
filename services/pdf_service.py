from pathlib import Path
from typing import Optional
import fitz
from pdf2image import convert_from_path
from services.ocr_service import OCRService
from utils.logger import logger


class PDFService:
    def __init__(self, ocr_service: OCRService):
        self.ocr_service = ocr_service

    async def extract_text(self, pdf_path: Path) -> Optional[str]:
        try:
            logger.info(f"开始提取PDF文本: {pdf_path}")
            doc = fitz.open(pdf_path)
            logger.info(f"PDF页数: {len(doc)}")
            texts = []

            for page_num, page in enumerate(doc):
                text = page.get_text()
                logger.info(f"第 {page_num+1} 页提取到 {len(text)} 字符")
                if text.strip():
                    texts.append(text)

            doc.close()

            full_text = "\n".join(texts)
            logger.info(f"合并后总长度: {len(full_text)} 字符")

            if len(full_text.strip()) < 50:
                logger.info("PDF文本较少，尝试OCR")
                full_text = await self._extract_with_ocr(pdf_path)
                logger.info(f"OCR后总长度: {len(full_text)} 字符")

            logger.info(f"PDF提取完成，最终文本长度: {len(full_text)}")
            if full_text:
                logger.info(f"前200字符: {repr(full_text[:200])}")
            return full_text
        except Exception as e:
            logger.error(f"PDF提取错误: {e}", exc_info=True)
            return None

    async def _extract_with_ocr(self, pdf_path: Path) -> str:
        try:
            logger.info(f"开始PDF OCR处理: {pdf_path}")
            images = convert_from_path(pdf_path)
            logger.info(f"转换为 {len(images)} 页图片")
            all_texts = []

            for i, image in enumerate(images):
                temp_path = pdf_path.parent / f"temp_page_{i}.png"
                image.save(temp_path, "PNG")
                try:
                    text = await self.ocr_service.extract_text(temp_path)
                    if text:
                        all_texts.append(text)
                        logger.info(f"第 {i+1} 页OCR提取到 {len(text)} 字符")
                finally:
                    if temp_path.exists():
                        temp_path.unlink()

            result = "\n".join(all_texts)
            logger.info(f"OCR完成，总长度: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"PDF OCR错误: {e}", exc_info=True)
            return ""
