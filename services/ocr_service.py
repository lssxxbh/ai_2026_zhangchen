from pathlib import Path
from typing import Optional
from utils.logger import logger

# 优先尝试 EasyOCR（Python 3.11 更稳定），备选 PaddleOCR
_OCR_ENGINE = None

try:
    import easyocr
    _OCR_ENGINE = "easyocr"
    logger.info("✅ Using EasyOCR for OCR")
except ImportError:
    try:
        from paddleocr import PaddleOCR
        _OCR_ENGINE = "paddleocr"
        logger.info("✅ Using PaddleOCR for OCR")
    except ImportError:
        logger.warning("⚠️ No OCR engine available")


class OCRService:
    def __init__(self):
        self.reader = None
        self.paddle_ocr = None
        self.engine = _OCR_ENGINE
        self._initialize_ocr()

    def _initialize_ocr(self):
        """初始化 OCR 引擎"""
        if self.engine == "easyocr":
            try:
                self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                logger.info("✅ EasyOCR initialized successfully")
            except Exception as e:
                logger.error(f"❌ EasyOCR init failed: {e}")
                self.engine = None
        elif self.engine == "paddleocr":
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
                logger.info("✅ PaddleOCR initialized successfully")
            except Exception as e:
                logger.error(f"❌ PaddleOCR init failed: {e}")
                self.engine = None
        else:
            logger.warning("⚠️ No OCR engine available")

    async def extract_text(self, image_path: Path) -> Optional[str]:
        """从图片提取文字"""
        if self.engine is None:
            logger.error("❌ No OCR engine available")
            return None

        try:
            if self.engine == "easyocr":
                return await self._extract_easyocr(image_path)
            elif self.engine == "paddleocr":
                return await self._extract_paddleocr(image_path)
            else:
                return None
        except Exception as e:
            logger.error(f"❌ OCR extraction error: {e}")
            return None

    async def _extract_easyocr(self, image_path: Path) -> Optional[str]:
        """使用 EasyOCR 提取文字"""
        try:
            result = self.reader.readtext(str(image_path))
            texts = []
            for detection in result:
                if len(detection) > 1:
                    texts.append(detection[1])
            full_text = "\n".join(texts)
            logger.info(f"✅ EasyOCR extracted {len(full_text)} characters")
            return full_text
        except Exception as e:
            logger.error(f"❌ EasyOCR error: {e}")
            return None

    async def _extract_paddleocr(self, image_path: Path) -> Optional[str]:
        """使用 PaddleOCR 提取文字"""
        try:
            result = self.paddle_ocr.ocr(str(image_path), cls=True)
            if not result or not result[0]:
                return ""

            texts = []
            for line in result[0]:
                if line and len(line) > 1:
                    texts.append(line[1][0])

            full_text = "\n".join(texts)
            logger.info(f"✅ PaddleOCR extracted {len(full_text)} characters")
            return full_text
        except Exception as e:
            logger.error(f"❌ PaddleOCR error: {e}")
            return None
