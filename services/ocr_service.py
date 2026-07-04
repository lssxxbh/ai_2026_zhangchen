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

    def _validate_image(self, image_path: Path) -> bool:
        """验证图片文件是否有效"""
        try:
            if not image_path.exists():
                logger.error(f"❌ Image file not found: {image_path}")
                return False
            
            file_size = image_path.stat().st_size
            if file_size == 0:
                logger.error(f"❌ Image file is empty: {image_path}")
                return False
            
            # 验证图片格式
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    img.verify()
                return True
            except Exception as e:
                logger.error(f"❌ Invalid image format: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ Image validation error: {e}")
            return False

    def _preprocess_image(self, image_path: Path) -> Optional[Path]:
        """预处理图片，转换为安全格式"""
        try:
            from PIL import Image
            import tempfile
            
            with Image.open(image_path) as img:
                # 转换为 RGB 模式，避免模式问题
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                
                # 保存为临时文件
                temp_path = Path(tempfile.mktemp(suffix='.png'))
                img.save(temp_path, format='PNG')
                logger.info(f"✅ Preprocessed image saved to: {temp_path}")
                return temp_path
        except Exception as e:
            logger.error(f"❌ Image preprocessing error: {e}")
            return None

    async def extract_text(self, image_path: Path) -> Optional[str]:
        """从图片提取文字"""
        if self.engine is None:
            logger.error("❌ No OCR engine available")
            return None
        
        # 验证图片
        if not self._validate_image(image_path):
            return None
        
        temp_image_path = None
        try:
            # 预处理图片
            temp_image_path = self._preprocess_image(image_path)
            process_path = temp_image_path if temp_image_path else image_path
            
            if self.engine == "easyocr":
                result = await self._extract_easyocr(process_path)
                if result is not None:
                    return result
                logger.warning("⚠️ EasyOCR failed, trying fallback method")
            elif self.engine == "paddleocr":
                return await self._extract_paddleocr(process_path)
            
            return None
        except Exception as e:
            logger.error(f"❌ OCR extraction error: {e}", exc_info=True)
            return None
        finally:
            # 清理临时文件
            if temp_image_path and temp_image_path.exists():
                try:
                    temp_image_path.unlink()
                except:
                    pass

    async def _extract_easyocr(self, image_path: Path) -> Optional[str]:
        """使用 EasyOCR 提取文字"""
        try:
            if not self.reader:
                logger.error("❌ EasyOCR reader not initialized")
                return None
            
            result = self.reader.readtext(str(image_path))
            
            if result is None:
                logger.warning("⚠️ EasyOCR returned None")
                return ""
            
            texts = []
            for detection in result:
                if detection and len(detection) > 1:
                    text = detection[1]
                    if text and isinstance(text, str):
                        texts.append(text)
            
            full_text = "\n".join(texts)
            logger.info(f"✅ EasyOCR extracted {len(full_text)} characters")
            return full_text
        except Exception as e:
            logger.error(f"❌ EasyOCR error: {e}", exc_info=True)
            return None

    async def _extract_paddleocr(self, image_path: Path) -> Optional[str]:
        """使用 PaddleOCR 提取文字"""
        try:
            if not self.paddle_ocr:
                logger.error("❌ PaddleOCR not initialized")
                return None
            
            result = self.paddle_ocr.ocr(str(image_path), cls=True)
            
            if not result or not result[0]:
                logger.warning("⚠️ PaddleOCR returned empty result")
                return ""

            texts = []
            for line in result[0]:
                if line and len(line) > 1 and line[1]:
                    text = line[1][0] if len(line[1]) > 0 else ""
                    if text:
                        texts.append(text)

            full_text = "\n".join(texts)
            logger.info(f"✅ PaddleOCR extracted {len(full_text)} characters")
            return full_text
        except Exception as e:
            logger.error(f"❌ PaddleOCR error: {e}", exc_info=True)
            return None
