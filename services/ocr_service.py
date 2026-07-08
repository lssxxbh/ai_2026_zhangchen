from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple
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
            from PIL import Image, ImageEnhance, ImageFilter, ImageOps
            import tempfile
            
            with Image.open(image_path) as img:
                # 转换为 RGB 模式，避免模式问题
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')

                # 检验报告通常是细线表格 + 小字号数字，适当放大和增强对比度能显著减少
                # 7/T、8/吕、6/G 这类混淆。
                width, height = img.size
                target_width = 1800
                if width < target_width:
                    scale = target_width / max(width, 1)
                    img = img.resize(
                        (target_width, int(height * scale)),
                        Image.Resampling.LANCZOS
                    )

                img = ImageOps.grayscale(img)
                img = ImageOps.autocontrast(img)
                img = ImageEnhance.Contrast(img).enhance(1.35)
                img = img.filter(ImageFilter.SHARPEN)
                
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
            detections = []
            for detection in result:
                if detection and len(detection) > 1:
                    text = detection[1]
                    if text and isinstance(text, str):
                        texts.append(text)
                        detections.append({
                            "box": detection[0],
                            "text": text,
                            "confidence": detection[2] if len(detection) > 2 else None
                        })
            
            full_text = self._format_detections_as_text(detections) or "\n".join(texts)
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
            detections = []
            for line in result[0]:
                if line and len(line) > 1 and line[1]:
                    text = line[1][0] if len(line[1]) > 0 else ""
                    if text:
                        texts.append(text)
                        detections.append({
                            "box": line[0],
                            "text": text,
                            "confidence": line[1][1] if len(line[1]) > 1 else None
                        })

            full_text = self._format_detections_as_text(detections) or "\n".join(texts)
            logger.info(f"✅ PaddleOCR extracted {len(full_text)} characters")
            return full_text
        except Exception as e:
            logger.error(f"❌ PaddleOCR error: {e}", exc_info=True)
            return None

    def _format_detections_as_text(self, detections: List[Dict[str, Any]]) -> str:
        """按 OCR 检测框重建阅读顺序，尽量保留表格的行列关系。"""
        cells = []
        for item in detections:
            box = item.get("box")
            text = str(item.get("text", "")).strip()
            if not box or not text:
                continue

            xs, ys = self._box_xy(box)
            if not xs or not ys:
                continue
            height = max(ys) - min(ys)
            cells.append({
                "text": text,
                "x": (min(xs) + max(xs)) / 2,
                "y": (min(ys) + max(ys)) / 2,
                "height": max(height, 1),
            })

        if not cells:
            return ""

        heights = [cell["height"] for cell in cells]
        row_threshold = max(12, median(heights) * 0.8)
        rows: List[Dict[str, Any]] = []

        for cell in sorted(cells, key=lambda c: (c["y"], c["x"])):
            target = None
            for row in rows:
                if abs(cell["y"] - row["y"]) <= row_threshold:
                    target = row
                    break
            if target is None:
                rows.append({"y": cell["y"], "cells": [cell]})
            else:
                target["cells"].append(cell)
                target["y"] = sum(c["y"] for c in target["cells"]) / len(target["cells"])

        lines = []
        for row in sorted(rows, key=lambda r: r["y"]):
            row_cells = sorted(row["cells"], key=lambda c: c["x"])
            line = "\t".join(cell["text"] for cell in row_cells)
            if line.strip():
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _box_xy(box: Any) -> Tuple[List[float], List[float]]:
        xs: List[float] = []
        ys: List[float] = []
        try:
            for point in box:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    xs.append(float(point[0]))
                    ys.append(float(point[1]))
        except Exception:
            return [], []
        return xs, ys
