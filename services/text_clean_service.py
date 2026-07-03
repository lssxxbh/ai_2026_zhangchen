import re
from utils.logger import logger


class TextCleanService:
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""

        original_len = len(text)

        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        text = re.sub(r"\r\n|\r", "\n", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        text = re.sub(r"[ \t]+", " ", text)

        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)

        logger.info(f"Text cleaned: {original_len} -> {len(text)} chars")
        return text
