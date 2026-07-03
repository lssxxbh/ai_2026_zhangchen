from pydantic import BaseModel
from typing import Dict, Any, Optional


class ReportParseResponse(BaseModel):
    success: bool
    raw_text: Optional[str] = None
    parsed_json: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
