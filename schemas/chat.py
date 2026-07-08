from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime


class ChatRequest(BaseModel):
    text: str
    conversation_id: Optional[int] = None


class ChatMessageResponse(BaseModel):
    id: int
    role: str
    message: str
    json_result: Optional[Dict[str, Any]] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    report_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    conversation_id: int
    message_id: int
    result: Dict[str, Any]


class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: datetime

    class Config:
        from_attributes = True


class MessageListResponse(BaseModel):
    messages: List[ChatMessageResponse]
