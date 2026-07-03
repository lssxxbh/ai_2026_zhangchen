from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.sql import func
from sqlalchemy.types import TypeDecorator, VARCHAR
from database import Base
import enum


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    
    @classmethod
    def _missing_(cls, value):
        """支持不区分大小写的枚举值查找"""
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None


class MessageRoleType(TypeDecorator):
    """自定义类型，处理 MessageRole 枚举的序列化和反序列化"""
    impl = VARCHAR(50)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, MessageRole):
            return value.value
        if isinstance(value, str):
            return value.lower()
        return str(value).lower()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return MessageRole._missing_(value) or value


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(MessageRoleType, nullable=False)
    message = Column(Text, nullable=False)
    json_result = Column(Text, nullable=True)
    file_name = Column(String(255), nullable=True)
    file_type = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
