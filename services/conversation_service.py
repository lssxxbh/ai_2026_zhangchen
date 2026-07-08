from typing import List, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from models import User, Conversation, ChatMessage
from models.chat_message import MessageRole
from utils.logger import logger


class ConversationService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_conversation(self, user_id: int, title: str) -> Conversation:
        conv = Conversation(user_id=user_id, title=title)
        self.db.add(conv)
        await self.db.commit()
        await self.db.refresh(conv)
        return conv

    async def get_user_conversations(self, user_id: int) -> List[Conversation]:
        result = await self.db.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_conversation(self, conv_id: int, user_id: int) -> Optional[Conversation]:
        result = await self.db.execute(
            select(Conversation)
            .where(Conversation.id == conv_id, Conversation.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def delete_conversation(self, conv_id: int, user_id: int) -> bool:
        result = await self.db.execute(
            delete(Conversation)
            .where(Conversation.id == conv_id, Conversation.user_id == user_id)
        )
        await self.db.commit()
        return result.rowcount > 0

    async def add_message(
        self,
        conv_id: int,
        role: MessageRole,
        message: str,
        json_result: Optional[str] = None,
        file_name: Optional[str] = None,
        file_type: Optional[str] = None,
        report_id: Optional[str] = None,
        visualization_json: Optional[str] = None
    ) -> ChatMessage:
        msg = ChatMessage(
            conversation_id=conv_id,
            role=role,
            message=message,
            json_result=json_result,
            file_name=file_name,
            file_type=file_type,
            report_id=report_id,
            visualization_json=visualization_json
        )
        self.db.add(msg)
        await self.db.commit()
        await self.db.refresh(msg)
        return msg

    async def get_messages(self, conv_id: int) -> List[ChatMessage]:
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conv_id)
            .order_by(ChatMessage.created_at)
        )
        return list(result.scalars().all())
