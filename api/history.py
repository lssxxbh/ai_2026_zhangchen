import json
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from models import User
from models.chat_message import MessageRole
from schemas.chat import ConversationResponse, ChatMessageResponse
from api.deps import get_current_user
from services import ConversationService
from utils import success_response, error_response, logger

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/conversations")
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        conv_service = ConversationService(db)
        conversations = await conv_service.get_user_conversations(current_user.id)

        result = [
            ConversationResponse(
                id=conv.id,
                title=conv.title,
                created_at=conv.created_at
            )
            for conv in conversations
        ]

        return success_response(data=result)
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        return error_response(msg=str(e))


@router.get("/history/{conversation_id}")
async def get_history(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        conv_service = ConversationService(db)

        conv = await conv_service.get_conversation(conversation_id, current_user.id)
        if not conv:
            return error_response(msg="Conversation not found")

        messages = await conv_service.get_messages(conversation_id)

        result = []
        for msg in messages:
            json_result = None
            if msg.json_result:
                try:
                    json_result = json.loads(msg.json_result)
                except:
                    pass

            # 正确处理 role 字段
            if isinstance(msg.role, MessageRole):
                role_str = msg.role.value
            else:
                role_str = str(msg.role).lower()
            
            result.append(
                ChatMessageResponse(
                    id=msg.id,
                    role=role_str,
                    message=msg.message,
                    json_result=json_result,
                    file_name=msg.file_name,
                    file_type=msg.file_type,
                    created_at=msg.created_at
                )
            )

        return success_response(data=result)
    except Exception as e:
        logger.error(f"Get history error: {e}")
        return error_response(msg=str(e))


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        conv_service = ConversationService(db)
        success = await conv_service.delete_conversation(conversation_id, current_user.id)

        if success:
            return success_response(msg="Conversation deleted")
        else:
            return error_response(msg="Conversation not found")
    except Exception as e:
        logger.error(f"Delete conversation error: {e}")
        return error_response(msg=str(e))
