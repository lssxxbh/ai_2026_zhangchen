from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from models import User
from schemas.auth import UserCreate, UserLogin, Token
from utils import verify_password, get_password_hash, create_access_token, success_response, error_response, logger

router = APIRouter(prefix="/api", tags=["auth"])


@router.post("/register")
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(
            select(User).where(User.username == user_data.username)
        )
        existing_user = result.scalar_one_or_none()
        if existing_user:
            return error_response(msg="Username already exists")

        hashed_password = get_password_hash(user_data.password)
        new_user = User(
            username=user_data.username,
            password_hash=hashed_password
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        access_token = create_access_token(
            data={"sub": new_user.username, "user_id": new_user.id}
        )

        return success_response(
            data=Token(
                access_token=access_token,
                user_id=new_user.id,
                username=new_user.username
            )
        )
    except Exception as e:
        logger.error(f"Register error: {e}")
        return error_response(msg=str(e))


@router.post("/login")
async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(
            select(User).where(User.username == user_data.username)
        )
        user = result.scalar_one_or_none()

        if not user or not verify_password(user_data.password, user.password_hash):
            return error_response(msg="Invalid username or password")

        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id}
        )

        return success_response(
            data=Token(
                access_token=access_token,
                user_id=user.id,
                username=user.username
            )
        )
    except Exception as e:
        logger.error(f"Login error: {e}")
        return error_response(msg=str(e))
