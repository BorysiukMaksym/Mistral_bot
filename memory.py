from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import ChatHistory

async def save_message(session: AsyncSession, user_id: int, role: str, content: str):
    message = ChatHistory(user_id=user_id, role=role, content=content)
    session.add(message)
    await session.commit()

async def get_chat_history(session: AsyncSession, user_id: int, limit: int = 5):
    result = await session.execute(
        select(ChatHistory).where(ChatHistory.user_id == user_id).order_by(ChatHistory.created_at.desc()).limit(limit)
    )
    messages = result.scalars().all()

    return list(reversed(messages))
