from sqlalchemy import Column, Integer, BigInteger, Text, TIMESTAMP, func
from database import Base

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, nullable=False)
    role = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
