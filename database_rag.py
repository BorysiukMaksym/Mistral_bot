from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection

from settings import PGVECTOR_URL_ASYNC

async_engine = create_async_engine(PGVECTOR_URL_ASYNC, echo=True)
pgvector_session = async_sessionmaker(async_engine, expire_on_commit=False)

async def init_pgvector():
    async with async_engine.connect() as conn:
        raw_conn_wrapper = await conn.get_raw_connection()
        raw = raw_conn_wrapper.driver_connection

        if not isinstance(raw, AsyncConnection):
            raise TypeError(f"Expected AsyncConnection, got {type(raw)}")

        await register_vector_async(raw)

# -------- Створення EXTENSION, якщо потрібно --------
async def ensure_pgvector_extension():
    async with async_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

# -------- Dependency --------
async def get_pgvector_session() -> AsyncSession:
    async with pgvector_session() as session:
        yield session
