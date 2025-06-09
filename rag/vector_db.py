import logging
from sqlalchemy import select, bindparam
from sqlalchemy.ext.asyncio import AsyncSession
from models import Document
from pgvector.sqlalchemy import Vector
import numpy as np

logger = logging.getLogger(__name__)

async def search_similar_documents(session: AsyncSession, query_embedding: list[float], limit: int = 3) -> list[str]:
    try:
        embedding_array = np.array(query_embedding, dtype=np.float32)
        embedding_list = embedding_array.tolist()

        embedding_param = bindparam('embedding_1', type_=Vector(768))
        distance_expr = Document.embedding.op('<->')(embedding_param)

        stmt = select(Document.content).order_by(distance_expr).limit(limit)
        params = {'embedding_1': embedding_list, 'param_1': limit}
        result = await session.execute(stmt, params)
        rows = result.fetchall()
        return [row[0] for row in rows]

    except Exception as e:
        logger.exception(f"Exception during vector search: {e}")
        return []
