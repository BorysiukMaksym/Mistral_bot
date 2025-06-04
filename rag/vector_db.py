import logging
from sqlalchemy import select, bindparam
from sqlalchemy.ext.asyncio import AsyncSession
from models import Document
from pgvector.sqlalchemy import Vector
import numpy as np

# Налаштування базового логування
logging.basicConfig(
    level=logging.DEBUG,  # Виставити рівень виводу в DEBUG
    format='[%(levelname)s] %(message)s',
)

# Отримуємо logger для поточного модуля
logger = logging.getLogger(__name__)

async def search_similar_documents(session: AsyncSession, query_embedding: list[float], limit: int = 3) -> list[str]:
    try:
        logger.debug(f"Embedding type before np.array: {type(query_embedding)}")
        embedding_array = np.array(query_embedding, dtype=np.float32)
        embedding_list = embedding_array.tolist()

        logger.debug(f"Embedding type before execute: {type(embedding_list)}")
        logger.debug(f"Embedding sample before execute: {embedding_list[:5]}")

        embedding_param = bindparam('embedding_1', type_=Vector(384))
        distance_expr = Document.embedding.op('<->')(embedding_param)

        stmt = select(Document.content).order_by(distance_expr).limit(limit)

        # Передаємо САМ СПИСОК чисел, а не рядок
        logger.debug(f"Type of embedding_list: {type(embedding_list)}")
        logger.debug(f"Sample embedding_list (first 5 elements): {embedding_list[:5]}")

        params = {'embedding_1': embedding_list, 'param_1': limit}

        logger.debug(f"Params to execute: {params}")
        logger.debug(f"Type of embedding_list param: {type(params['embedding_1'])}")

        result = await session.execute(stmt, params)

        rows = result.fetchall()
        logger.debug(f"Number of rows fetched: {len(rows)}")

        return [row[0] for row in rows]

    except Exception as e:
        logger.exception(f"Exception during vector search: {e}")
        return []
