from rag.embedder import embed_text
from rag.vector_db import search_similar_documents
from database_rag import get_pgvector_session
import logging
logger = logging.getLogger(__name__)

async def retrieve_context(user_message: str, session) -> list[str]:
    embedding = embed_text(user_message)
    results = await search_similar_documents(session, embedding)
    logger.info(f"RAG: Retrieved {len(results)} chunks for message: {user_message[:50]}...")
    return results

