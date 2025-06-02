from rag.embedder import get_embedding
from rag.vector_db import search_similar_documents

async def retrieve_context(user_message: str) -> list[str]:
    embedding = get_embedding(user_message)
    results = search_similar_documents(embedding)
    return results
