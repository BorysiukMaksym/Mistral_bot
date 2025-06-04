from rag.embedder import get_embedding
from rag.vector_db import search_similar_documents
from database_rag import get_pgvector_session

async def retrieve_context(user_message: str, session) -> list[str]:
    embedding = get_embedding(user_message)
    print(f"[DEBUG] Embedding type: {type(embedding)}")
    print(f"[DEBUG] First 3 values: {embedding[:3]}")
    results = await search_similar_documents(session, embedding)
    return results

