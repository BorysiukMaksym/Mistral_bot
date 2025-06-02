import psycopg2
from typing import List
import os
from settings import PGVECTOR_URL

def search_similar_documents(embedding: List[float], top_k: int = 3) -> List[str]:
    conn = psycopg2.connect(PGVECTOR_URL)
    cur = conn.cursor()

    # Формуємо вектор у форматі, який приймає pgvector
    vector_str = f"[{', '.join(str(x) for x in embedding)}]"

    # ⚠️ Без параметрів для вектора — вставляємо напряму
    query = f"""
        SELECT content
        FROM documents
        ORDER BY embedding <-> '{vector_str}'::vector
        LIMIT %s;
    """

    cur.execute(query, (top_k,))

    results = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results
