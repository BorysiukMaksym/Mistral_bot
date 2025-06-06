from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed_text(text: str) -> list[float]:
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
