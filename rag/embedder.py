from sentence_transformers import SentenceTransformer
import numpy as np

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str) -> list[float]:
    embedding = _model.encode(text, convert_to_numpy=True).astype(np.float32)
    return embedding.tolist()
