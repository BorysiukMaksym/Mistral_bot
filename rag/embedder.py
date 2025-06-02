from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("intfloat/e5-small-v2")

def get_embedding(text: str) -> list[float]:
    return _model.encode(text).tolist()