import os
import psycopg
import numpy as np
import hashlib
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from settings import PGVECTOR_URL
import fitz  # PyMuPDF
import docx
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# Ініціалізація embedding-моделі
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Повертає float32 вектор у форматі list
def embed(text: str) -> list[float]:
    embedding = model.encode(text, convert_to_numpy=True).astype(np.float32)
    return embedding.tolist()

# Нормалізація тексту перед вставкою
def normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())

# Вставка батчем
def insert_documents_bulk(documents: list[tuple[int, str, list[float]]]):
    if not documents:
        return
    with psycopg.connect(PGVECTOR_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO documents (id, content, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, documents)
        conn.commit()

# Розбиття PDF на чанки ~1000 символів
def extract_chunks_from_pdf(path, max_chars=1000):
    doc = fitz.open(path)
    chunks = []
    current = ""

    for page in doc:
        text = page.get_text()
        for paragraph in text.split("\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(current) + len(paragraph) < max_chars:
                current += paragraph + "\n"
            else:
                chunks.append(normalize_text(current))
                current = paragraph + "\n"
    if current:
        chunks.append(normalize_text(current))
    return chunks

# Обробка DOCX
def extract_text_from_docx(path):
    doc = docx.Document(path)
    full_text = "\n".join([para.text for para in doc.paragraphs])
    return normalize_text(full_text)

# Обробка одного чанку
def process_chunk(chunk: str) -> tuple[int, str, list[float]]:
    vector = embed(chunk)
    doc_id = int(hashlib.sha256(chunk.encode("utf-8")).hexdigest(), 16) % (10 ** 9)
    return (doc_id, chunk, vector)

# Основна логіка завантаження
def load_documents_from_directory(directory: str, batch_size: int = 50, max_workers: int = 4):
    buffer = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if filename.endswith(".pdf"):
                print(f"[PDF] Обробка: {filename}")
                chunks = extract_chunks_from_pdf(path)
            elif filename.endswith(".docx"):
                print(f"[DOCX] Обробка: {filename}")
                content = extract_text_from_docx(path)
                chunks = [content]
            else:
                print(f"[Пропущено] Непідтриманий тип: {filename}")
                continue

            # Паралельна обробка чанків
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

            for future in as_completed(futures):
                try:
                    buffer.append(future.result())
                except Exception as e:
                    print(f"⚠️ Помилка при обробці чанку: {e}")

                if len(buffer) >= batch_size:
                    insert_documents_bulk(buffer)
                    buffer = []

    if buffer:
        insert_documents_bulk(buffer)

if __name__ == "__main__":
    directory = "./documents"
    load_documents_from_directory(directory)
    print("✅ Завантаження завершено.")
