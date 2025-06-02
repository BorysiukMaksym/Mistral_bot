import os
import psycopg2
import fitz  # PyMuPDF
import docx
import numpy as np
import hashlib
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Налаштування бази даних pgvector (з докера, порт 5433)
PGVECTOR_URL = os.getenv("PGVECTOR_URL", "dbname=llm_bot user=user password=WarGar231 host=localhost port=5433")

# Ініціалізація embedding-моделі
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(text):
    embedding = model.encode(text)
    return embedding.astype(np.float32)  # pgvector потребує float32


def insert_document(text: str):
    vector = embed(text)
    # Унікальний id по хешу тексту
    doc_id = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (10 ** 9)

    conn = psycopg2.connect(PGVECTOR_URL)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO documents (id, user_id, content, embedding)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """, (doc_id, 0, text, vector.tolist()))
    conn.commit()
    cur.close()
    conn.close()


def extract_chunks_from_pdf(path, max_chars=1000):
    doc = fitz.open(path)
    chunks = []
    current = ""

    for page in doc:
        text = page.get_text()
        for paragraph in text.split("\n"):
            if len(current) + len(paragraph) < max_chars:
                current += paragraph + "\n"
            else:
                chunks.append(current.strip())
                current = paragraph + "\n"
    if current:
        chunks.append(current.strip())
    return chunks


def extract_text_from_docx(path):
    doc = docx.Document(path)
    full_text = "\n".join([para.text for para in doc.paragraphs])
    return full_text.strip()


def load_documents_from_directory(directory):
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            print(f"[PDF] Обробка: {filename}")
            chunks = extract_chunks_from_pdf(path)
            print(f"  Знайдено {len(chunks)} частин...")
            for chunk in chunks:
                insert_document(chunk)
        elif filename.endswith(".docx"):
            print(f"[DOCX] Обробка: {filename}")
            content = extract_text_from_docx(path)
            insert_document(content)
        else:
            print(f"[Пропущено] Непідтриманий тип: {filename}")


if __name__ == "__main__":
    directory = "./documents"  # ← створити папку й додати файли сюди
    load_documents_from_directory(directory)
    print("✅ Завантаження завершено.")
