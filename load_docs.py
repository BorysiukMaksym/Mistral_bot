import os
import re
import hashlib
from typing import List, Tuple, Union
import numpy as np
import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import docx
from concurrent.futures import ThreadPoolExecutor, as_completed
from settings import PGVECTOR_URL
import textract
import io

load_dotenv()

model = SentenceTransformer("BAAI/bge-base-en-v1.5")


# 🔧 Витяг тексту з байтів файлу за ім'ям
async def extract_text_from_file(file_bytes: io.BytesIO, filename: str) -> str:
    ext = filename.lower()
    try:
        if ext.endswith(".pdf"):
            doc = fitz.open(stream=file_bytes.getvalue(), filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
        elif ext.endswith(".docx"):
            doc = docx.Document(file_bytes)
            text = "\n".join(para.text for para in doc.paragraphs)
        elif ext.endswith(".txt"):
            text = file_bytes.read().decode("utf-8", errors="ignore")
        else:
            # fallback на textract
            text = textract.process(filename, input_encoding='utf-8').decode("utf-8")
    except Exception as e:
        print(f"❌ Error extracting text from {filename}: {e}")
        raise

    return normalize_text(text)


def embed(text: str) -> List[float]:
    formatted = f"<s>{text}</s>"
    return model.encode(formatted, convert_to_numpy=True).astype(np.float32).tolist()


def normalize_text(text: str) -> str:
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def split_text_by_paragraphs(text: str, max_chars: int = 1000) -> List[str]:
    chunks, current = [], ""
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(current) + len(paragraph) + 1 <= max_chars:
            current += paragraph + "\n"
        else:
            chunks.append(normalize_text(current))
            current = paragraph + "\n"
    if current:
        chunks.append(normalize_text(current))
    return chunks


def process_chunk(chunk: str, filename: str) -> Tuple[int, str, List[float], str]:
    doc_id = int(hashlib.sha256(chunk.encode("utf-8")).hexdigest(), 16) % (10 ** 9)
    vector = embed(chunk)
    return (doc_id, chunk, vector, filename)


def insert_documents_bulk(documents: List[Tuple[int, str, List[float], str]]):
    if not documents:
        return
    with psycopg.connect(PGVECTOR_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO documents (id, content, embedding, filename)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, documents)
        conn.commit()


def load_documents_from_directory(directory: str, batch_size: int = 50, max_workers: int = 4):
    buffer = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            ext = filename.lower()

            try:
                with open(path, "rb") as f:
                    file_bytes = io.BytesIO(f.read())
                text = normalize_text(textract.process(path).decode("utf-8"))
                chunks = split_text_by_paragraphs(text)
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
                continue

            chunks = [f"[{filename}] {chunk}" for chunk in chunks]
            futures = [executor.submit(process_chunk, chunk, filename) for chunk in chunks]

            for future in as_completed(futures):
                try:
                    buffer.append(future.result())
                except Exception as e:
                    print(f"⚠️ Error embedding chunk: {e}")

                if len(buffer) >= batch_size:
                    insert_documents_bulk(buffer)
                    buffer = []

    if buffer:
        insert_documents_bulk(buffer)

    print("✅ All documents processed.")


if __name__ == "__main__":
    directory = "./documents"
    load_documents_from_directory(directory)
