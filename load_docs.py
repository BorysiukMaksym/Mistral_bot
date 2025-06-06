import os
import re
import hashlib
from typing import List, Tuple
import numpy as np
import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import docx
from concurrent.futures import ThreadPoolExecutor, as_completed
from settings import PGVECTOR_URL

load_dotenv()

# Initialize the embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Embed text into a float vector
def embed(text: str) -> List[float]:
    formatted = f"<s>{text}</s>"
    return model.encode(formatted, convert_to_numpy=True).astype(np.float32).tolist()

# Normalize and clean text
def normalize_text(text: str) -> str:
    text = re.sub(r'\n{2,}', '\n', text)  # Collapse multiple newlines
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Remove page numbers
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

# Split text into chunks by paragraphs
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

# Extract text from PDF
def extract_chunks_from_pdf(path: str, max_chars: int = 1000) -> List[str]:
    doc = fitz.open(path)
    all_text = ""
    for page in doc:
        text = page.get_text()
        all_text += normalize_text(text) + "\n"
    return split_text_by_paragraphs(all_text, max_chars=max_chars)

# Extract text from DOCX
def extract_chunks_from_docx(path: str, max_chars: int = 1000) -> List[str]:
    doc = docx.Document(path)
    text = "\n".join(para.text for para in doc.paragraphs)
    return split_text_by_paragraphs(normalize_text(text), max_chars=max_chars)

# Extract from TXT
def extract_chunks_from_txt(path: str, max_chars: int = 1000) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return split_text_by_paragraphs(normalize_text(text), max_chars=max_chars)

# Generate document ID and embedding
def process_chunk(chunk: str, filename: str) -> Tuple[int, str, List[float], str]:
    doc_id = int(hashlib.sha256(chunk.encode("utf-8")).hexdigest(), 16) % (10 ** 9)
    vector = embed(chunk)
    return (doc_id, chunk, vector, filename)

# Insert chunks in bulk into DB
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

# Main loading function
def load_documents_from_directory(directory: str, batch_size: int = 50, max_workers: int = 4):
    buffer = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            ext = filename.lower()

            try:
                if ext.endswith(".pdf"):
                    print(f"[PDF] Processing: {filename}")
                    chunks = extract_chunks_from_pdf(path)
                elif ext.endswith(".docx"):
                    print(f"[DOCX] Processing: {filename}")
                    chunks = extract_chunks_from_docx(path)
                elif ext.endswith(".txt"):
                    print(f"[TXT] Processing: {filename}")
                    chunks = extract_chunks_from_txt(path)
                else:
                    print(f"[SKIP] Unsupported file: {filename}")
                    continue
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
                continue

            # Optional: Add title metadata to each chunk (from filename)
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
