import logging
import sys
import asyncio
from os import getenv

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

from llm_client import main_llm
from settings import TOKEN_BOT
from database import get_session
from memory import save_message, get_chat_history
from rag.rag_logic import retrieve_context
from database_rag import init_pgvector, ensure_pgvector_extension, get_pgvector_session

from aiogram.types import Document
from pathlib import Path
from tempfile import TemporaryDirectory
import aiofiles
from load_docs import extract_text_from_file

TOKEN = TOKEN_BOT

dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with /start command
    """
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}!")


@dp.message()
async def echo_handler(message: Message) -> None:
    user_id = message.from_user.id

    # Перевірка: чи є документ
    if message.document:
        document: Document = message.document

        # Завантаження файлу
        file = await message.bot.get_file(document.file_id)
        file_path = file.file_path
        downloaded_file = await message.bot.download_file(file_path)

        # Витяг тексту
        text = await extract_text_from_file(downloaded_file, filename=document.file_name)

        # Класифікація
        content_type = await classify_text_type(text)

        # Текст в історії замінюємо назвою файлу
        content = f"[{document.file_name}] (type: {content_type})"
    else:
        content = message.text
        text = content  # для RAG, якщо це не файл

    async for session in get_session():
        await save_message(session, user_id, "user", content)
        history = await get_chat_history(session, user_id, limit=10)

        context = [{"role": msg.role, "content": msg.content} for msg in history]
        context.append({"role": "user", "content": text})  # використовуємо оригінальний текст

        async for rag_session in get_pgvector_session():
            retrieved_chunks = await retrieve_context(text, rag_session)

        if retrieved_chunks:
            rag_context = "\n---\n".join(retrieved_chunks)
            system_prompt = f"""Use the context below if relevant. If there is no relevant information, do not make anything up.
Context:
{rag_context}
"""
        else:
            system_prompt = "If there is no relevant context, do not make anything up."

        context.insert(0, {
            "role": "system",
            "content": system_prompt
        })

        response = await main_llm(context)

        await save_message(session, user_id, "assistant", response)
        await message.answer(response)

@dp.message(lambda message: message.document is not None)
async def handle_file(message: Message):
    document: Document = message.document
    user_id = message.from_user.id

    # Завантажуємо файл тимчасово
    file = await message.bot.get_file(document.file_id)
    temp_dir = TemporaryDirectory()
    file_path = Path(temp_dir.name) / document.file_name
    await message.bot.download_file(file.file_path, destination=file_path)

    # Отримуємо текст
    try:
        extracted_text = extract_text_from_file(file_path)
    except Exception as e:
        await message.answer("Не вдалося прочитати файл.")
        return

    # Формуємо запит до LLM
    file_description_prompt = f"""
Визначи тип і зміст цього документу. Це може бути інструкція, мануал, угода, гайд, звіт тощо. Ось вміст:

{extracted_text[:2000]}  # обмеження, щоб не перевищити prompt
"""

    async for session in get_session():
        await save_message(session, user_id, "user", document.file_name)  # замість тексту
        history = await get_chat_history(session, user_id, limit=10)

        context = [{"role": msg.role, "content": msg.content} for msg in history]
        context.append({"role": "user", "content": file_description_prompt})

        context.insert(0, {
            "role": "system",
            "content": "Analyze the content of the document and determine what type of document it is.."
        })

        response = await main_llm(context)

        await save_message(session, user_id, "assistant", response)
        await message.answer(response)


async def main() -> None:
    logging.info("Bot is starting...")

    await ensure_pgvector_extension()
    await init_pgvector()

    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    await dp.start_polling(bot)

async def classify_text_type(text: str) -> str:
    prompt = f"""
You are a classifier that detects document type.
Text:
{text[:2000]}
Answer in one word:
"""
    result = await main_llm([{"role": "user", "content": prompt}])
    return result.strip().lower()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
