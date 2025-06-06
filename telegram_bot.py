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
    content = message.text

    async for session in get_session():
        await save_message(session, user_id, "user", content)
        history = await get_chat_history(session, user_id, limit=10)

        # Base context from history
        context = [{"role": msg.role, "content": msg.content} for msg in history]
        context.append({"role": "user", "content": content})

        async for rag_session in get_pgvector_session():
            retrieved_chunks = await retrieve_context(content, rag_session)

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



async def main() -> None:
    logging.info("Bot is starting...")

    await ensure_pgvector_extension()
    await init_pgvector()

    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
