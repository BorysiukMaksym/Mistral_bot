import asyncio
import logging
import sys
from os import getenv

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


TOKEN = TOKEN_BOT

# All handlers should be attached to the Router (or Dispatcher)

dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with /start command
    """
    # Most event objects have aliases for API methods that can be called in events' context
    # For example if you want to answer to incoming message you can use message.answer(...) alias
    # and the target chat will be passed to :ref:aiogram.methods.send_message.SendMessage
    # method automatically or call API method directly via
    # Bot instance: bot.send_message(chat_id=message.chat.id, ...)
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}!")


@dp.message()
async def echo_handler(message: Message) -> None:
    user_id = message.from_user.id
    content = message.text

    async for session in get_session():
        await save_message(session, user_id, "user", content)

        history = await get_chat_history(session, user_id, limit=10)

        context = [{"role": msg.role, "content": msg.content} for msg in history]
        context.append({"role": "user", "content": content})

        # >>> Отримуємо зовнішній контекст з бази документів (RAG)
        retrieved_chunks = await retrieve_context(content)

        # >>> Додаємо у вигляді системного повідомлення
        if retrieved_chunks:
            combined = "\n---\n".join(retrieved_chunks)
            context.insert(0, {"role": "system", "content": f"Контекст:\n{combined}"})

        response = await main_llm(context)

        await save_message(session, user_id, "assistant", response)
        await message.answer(response)


async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())