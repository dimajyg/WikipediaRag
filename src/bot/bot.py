"""Telegram bot interface for the Wikipedia RAG system using aiogram."""

import asyncio
import logging

from aiogram import Bot, Dispatcher, types
from aiogram.client.bot import DefaultBotProperties
from aiogram.filters import CommandStart, Command
from aiogram.enums import ParseMode, ChatAction
from aiogram.types import Message

from src.core.config import TELEGRAM_BOT_TOKEN
from src.core.rag_chain import create_rag_chain, answer_question
from src.core.vector_store_manager import get_retriever, load_vector_store

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global RAG chain (initialize once)
RAG_CHAIN = None

# Bot and Dispatcher instances
bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()


async def initialize_rag_system():
    """Initializes the RAG chain components."""
    global RAG_CHAIN
    if RAG_CHAIN is None:
        logger.info("Initializing RAG system for the bot...")
        try:
            vector_store = load_vector_store()
            retriever = await get_retriever(vector_store)
            RAG_CHAIN = create_rag_chain(retriever)  # type: ignore
            logger.info("RAG system initialized successfully for the bot.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
            RAG_CHAIN = None


@dp.message(CommandStart())
async def start_command_handler(message: Message) -> None:
    """Sends a welcome message when the /start command is issued."""
    user = message.from_user
    await message.answer(
        f"Hi {user.mention_html()}! I'm a Wikipedia RAG bot. Ask me anything!"
    )


@dp.message(Command("help"))
async def help_command_handler(message: Message) -> None:
    """Sends a help message when the /help command is issued."""
    await message.answer(
        "I can answer questions based on Wikipedia articles. Just send me your question!"
    )


@dp.message()
async def message_handler(message: Message) -> None:
    """Handles user messages by querying the RAG chain."""
    global RAG_CHAIN
    if RAG_CHAIN is None:
        await message.answer(
            "Sorry, the RAG system is not currently available. Please try again later."
        )
        logger.info("Attempting to re-initialize RAG system on message...")
        await initialize_rag_system()
        if RAG_CHAIN is None:  # Still None after retry
            return

    question = message.text
    if not question: # Should not happen for text messages, but good practice
        return
        
    logger.info(f"Received question from {message.from_user.id if message.from_user else 'unknown user'}: {question}")

    # Typing action
    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    response = await answer_question(RAG_CHAIN, question)
    await message.answer(response)


async def on_startup(bot: Bot):
    """Actions to perform on bot startup, like initializing RAG."""
    logger.info("Bot starting up...")
    await initialize_rag_system()
    if RAG_CHAIN:
        logger.info("RAG system pre-initialized successfully during startup.")
    else:
        logger.warning("RAG system failed to pre-initialize during startup. Bot will attempt lazy init on first message.")

async def run_bot() -> None:
    """Starts the Telegram bot using aiogram."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found. Bot cannot start.")
        return

    logger.info("Starting Telegram bot with aiogram...")
    # Register startup hook
    dp.startup.register(on_startup)
    
    # Start polling
    # The `await dp.start_polling(bot)` is the modern way for aiogram 3.x
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Error during bot polling: {e}", exc_info=True)
    finally:
        await bot.session.close()
        logger.info("Bot polling stopped and session closed.")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()  # Load .env for TELEGRAM_BOT_TOKEN and GROQ_API_KEY

    # Aiogram uses asyncio.run() directly for its main loop
    try:
        asyncio.run(run_bot())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped manually.")
    except Exception as e:
        logger.critical(f"Critical error running the bot: {e}", exc_info=True)