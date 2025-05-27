"""Main application file for the Wikipedia RAG system."""

import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


import os
from src.data_processing.data_loader import load_and_process_data
from src.core.vector_store_manager import create_vector_store, load_vector_store
from src.core.config import CHROMA_DB_PATH
from src.bot.bot import run_bot, initialize_rag_system as initialize_bot_rag_system


async def main() -> None:
    """Main function to run the RAG system."""
    logger.info("Starting Wikipedia RAG system...")

    # 1. Load and process data
    # This step is crucial for creating the vector store if it doesn't exist.
    # We can make this conditional or a separate script for pre-processing.
    # For now, let's assume we might need to create it.
    try:
        logger.info(f"Checking for existing vector store at {CHROMA_DB_PATH}...")
        if not os.path.exists(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
            logger.info("Vector store not found or empty. Proceeding to create one.")
            split_articles, _ = await load_and_process_data()
            if split_articles:
                create_vector_store(split_articles)
                logger.info("Vector store created successfully.")
            else:
                logger.warning("No articles loaded, vector store not created.")
        else:
            logger.info("Existing vector store found. Skipping creation.")
            # Optionally, load it here to confirm it's valid, though bot init will also do this.
            # load_vector_store() 

    except Exception as e:
        logger.error(f"Error during data processing or vector store creation: {e}", exc_info=True)
        # Decide if the application should proceed if this fails.
        # For now, we'll let it proceed, and the bot might fail to initialize RAG.

    # 2. Start the Telegram bot
    # The RAG system initialization is now handled by aiogram's startup hook in bot.py
    logger.info("Starting Telegram bot polling with aiogram...")
    await run_bot() # This will run the aiogram event loop

    logger.info("Wikipedia RAG system finished (bot polling stopped or error occurred).")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)