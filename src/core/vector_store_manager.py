"""Manages the ChromaDB vector store for the RAG system."""

import logging
from typing import List

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from tqdm import tqdm

from src.core.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL_NAME,
)

logger = logging.getLogger(__name__)


def get_embedding_function():
    """Initializes and returns the embedding function."""
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_vector_store(documents: List[Document]) -> Chroma:
    """Creates and persists a ChromaDB vector store from documents with a progress bar."""
    logger.info(f"Creating vector store at: {CHROMA_DB_PATH}")
    logger.info(f"Using collection name: {CHROMA_COLLECTION_NAME}")
    logger.info(f"Processing {len(documents)} documents.")
    embedding_function = get_embedding_function()

    # Initialize Chroma vector store first
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_function,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    # Add documents in batches with a progress bar
    batch_size = 100  # Adjust batch size as needed
    for i in tqdm(range(0, len(documents), batch_size), desc="Adding documents to ChromaDB"):
        batch = documents[i:i + batch_size]
        vector_store.add_documents(documents=batch) # Embeddings are generated here if not pre-computed

    # Chroma with persist_directory usually persists automatically on add.
    # If explicit persistence is needed: vector_store.persist()
    logger.info("Vector store created and persisted.")
    return vector_store


def load_vector_store() -> Chroma:
    """Loads an existing ChromaDB vector store."""
    logger.info(f"Loading vector store from: {CHROMA_DB_PATH}")
    logger.info(f"Using collection name: {CHROMA_COLLECTION_NAME}")
    embedding_function = get_embedding_function()

    try:
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_function,
            collection_name=CHROMA_COLLECTION_NAME,
        )
        # Check if the collection exists and has documents
        # This is a simple check; Chroma might offer more direct ways
        if vector_store._collection.count() == 0: # type: ignore
            logger.warning(
                f"Vector store loaded, but collection '{CHROMA_COLLECTION_NAME}' is empty."
            )
        else:
            logger.info(
                f"Vector store loaded successfully with {vector_store._collection.count()} items." # type: ignore
            )
        return vector_store
    except Exception as e:
        # This exception handling might need to be more specific based on Chroma's behavior
        # if the database or collection doesn't exist.
        logger.error(f"Failed to load vector store: {e}")
        logger.error(
            "This might happen if the database does not exist or the collection name is incorrect."
        )
        logger.error("Consider running the data ingestion process first.")
        raise


async def get_retriever(vector_store: Chroma, k: int = 25):
    """Gets a retriever from the vector store."""
    logger.info(f"Creating retriever with k={k}")
    return vector_store.as_retriever(search_kwargs={"k": k})


# Example usage (for testing or initial setup)
if __name__ == "__main__":
    import asyncio
    from src.data_processing.data_loader import load_and_process_data
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    load_dotenv() # Ensure .env is loaded for config

    async def _test_vector_store():
        logger.info("Testing vector store creation and loading...")
        # 1. Load and process data
        split_articles, _ = await load_and_process_data()

        if not split_articles:
            logger.error("No articles loaded, cannot test vector store.")
            return

        # 2. Create vector store
        try:
            created_store = create_vector_store(split_articles)
            logger.info(f"Created store size: {created_store._collection.count()}") # type: ignore
        except Exception as e:
            logger.error(f"Error during vector store creation test: {e}", exc_info=True)
            return

        # 3. Load vector store
        try:
            loaded_store = load_vector_store()
            logger.info(f"Loaded store size: {loaded_store._collection.count()}") # type: ignore

            # 4. Test retriever
            retriever = await get_retriever(loaded_store)
            sample_query = "Что такое цунами?"
            results = await retriever.aget_relevant_documents(sample_query)
            logger.info(f"Retrieved {len(results)} documents for query: '{sample_query}'")
            for i, doc in enumerate(results):
                logger.info(f"Result {i+1}: {doc.page_content[:100]}... (Source: {doc.metadata.get('source')})")

        except Exception as e:
            logger.error(f"Error during vector store loading/retrieval test: {e}", exc_info=True)

    asyncio.run(_test_vector_store())