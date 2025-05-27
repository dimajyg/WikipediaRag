"""Loads and processes Wikipedia articles from the ru_rag_test_dataset."""

import logging
import os
from typing import List, Tuple

import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.core.config import CHUNK_OVERLAP, CHUNK_SIZE, DATASET_PATH, DATASET_PKL_PATH

logger = logging.getLogger(__name__)


def load_wikipedia_articles() -> List[Document]:
    """Loads Wikipedia articles from the specified directory."""
    documents: List[Document] = []
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset path not found: {DATASET_PATH}")
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

    logger.info(f"Loading documents from: {DATASET_PATH}")
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATASET_PATH, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Use filename (page_id) as part of the metadata
                doc = Document(page_content=content, metadata={"source": filename})
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error loading file {filename}: {e}")
    logger.info(f"Loaded {len(documents)} documents.")
    return documents


def load_qa_dataset() -> pd.DataFrame:
    """Loads the Q&A dataset from the Pickle file."""
    if not os.path.exists(DATASET_PKL_PATH):
        logger.error(f"Q&A dataset pickle file not found: {DATASET_PKL_PATH}")
        raise FileNotFoundError(
            f"Q&A dataset pickle file not found: {DATASET_PKL_PATH}"
        )
    logger.info(f"Loading Q&A dataset from: {DATASET_PKL_PATH}")
    try:
        df = pd.read_pickle(DATASET_PKL_PATH)
        logger.info(f"Loaded Q&A dataset with {len(df)} records.")
        # Expected columns: 'Вопрос', 'Правильный ответ', 'Контекст', 'Название файла'
        # Rename columns for consistency if needed, e.g.:
        # df = df.rename(columns={'Вопрос': 'question', 'Правильный ответ': 'answer', 
        #                         'Контекст': 'context', 'Название файла': 'source_file'})
        return df
    except Exception as e:
        logger.error(f"Error loading Q&A dataset pickle: {e}")
        raise


def split_documents(documents: List[Document]) -> List[Document]:
    """Splits documents into smaller chunks."""
    logger.info(
        f"Splitting {len(documents)} documents into chunks (size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})"
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(split_docs)} chunks.")
    return split_docs


async def load_and_process_data() -> Tuple[List[Document], pd.DataFrame]:
    """Loads all data and processes it for RAG."""
    logger.info("Starting data loading and processing...")
    articles = load_wikipedia_articles()
    split_articles = split_documents(articles)
    qa_df = load_qa_dataset()
    logger.info("Data loading and processing complete.")
    return split_articles, qa_df


if __name__ == "__main__":
    # Example usage:
    logging.basicConfig(level=logging.INFO)
    # Ensure .env file is loaded for config
    from dotenv import load_dotenv

    load_dotenv()

    processed_articles, qa_data = asyncio.run(load_and_process_data())
    logger.info(f"Processed {len(processed_articles)} article chunks.")
    logger.info(f"QA DataFrame shape: {qa_data.shape}")
    if not processed_articles:
        logger.warning("No articles were processed.")
    else:
        logger.info(f"First chunk example: {processed_articles[0].page_content[:200]}...")
        logger.info(f"First chunk metadata: {processed_articles[0].metadata}")

    if qa_data.empty:
        logger.warning("QA data is empty.")
    else:
        logger.info(f"QA data head:\n{qa_data.head()}")