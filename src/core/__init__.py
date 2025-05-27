"""Core components for the RAG system, including configuration, vector store, and RAG chain logic."""

from .config import (
    GROQ_API_KEY,
    TELEGRAM_BOT_TOKEN,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    DATASET_PATH,
    DATASET_PKL_PATH,
)
from .vector_store_manager import (
    create_vector_store,
    load_vector_store,
    get_retriever,
    get_embedding_function,
)
from .rag_chain import create_rag_chain, answer_question, get_llm

__all__ = [
    # Config
    "GROQ_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "CHROMA_DB_PATH",
    "CHROMA_COLLECTION_NAME",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "EMBEDDING_MODEL_NAME",
    "LLM_MODEL_NAME",
    "DATASET_PATH",
    "DATASET_PKL_PATH",
    # Vector Store
    "create_vector_store",
    "load_vector_store",
    "get_retriever",
    "get_embedding_function",
    # RAG Chain
    "create_rag_chain",
    "answer_question",
    "get_llm",
]