"""Configuration settings for the Wikipedia RAG system."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_API_KEY = os.getenv("HF_API_KEY")

# ChromaDB settings
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "wikipedia_rag")

# RAG settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "16"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "ai-forever/ru-en-RoSBERTa")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "grok")
#LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant") # Groq model
#LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0") # TinyLlama
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "microsoft/phi-1") # Phi-1



# Dataset paths
DATASET_PATH = os.getenv("DATASET_PATH", "./ru_rag_test_dataset/files")
DATASET_PKL_PATH = os.getenv("DATASET_PKL_PATH", "./ru_rag_test_dataset/ru_rag_test_dataset.pkl")


# Ensure essential API keys are set
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set.")

if not HF_API_KEY:
    raise ValueError("HF_API_KEY environment variable not set.")