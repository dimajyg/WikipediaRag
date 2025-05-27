"""Data processing module for loading and preparing Wikipedia data."""

from .data_loader import load_and_process_data, load_qa_dataset, load_wikipedia_articles, split_documents

__all__ = [
    "load_and_process_data",
    "load_qa_dataset",
    "load_wikipedia_articles",
    "split_documents",
]