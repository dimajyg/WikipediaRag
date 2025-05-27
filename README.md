# Wikipedia RAG System

This project implements a Retrieval Augmented Generation (RAG) system that uses Wikipedia data to answer questions. It features a Telegram bot interface for user interaction and an evaluation framework to assess its performance.

## Features

*   **Wikipedia Data Processing**: Loads and processes Russian Wikipedia articles.
*   **Vector Store**: Uses ChromaDB to store and retrieve relevant text embeddings.
*   **RAG Chain**: Integrates a Groq LLM with the retrieval mechanism using LangChain.
*   **Telegram Bot**: Provides a user-friendly interface via `aiogram` for asking questions.
*   **Asynchronous Operations**: Built with `asyncio` for efficient I/O operations.
*   **Configuration Management**: Centralized settings using `src/core/config.py` and `.env` files.
*   **Evaluation Suite**: Includes scripts to evaluate the RAG system's performance using `groqeval` and a custom Russian Q&A dataset.
*   **Dependency Management**: Uses Poetry for managing project dependencies.
*   **Linting and Formatting**: Pre-configured with `black`, `isort`, `flake8`, `mypy`, and `pylint`.

## Prerequisites

*   Python 3.11+
*   Poetry (for dependency management)

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd WikipediaRag
    ```

2.  **Set Up Environment Variables**:
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your API keys:
        *   `GROQ_API_KEY`: Your API key for Groq (used by the RAG chain and evaluation).
        *   `TELEGRAM_BOT_TOKEN`: Your Telegram Bot token.
        ```env
        GROQ_API_KEY="YOUR_GROQ_API_KEY"
        TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
        ```
    *   Other settings like `CHROMA_DB_PATH`, `CHUNK_SIZE`, etc., can also be configured in the `.env` file if you need to override the defaults in `src/core/config.py`.

3.  **Install Dependencies**:
    You can use the Makefile command or Poetry directly:
    ```bash
    make install
    ```
    Or:
    ```bash
    poetry install
    ```

## Running the Application

1.  **Construct ChromaDB Vector Store (First-Time Setup)**:
    The vector store is automatically created or loaded when you run the main application. If the ChromaDB database specified by `CHROMA_DB_PATH` (default: `./data/chroma_db`) does not exist or is empty, the system will:
    *   Load and process Wikipedia articles from the configured dataset path (default: `ru_rag_test_dataset/files`).
    *   Create embeddings and populate the ChromaDB vector store.
    This process happens automatically when you start the bot for the first time.

2.  **Run the Telegram Bot**:
    To start the Telegram bot and the RAG system, run:
    ```bash
    python src/main.py
    ```
    The bot will initialize the RAG system (loading or creating the vector store as needed) and start polling for messages.

## Evaluation

To evaluate the RAG system's performance:

1.  Ensure all dependencies are installed (`make install` or `poetry install`).
2.  Set up your `.env` file with the `GROQ_API_KEY`.
3.  Make sure the ChromaDB vector store has been created (run `python src/main.py` at least once).
4.  Run the evaluation script from the project root:
    ```bash
    python evaluation/evaluate_rag.py
    ```
Results will be saved in `evaluation/outputs/rag_evaluation_results.csv`.

For more detailed instructions on the evaluation process, refer to <mcfolder name="evaluation" path="/Users/dtikhanovskii/Documents/WikipediaRag/evaluation/"></mcfolder>/<mcfile name="README.md" path="/Users/dtikhanovskii/Documents/WikipediaRag/evaluation/README.md"></mcfile>.

## Project Structure

```
WikipediaRag/
├── .env.example            # Example environment variables
├── .gitignore
├── Makefile                # For common development tasks
├── README.md               # This file
├── evaluation/             # RAG system evaluation scripts and results
│   ├── README.md           # Evaluation specific README
│   ├── evaluate_rag.py     # Main evaluation script
│   └── outputs/            # Directory for evaluation results
├── pyproject.toml          # Poetry dependency and project configuration
├── ru_rag_test_dataset/    # Dataset for RAG and evaluation
│   ├── files/              # Raw text files (Wikipedia articles)
│   └── ru_rag_test_dataset.pkl # Processed Q&A pairs
└── src/                    # Source code
    ├── bot/                # Telegram bot implementation
    │   ├── __init__.py
    │   └── bot.py
    ├── core/               # Core RAG components
    │   ├── __init__.py
    │   ├── config.py       # Configuration management
    │   ├── rag_chain.py    # RAG chain logic
    │   └── vector_store_manager.py # ChromaDB interaction
    ├── data_processing/    # Data loading and preprocessing
    │   ├── __init__.py
    │   └── data_loader.py
    └── main.py             # Main application entry point
```

## Makefile Commands

*   `make help`: Display available commands.
*   `make install`: Install project dependencies using Poetry.
*   `make clean`: Remove the Poetry virtual environment.
*   `make format`: Format code using `isort` and `black`.
*   `make lint`: Run all linters (`isort`, `black`, `flake8`, `mypy`, `pylint`).