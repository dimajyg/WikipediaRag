# RAG System Evaluation

This directory contains scripts and resources for evaluating the Wikipedia RAG system.

## `evaluate_rag.py`

This script uses the `groqeval` framework and the `ru_rag_test_dataset` to assess the performance of the RAG system. It measures metrics like answer relevance, faithfulness to context, and context relevance.

### Prerequisites

1.  **Install Dependencies**: Ensure all project dependencies, including `groqeval`, are installed. You can typically do this by running `make install` or `poetry install` from the project root.
2.  **Environment Variables**: 
    *   Create a `.env` file in the project root directory (you can copy `.env.example` as a template).
    *   Set your `GROQ_API_KEY` in the `.env` file. This is required for `groqeval` to function.
    *   Ensure `DATASET_PKL_PATH` in `src/core/config.py` (or your `.env` file if overridden) points to the correct location of `ru_rag_test_dataset.pkl`.
3.  **Vector Store**: The RAG system needs an existing ChromaDB vector store. Run the main application (`python src/main.py`) at least once to create and populate the vector store if it doesn't exist.
4.  **Dataset**: The `ru_rag_test_dataset.pkl` file must be available at the path specified in the configuration.

### How to Run

1.  **Navigate to Project Root**: Open your terminal and change to the root directory of the `WikipediaRag` project.
2.  **Set PYTHONPATH (if needed)**: If Python has trouble finding the `src` module, you might need to add the project root to your `PYTHONPATH`:
    ```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    ```
    Alternatively, ensure your IDE is configured to recognize the `src` directory as a source root.
3.  **Execute the Script**:
    ```bash
    python evaluation/evaluate_rag.py
    ```

### Output

The script will:

*   Log the evaluation process for each question.
*   Print a summary of evaluation results to the console, including individual scores for each question and average scores for each metric.
*   Save a detailed CSV file named `rag_evaluation_results.csv` in the `evaluation/outputs/` directory. This file will contain:
    *   `question`: The input question.
    *   `ground_truth_answer`: The correct answer from the dataset.
    *   `generated_answer`: The answer produced by the RAG system.
    *   `ground_truth_context`: The context paragraph from the dataset.
    *   `answer_relevance`: Score indicating how relevant the generated answer is to the question.
    *   `faithfulness`: Score indicating how well the generated answer adheres to the provided (ground truth) context.
    *   `context_relevance_to_gt`: Score indicating how relevant the ground truth context was to the question (as assessed by GroqEval).
    *   Breakdown scores for each metric, if provided by `groqeval`.

### Interpreting Results

*   **Answer Relevance**: Higher scores are better. Indicates if the RAG system's answer addresses the question.
*   **Faithfulness**: Higher scores are better. Indicates if the RAG system's answer is factually consistent with the context it was (supposedly) based on. Low scores might indicate hallucination or misinterpretation of context.
*   **Context Relevance (to Ground Truth)**: This metric, as implemented, evaluates the relevance of the *dataset's provided context* to the question. It's more a check on the dataset quality or how GroqEval perceives the dataset's context-question alignment. For evaluating the RAG system's *retrieved* context, the `evaluate_single_question` function would need to be modified to fetch and pass the actual retrieved context to the `faithfulness` and `context_relevance` metrics.

### Customization

*   **Number of Questions**: By default, the script evaluates a small sample (e.g., 5 questions) for a quick test. To evaluate the entire dataset, modify the line `sample_qa_df = qa_df.head(5)` in `evaluate_rag.py` to `sample_qa_df = qa_df`.
*   **Metrics**: You can explore other metrics available in `groqeval` by calling `evaluator.list_metrics()` and modifying the `evaluate_single_question` function to include them.

## `outputs/` Directory

This directory is automatically created to store the output CSV files from the evaluation script.

*   `.gitkeep`: An empty file to ensure the directory is tracked by Git even when empty.