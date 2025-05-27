"""evaluate_rag.py
Evaluate a RAG pipeline on a Q&A dataset using GroqEval.
Run with:
    python evaluate_rag.py            # full dataset
    python evaluate_rag.py --sample 5 # first 5 questions (default)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import difflib
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from groqeval.evaluate import GroqEval

# ---------------------------------------------------------------------------
# Project‑specific imports (adjust paths if your project structure differs)
# ---------------------------------------------------------------------------
from src.core.vector_store_manager import get_retriever, load_vector_store
from src.core.rag_chain import create_rag_chain, answer_question
from src.data_processing.data_loader import load_qa_dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise EnvironmentError("Environment variable GROQ_API_KEY is required but not found.")
    return key


def _numeric_score(score_obj: Any) -> float:
    if isinstance(score_obj, dict):
        return float(score_obj.get("score", 0))
    try:
        return float(score_obj)
    except Exception:
        return 0.0


def _breakdown(score_obj: Any) -> Dict[str, Any]:
    return score_obj.get("score_breakdown", {}) if isinstance(score_obj, dict) else {}


def _normalise_context(ctx: str, max_len: int = 8_000) -> str:
    lines = ctx.splitlines()
    bullet_like = sum(1 for l in lines if re.match(r"^\s*-\s?.$", l))
    if bullet_like > len(lines) * 0.6:
        ctx = "".join(re.sub(r"^\s*-\s?", "", l).strip() for l in lines)
    else:
        ctx = "\n".join(re.sub(r"^\s*-\s?", "", l).rstrip() for l in lines)
    ctx = re.sub(r"\s+", " ", ctx).strip()
    return ctx[:max_len]


def _string_similarity(a: str, b: str) -> float:
    """Return simple similarity ratio 0‑1 using difflib."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

# ---------------------------------------------------------------------------
# RAG & evaluation helpers
# ---------------------------------------------------------------------------

async def _build_rag_chain():
    vector_store = load_vector_store()
    retriever = await get_retriever(vector_store)
    return create_rag_chain(retriever)  # type: ignore


async def _evaluate_one(
    rag_chain,
    evaluator: GroqEval,
    question: str,
    ground_truth_answer: str,
    context: str,
    correctness_metric: str | None,
) -> Dict[str, Any]:
    generated_answer = await answer_question(rag_chain, question)
    cleaned_context = _normalise_context(context)

    def _safe(metric: str, **kwargs):
        try:
            return evaluator(metric, **kwargs).score()
        except Exception as exc:
            logger.error("%s metric failed: %s", metric, exc, exc_info=True)
            return 0

    answer_rel = _safe("answer_relevance", prompt=question, output=generated_answer)
    faithfulness = _safe("faithfulness", context=cleaned_context, output=generated_answer)
    context_rel = _safe("context_relevance", prompt=question, context=cleaned_context)

    if correctness_metric:
        correctness = _safe(correctness_metric, reference=ground_truth_answer, output=generated_answer)
        correctness_num = _numeric_score(correctness)
        correctness_break = _breakdown(correctness)
    else:
        # Fallback: string similarity 0‑1 scaled to 0‑10
        sim_ratio = _string_similarity(ground_truth_answer, generated_answer)
        correctness_num = sim_ratio * 10
        correctness_break = {"method": "difflib_ratio", "ratio": sim_ratio}

    return {
        "question": question,
        "ground_truth_answer": ground_truth_answer,
        "generated_answer": generated_answer,
        "cleaned_context": cleaned_context,
        "answer_relevance": _numeric_score(answer_rel),
        "faithfulness": _numeric_score(faithfulness),
        "context_relevance_to_gt": _numeric_score(context_rel),
        "answer_correctness": correctness_num,
        "answer_relevance_breakdown": _breakdown(answer_rel),
        "faithfulness_breakdown": _breakdown(faithfulness),
        "context_relevance_breakdown": _breakdown(context_rel),
        "answer_correctness_breakdown": correctness_break,
    }


async def run_evaluation(sample_size: int | None = 5):
    evaluator = GroqEval(api_key=_load_api_key())
    available = set(evaluator.list_metrics())
    # Choose built‑in correctness‑type metric if present
    if "answer_correctness" in available:
        correctness_metric = "answer_correctness"
    elif "exact_match" in available:
        correctness_metric = "exact_match"
    else:
        correctness_metric = None  # will use fallback
        logger.warning("No built‑in correctness metric found — using string similarity fallback.")

    logger.info("GroqEval metrics available: %s", available)

    rag_chain = await _build_rag_chain()
    logger.info("RAG chain ready.")

    qa_df = load_qa_dataset().rename(columns={"Вопрос": "question", "Правильный ответ": "answer", "Контекст": "context"})
    required = {"question", "answer", "context"}
    if missing := required.difference(qa_df.columns):
        raise KeyError(f"Dataset missing columns: {missing}")

    if sample_size is not None:
        qa_df = qa_df.head(sample_size)
    logger.info("Evaluating %d questions…", len(qa_df))

    rows: List[Dict[str, Any]] = []
    for row in qa_df.itertuples(index=False):
        try:
            rows.append(
                await _evaluate_one(
                    rag_chain,
                    evaluator,
                    row.question,
                    row.answer,
                    row.context,
                    correctness_metric,
                )
            )
            await asyncio.sleep(1)
        except Exception as exc:
            logger.error("Evaluation failed for '%s': %s", row.question, exc, exc_info=True)

    results_df = pd.DataFrame(rows)

    os.makedirs("evaluation/outputs", exist_ok=True)
    out_file = "evaluation/outputs/rag_evaluation_results.csv"
    results_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    logger.info("Results saved ➜ %s", out_file)

    for col in [
        "answer_relevance",
        "faithfulness",
        "context_relevance_to_gt",
        "answer_correctness",
    ]:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    logger.info(
        "Averages — Relevance: %.2f │ Faithfulness: %.2f │ ContextRel: %.2f │ Correctness: %.2f",
        results_df.get("answer_relevance", pd.Series(dtype=float)).mean(),
        results_df.get("faithfulness", pd.Series(dtype=float)).mean(),
        results_df.get("context_relevance_to_gt", pd.Series(dtype=float)).mean(),
        results_df.get("answer_correctness", pd.Series(dtype=float)).mean(),
    )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG with GroqEval.")
    parser.add_argument("--sample", type=int, default=50, metavar="N", help="Evaluate only the first N rows (omit for full dataset)")
    args = parser.parse_args()

    try:
        asyncio.run(run_evaluation(sample_size=args.sample))
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user.")
