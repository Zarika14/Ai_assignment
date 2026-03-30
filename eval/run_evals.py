"""
eval/run_evals.py — Part 5: RAG Evaluation Script

Metrics reported:
- Retrieval hit rate  (did the correct source document appear in top-3?)
- LLM-as-judge score (1–5) per query, and average

Output: structured JSON logs via structlog + eval_results.json saved to disk.
No print statements — all output is structured JSON as required.
"""

import os
import sys
import json
import time
import requests
import re
from pathlib import Path

import structlog

# Add project root so we can import rag/
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Structured JSON logging (matches model_server and agent_server format)
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# MODEL SERVER URL
# ---------------------------------------------------------------------------
MODEL_SERVER_URL = os.environ.get("EVAL_MODEL_SERVER_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# 10 Evaluation Query-Answer Pairs (covers all 3 policy documents)
# ---------------------------------------------------------------------------
EVAL_DATA = [
    {
        "query": "What is the collision deductible?",
        "expected_source": "policy_auto_comprehensive.txt",
        "reference": "The collision deductible is $500.",
    },
    {
        "query": "Does the auto policy cover intentional damage?",
        "expected_source": "policy_auto_comprehensive.txt",
        "reference": "No, the policy strictly excludes intentional damage or destruction.",
    },
    {
        "query": "When are renewal documents mailed for the auto policy?",
        "expected_source": "policy_auto_comprehensive.txt",
        "reference": "Renewal documents are mailed 45 days before expiration.",
    },
    {
        "query": "What is the individual annual deductible for the Bronze plan?",
        "expected_source": "policy_health_bronze.txt",
        "reference": "The individual annual deductible is $3,000.",
    },
    {
        "query": "Is prior authorization required for an MRI?",
        "expected_source": "policy_health_bronze.txt",
        "reference": "Yes, pre-authorization is required for MRI and advanced imaging.",
    },
    {
        "query": "How much is the copay for a Tier 1 generic drug?",
        "expected_source": "policy_health_bronze.txt",
        "reference": "The copay for a Tier 1 generic drug is $15 per 30-day supply.",
    },
    {
        "query": "What is the limit for personal property coverage?",
        "expected_source": "policy_home_standard.txt",
        "reference": "The limit for personal property coverage is $150,000, which is 70% of dwelling coverage.",
    },
    {
        "query": "Does the standard homeowners policy cover flood damage?",
        "expected_source": "policy_home_standard.txt",
        "reference": "No, it does not cover flood damage; flood insurance must be purchased separately.",
    },
    {
        "query": "What is the additional living expenses coverage limit?",
        "expected_source": "policy_home_standard.txt",
        "reference": "The additional living expenses coverage limit is 20% of dwelling coverage, or $70,000.",
    },
    {
        "query": "How many quotes from licensed contractors are recommended for a home claim?",
        "expected_source": "policy_home_standard.txt",
        "reference": "A minimum of 2 quotes from licensed contractors is recommended.",
    },
]


def llm_as_judge(query: str, reference: str, answer: str) -> int:
    """
    Call the model server to score the generated answer 1–5.

    Score scale:
      1 = Completely incorrect / irrelevant
      5 = Perfectly matches reference answer

    Returns 1 as a safe default if the LLM call fails.
    """
    system_prompt = (
        "You are an expert evaluator. "
        "Score the given Answer based on how well it matches the Reference and answers the Query. "
        "Output ONLY a single integer from 1 to 5. "
        "1=Completely incorrect/irrelevant, 5=Perfectly matches reference."
    )
    user_message = f"Query: {query}\nReference: {reference}\nAnswer: {answer}"

    try:
        t0 = time.time()
        response = requests.post(
            f"{MODEL_SERVER_URL}/chat",
            json={"message": user_message, "system_prompt": system_prompt},
            timeout=300,
        )
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        latency_ms = round((time.time() - t0) * 1000, 2)

        match = re.search(r"[1-5]", result)
        score = int(match.group()) if match else 1

        logger.info(
            "llm_judge_scored",
            query=query[:60],
            score=score,
            judge_latency_ms=latency_ms,
        )
        return score

    except Exception as exc:
        logger.error("llm_judge_failed", error=str(exc), query=query[:60])
        return 1


def run_evals() -> dict:
    """
    Run the full evaluation suite.

    Steps per query:
      1. Retrieve top-k chunks and check if expected source is in results (hit rate)
      2. Generate a full RAG answer via pipeline.answer_with_sources()
      3. Score the answer using LLM-as-judge (1–5)

    All progress logged as structured JSON. Final report saved to eval_results.json.
    """
    from rag.pipeline import RAGPipeline

    logger.info("eval_started", total_queries=len(EVAL_DATA), model_server=MODEL_SERVER_URL)

    try:
        pipeline = RAGPipeline(rerank=True)
        pipeline.load_index()
        logger.info("rag_pipeline_ready", rerank=True)
    except Exception as exc:
        logger.error(
            "rag_pipeline_init_failed",
            error=str(exc),
            hint="Run: python rag/index_documents.py",
        )
        sys.exit(1)

    results = []
    hits = 0
    total_score = 0

    for i, item in enumerate(EVAL_DATA, 1):
        query = item["query"]
        expected_source = item["expected_source"]
        reference = item["reference"]

        logger.info(
            "eval_query_start",
            query_num=i,
            total=len(EVAL_DATA),
            query=query,
            expected_source=expected_source,
        )

        # 1. Retrieval hit rate
        t_ret = time.time()
        retrieved_chunks = pipeline.retrieve(query, top_k=3)
        ret_latency_ms = round((time.time() - t_ret) * 1000, 2)

        retrieved_sources = [chunk["source_file"] for chunk in retrieved_chunks]
        is_hit = expected_source in retrieved_sources
        if is_hit:
            hits += 1

        logger.info(
            "retrieval_result",
            query=query,
            expected_source=expected_source,
            retrieved_sources=retrieved_sources,
            is_hit=is_hit,
            retrieval_latency_ms=ret_latency_ms,
        )

        # 2. Answer generation
        t_ans = time.time()
        ans_result = pipeline.answer_with_sources(query, top_k=3)
        generated_answer = ans_result["answer"]
        ans_latency_ms = round((time.time() - t_ans) * 1000, 2)

        logger.info(
            "answer_generated",
            query=query,
            answer_length=len(generated_answer),
            generation_latency_ms=ans_latency_ms,
        )

        # 3. LLM-as-judge scoring
        score = llm_as_judge(query, reference, generated_answer)
        total_score += score

        results.append({
            "query": query,
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "is_hit": is_hit,
            "reference_answer": reference,
            "generated_answer": generated_answer,
            "score": score,
            "retrieval_latency_ms": ret_latency_ms,
            "generation_latency_ms": ans_latency_ms,
        })

    # Summary
    hit_rate = hits / len(EVAL_DATA)
    avg_score = round(total_score / len(EVAL_DATA), 2)

    final_report = {
        "summary": {
            "total_queries": len(EVAL_DATA),
            "hit_rate": hit_rate,
            "average_score": avg_score,
        },
        "details": results,
    }

    # Save to disk
    out_path = Path(__file__).parent.parent / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    # Log final summary as structured JSON
    logger.info(
        "eval_complete",
        total_queries=len(EVAL_DATA),
        hit_rate=hit_rate,
        average_score=avg_score,
        results_saved_to=str(out_path),
    )

    return final_report


if __name__ == "__main__":
    run_evals()
