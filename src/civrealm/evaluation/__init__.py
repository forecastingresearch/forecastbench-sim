"""Evaluation module for CivBench LLM forecasting benchmarks."""

from .sampling import (
    stratified_sample,
    stratified_sample_by_horizon_template,
    load_all_questions,
)
from .rate_limiter import ProviderRateLimiter
from .parallel_evaluator import (
    PredictionResult,
    query_model_async,
    evaluate_question,
    run_evaluation,
)

__all__ = [
    "stratified_sample",
    "stratified_sample_by_horizon_template",
    "load_all_questions",
    "ProviderRateLimiter",
    "PredictionResult",
    "query_model_async",
    "evaluate_question",
    "run_evaluation",
]
