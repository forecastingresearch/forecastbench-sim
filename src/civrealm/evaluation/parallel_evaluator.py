"""Parallel LLM evaluation engine for CivBench."""

import asyncio
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .rate_limiter import ProviderRateLimiter

logger = logging.getLogger("civbench_eval")

# Create a large thread pool to handle many concurrent model calls
# Default is min(32, cpu_count+4) which can exhaust with 18 models timing out
_thread_pool = ThreadPoolExecutor(max_workers=100, thread_name_prefix="llm_query")


@dataclass
class PredictionResult:
    """Result of a single model prediction."""
    model_id: str
    probability: float | None
    latency_ms: float
    error: str | None
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding raw_response for storage."""
        return {
            "probability": self.probability,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


def parse_probability(response: str) -> float | None:
    """
    Parse a probability value from model response.

    Args:
        response: Raw model response text

    Returns:
        Parsed probability clamped to [0, 1], or None if parsing fails
    """
    # Try to find a decimal number in the response
    match = re.search(r'(\d+\.?\d*)', response.strip())
    if match:
        value = float(match.group(1))
        # Clamp to [0, 1]
        return max(0.0, min(1.0, value))
    return None


def _extract_probabilities_from_text(text: str, num_questions: int) -> list[float | None]:
    """
    Extract probability values from text containing numeric lines.

    Args:
        text: Text containing probability values (one per line or comma-separated)
        num_questions: Expected number of values

    Returns:
        List of probabilities clamped to [0, 1], padded with None if needed.
    """
    probabilities: list[float | None] = []

    # Try line-by-line first
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Extract any decimal number from the line
        match = re.search(r'(\d*\.\d+|\d+\.?\d*)', line)
        if match:
            try:
                value = float(match.group(1))
                probabilities.append(max(0.0, min(1.0, value)))
            except ValueError:
                continue
        if len(probabilities) >= num_questions:
            break

    # Pad with None if needed
    while len(probabilities) < num_questions:
        probabilities.append(None)

    return probabilities[:num_questions]


def parse_batch_probabilities(response: str, num_questions: int) -> list[float | None]:
    """
    Parse multiple probability values from a batched model response.

    Handles various response formats:
    - Delimited block: <<<PROBABILITIES>>>\\n0.7\\n0.8\\n<<<END>>>
    - Numbered lines: "1. 0.7\\n2. 0.8\\n3. 0.6"
    - Plain lines: "0.7\\n0.8\\n0.6"
    - JSON array: [0.7, 0.8, 0.6]
    - Comma-separated: "0.7, 0.8, 0.6"

    Args:
        response: Raw model response text
        num_questions: Expected number of probability values

    Returns:
        List of probabilities (clamped to [0, 1]) or None for unparseable values.
        Length matches num_questions, padding with None if needed.
    """
    # Try delimiter extraction first (most reliable)
    delimiter_match = re.search(
        r'<<<PROBABILIT(?:Y|IES)>>>(.*?)<<<END>>>',
        response,
        re.DOTALL | re.IGNORECASE
    )
    if delimiter_match:
        delimited_content = delimiter_match.group(1).strip()
        probabilities = _extract_probabilities_from_text(delimited_content, num_questions)
        # Only use if we got at least one valid probability
        if any(p is not None for p in probabilities):
            return probabilities

    probabilities: list[float | None] = []

    # Try JSON array first
    try:
        import json as json_module
        # Look for array pattern
        match = re.search(r'\[[\d.,\s]+\]', response)
        if match:
            arr = json_module.loads(match.group())
            for val in arr:
                if isinstance(val, (int, float)):
                    probabilities.append(max(0.0, min(1.0, float(val))))
                else:
                    probabilities.append(None)
            # Pad or truncate to expected length
            while len(probabilities) < num_questions:
                probabilities.append(None)
            return probabilities[:num_questions]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try line-by-line parsing (handles numbered and plain formats)
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for numbered format: "1. 0.7", "1: 0.7", "1) 0.7"
        # Require whitespace after separator to avoid matching "0.7" as numbered
        numbered_match = re.match(r'^(\d+)[\.\:\)]\s+(\d*\.?\d+)', line)
        if numbered_match:
            try:
                value = float(numbered_match.group(2))
                probabilities.append(max(0.0, min(1.0, value)))
                if len(probabilities) >= num_questions:
                    break
                continue
            except ValueError:
                pass

        # Also check format without required space: "1.0.7" shouldn't match but "1:0.7" might
        # So check for colon/paren format that doesn't need space
        numbered_match2 = re.match(r'^(\d+)[\:\)](\d*\.?\d+)', line)
        if numbered_match2:
            try:
                value = float(numbered_match2.group(2))
                probabilities.append(max(0.0, min(1.0, value)))
                if len(probabilities) >= num_questions:
                    break
                continue
            except ValueError:
                pass

        # Fall back to finding any decimal number in the line (probabilities are typically 0.X)
        match = re.search(r'(\d*\.\d+|\d+\.\d*)', line)
        if match:
            try:
                value = float(match.group(1))
                probabilities.append(max(0.0, min(1.0, value)))
            except ValueError:
                probabilities.append(None)
        elif re.match(r'^\d+$', line.strip()):
            # Handle integer-only responses (e.g., "0" or "1")
            try:
                value = float(line.strip())
                probabilities.append(max(0.0, min(1.0, value)))
            except ValueError:
                pass

        if len(probabilities) >= num_questions:
            break

    # Try comma-separated if we didn't get enough
    if len(probabilities) < num_questions:
        probabilities = []
        parts = re.split(r'[,\s]+', response.strip())
        for part in parts:
            try:
                value = float(part.strip())
                probabilities.append(max(0.0, min(1.0, value)))
            except ValueError:
                continue
            if len(probabilities) >= num_questions:
                break

    # Pad with None if needed
    while len(probabilities) < num_questions:
        probabilities.append(None)

    return probabilities[:num_questions]


def build_batch_prompt(questions: list[dict], world_report: str) -> str:
    """
    Build a prompt with one or more questions sharing one world report.

    This batched approach reduces token usage by ~80-90% compared to
    sending each question in a separate prompt with the full world report.

    Args:
        questions: List of question dicts (from same game) with 'question_text' key.
                   Can be a single question or multiple questions.
        world_report: JSON string of world report data

    Returns:
        Formatted prompt string with numbered questions
    """
    num_questions = len(questions)
    questions_text = "\n".join(
        f"{i+1}. {q['question_text']}"
        for i, q in enumerate(questions)
    )

    # Use singular wording for single question, plural for multiple
    if num_questions == 1:
        question_section = "## Question"
        instruction = """You may analyze the data, but you MUST end your response with your probability in this exact format:
<<<PROBABILITY>>>
0.65
<<<END>>>

Replace 0.65 with your actual probability estimate between 0.0 and 1.0."""
    else:
        question_section = "## Questions"
        instruction = f"""You may analyze the data, but you MUST end your response with probabilities in this exact format:
<<<PROBABILITIES>>>
0.65
0.42
0.78
<<<END>>>

List one probability per line for all {num_questions} questions, in order. Replace the example values with your actual estimates between 0.0 and 1.0."""

    return f"""You are an expert superforecaster, familiar with the work of Tetlock and others. You are analyzing a FreeCiv game simulation. Make predictions based on the world report below.

You MUST give a probability estimate between 0 and 1 for each question UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, estimate 0.5, but return a number between 0 and 1.

## World Report
{world_report}

{question_section}
{questions_text}

{instruction}"""


def load_world_report(data_dir: Path, game_id: str) -> str:
    """
    Load world report as TXT for LLM context.

    Args:
        data_dir: Base directory containing game folders
        game_id: Game identifier (e.g., "s100")

    Returns:
        TXT world report content, or empty string if not found.
    """
    # Look for TXT report (preferred)
    txt_path = data_dir / game_id / "world_report" / "turn_050_report.txt"

    if txt_path.exists():
        with open(txt_path) as f:
            return f.read()

    return ""


async def query_model_async(
    model: Any,
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
    base_backoff: float = 2.0,
    timeout: float | None = None,
) -> PredictionResult:
    """
    Query a model asynchronously with rate limiting and retries.

    Uses asyncio.to_thread() to wrap the synchronous model.get_response() call,
    allowing it to run without blocking the event loop.

    Args:
        model: Model object with get_response() method
        prompt: The prompt to send
        semaphore: Rate limiting semaphore for this provider
        max_retries: Maximum retry attempts on failure
        base_backoff: Base seconds for exponential backoff
        timeout: Optional timeout in seconds per model query

    Returns:
        PredictionResult with probability, latency, and any errors
    """
    logger.debug(f"    [{model.id}] Waiting for semaphore...")
    async with semaphore:
        logger.info(f"    [{model.id}] Querying...")
        start = time.monotonic()
        last_error = None

        for attempt in range(max_retries):
            try:
                # Wrap sync call in thread to not block event loop
                # Use custom thread pool to avoid exhaustion with many timeouts
                loop = asyncio.get_running_loop()

                # Log thread pool status periodically
                active = threading.active_count()
                if active > 20:
                    logger.debug(f"    [{model.id}] Active threads: {active}")

                # Capture variables for thread (avoid closure issues)
                def make_call(m=model, p=prompt):
                    return m.get_response(p, temperature=0.0, max_tokens=50)

                api_call = loop.run_in_executor(_thread_pool, make_call)

                # Apply timeout if specified
                if timeout:
                    response = await asyncio.wait_for(api_call, timeout=timeout)
                else:
                    response = await api_call

                latency = (time.monotonic() - start) * 1000
                prob = parse_probability(response)

                if prob is not None:
                    logger.info(f"    [{model.id}] -> {prob:.2f} ({latency:.0f}ms)")
                    return PredictionResult(
                        model_id=model.id,
                        probability=prob,
                        latency_ms=latency,
                        error=None,
                        raw_response=response[:200] if response else None,
                    )
                else:
                    # Parse failed but call succeeded
                    logger.warning(
                        f"    [{model.id}] Parse failed: {response[:100]}"
                    )
                    return PredictionResult(
                        model_id=model.id,
                        probability=None,
                        latency_ms=latency,
                        error="parse_failed",
                        raw_response=response[:200] if response else None,
                    )

            except asyncio.TimeoutError:
                latency = (time.monotonic() - start) * 1000
                logger.warning(f"    [{model.id}] TIMEOUT after {timeout}s ({latency:.0f}ms)")
                return PredictionResult(
                    model_id=model.id,
                    probability=None,
                    latency_ms=latency,
                    error=f"timeout_after_{timeout}s",
                    raw_response=None,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"    [{model.id}] Attempt {attempt + 1}/{max_retries} failed: {last_error[:80]}"
                )

                if attempt < max_retries - 1:
                    backoff = base_backoff ** attempt
                    logger.debug(f"    [{model.id}] Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)

        # All retries exhausted
        latency = (time.monotonic() - start) * 1000
        logger.error(f"    [{model.id}] FAILED after {max_retries} attempts: {last_error[:80]}")
        return PredictionResult(
            model_id=model.id,
            probability=None,
            latency_ms=latency,
            error=last_error,
            raw_response=None,
        )


@dataclass
class BatchPredictionResult:
    """Result of a batch model prediction (multiple questions)."""
    model_id: str
    probabilities: list[float | None]
    latency_ms: float
    error: str | None
    raw_response: str | None = None


async def query_model_batch_async(
    model: Any,
    prompt: str,
    num_questions: int,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
    base_backoff: float = 2.0,
    timeout: float | None = None,
    verbose: bool = False,
) -> BatchPredictionResult:
    """
    Query a model with a batched prompt containing multiple questions.

    Similar to query_model_async but parses multiple probability values
    from the response.

    Args:
        model: Model object with get_response() method
        prompt: The batched prompt to send
        num_questions: Number of questions in the batch
        semaphore: Rate limiting semaphore for this provider
        max_retries: Maximum retry attempts on failure
        base_backoff: Base seconds for exponential backoff
        timeout: Optional timeout in seconds per model query

    Returns:
        BatchPredictionResult with probabilities list, latency, and any errors
    """
    logger.debug(f"    [{model.id}] Waiting for semaphore (batch)...")
    async with semaphore:
        logger.info(f"    [{model.id}] Querying batch of {num_questions} questions...")

        # Verbose: log full prompt
        if verbose:
            logger.info(f"\n{'='*80}\n[PROMPT to {model.id}]\n{'='*80}\n{prompt}\n{'='*80}")

        start = time.monotonic()
        last_error = None

        for attempt in range(max_retries):
            try:
                loop = asyncio.get_running_loop()

                # Log thread pool status periodically
                active = threading.active_count()
                if active > 20:
                    logger.debug(f"    [{model.id}] Active threads: {active}")

                # Capture variables for thread (avoid closure issues)
                # Allow more tokens for batched responses (20 per question)
                max_tokens = max(1000, num_questions * 150)  # Allow verbose reasoning + probabilities

                def make_call(m=model, p=prompt, mt=max_tokens):
                    return m.get_response(p, temperature=0.0, max_tokens=mt)

                api_call = loop.run_in_executor(_thread_pool, make_call)

                # Apply timeout if specified
                if timeout:
                    response = await asyncio.wait_for(api_call, timeout=timeout)
                else:
                    response = await api_call

                latency = (time.monotonic() - start) * 1000

                # Verbose: log full response
                if verbose:
                    logger.info(f"\n{'-'*80}\n[RESPONSE from {model.id}]\n{'-'*80}\n{response}\n{'-'*80}")

                probs = parse_batch_probabilities(response, num_questions)

                # Count successful parses
                success_count = sum(1 for p in probs if p is not None)
                logger.info(
                    f"    [{model.id}] -> {success_count}/{num_questions} parsed ({latency:.0f}ms)"
                )
                if verbose:
                    logger.info(f"    [{model.id}] Parsed probabilities: {probs}")

                return BatchPredictionResult(
                    model_id=model.id,
                    probabilities=probs,
                    latency_ms=latency,
                    error=None if success_count > 0 else "all_parse_failed",
                    raw_response=response[:500] if response else None,
                )

            except asyncio.TimeoutError:
                latency = (time.monotonic() - start) * 1000
                logger.warning(f"    [{model.id}] TIMEOUT after {timeout}s ({latency:.0f}ms)")
                return BatchPredictionResult(
                    model_id=model.id,
                    probabilities=[None] * num_questions,
                    latency_ms=latency,
                    error=f"timeout_after_{timeout}s",
                    raw_response=None,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"    [{model.id}] Attempt {attempt + 1}/{max_retries} failed: {last_error[:80]}"
                )

                if attempt < max_retries - 1:
                    backoff = base_backoff ** attempt
                    logger.debug(f"    [{model.id}] Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)

        # All retries exhausted
        latency = (time.monotonic() - start) * 1000
        logger.error(f"    [{model.id}] FAILED after {max_retries} attempts: {last_error[:80]}")
        return BatchPredictionResult(
            model_id=model.id,
            probabilities=[None] * num_questions,
            latency_ms=latency,
            error=last_error,
            raw_response=None,
        )


async def evaluate_question_batch(
    questions: list[dict],
    models: list[Any],
    rate_limiter: "ProviderRateLimiter",
    world_report: str,
    timeout: float | None = None,
    verbose: bool = False,
) -> list[dict]:
    """
    Evaluate a batch of questions (from the same game) across all models.

    All models are queried simultaneously with a single batched prompt per model.
    This dramatically reduces token usage compared to per-question prompts.

    Args:
        questions: List of question dicts from the same game
        models: List of model objects to evaluate
        rate_limiter: ProviderRateLimiter for managing concurrency
        world_report: JSON string of world report data
        timeout: Optional timeout in seconds per model query

    Returns:
        List of result dicts, one per question, each with predictions from all models
    """
    prompt = build_batch_prompt(questions, world_report)
    num_questions = len(questions)

    timeout_str = f" (timeout={timeout}s)" if timeout else ""
    logger.info(f"  Querying {len(models)} models with batch of {num_questions} questions{timeout_str}...")

    # Launch all model queries in parallel (rate-limited per provider)
    tasks = []
    for model in models:
        sem = rate_limiter.get_semaphore(model.provider_cls)
        tasks.append(query_model_batch_async(model, prompt, num_questions, sem, timeout=timeout, verbose=verbose))

    batch_results = await asyncio.gather(*tasks)

    # Build results list - one entry per question
    results = []
    for q_idx, question in enumerate(questions):
        predictions = {}
        for result in batch_results:
            prob = result.probabilities[q_idx] if q_idx < len(result.probabilities) else None
            predictions[result.model_id] = {
                "probability": prob,
                "latency_ms": result.latency_ms,
                "error": result.error if prob is None else None,
            }

        results.append({
            "question_id": question["question_id"],
            "game_id": question["game_id"],
            "template_id": question.get("template_id"),
            "question_text": question["question_text"],
            "difficulty": question.get("difficulty", {}),
            "ground_truth": question["ground_truth"],
            "predictions": predictions,
        })

    return results


async def evaluate_question(
    question: dict,
    models: list[Any],
    rate_limiter: ProviderRateLimiter,
    world_report: str,
    timeout: float | None = None,
) -> dict:
    """
    Evaluate a single question across all models in parallel.

    All models are queried simultaneously, with rate limiting applied
    per provider to avoid exceeding API limits.

    Args:
        question: Question dict with question_text, ground_truth, etc.
        models: List of model objects to evaluate
        rate_limiter: ProviderRateLimiter for managing concurrency
        world_report: JSON string of world report data
        timeout: Optional timeout in seconds per model query

    Returns:
        Dict with question metadata and predictions from all models
    """
    prompt = build_batch_prompt([question], world_report)

    timeout_str = f" (timeout={timeout}s)" if timeout else ""
    logger.info(f"  Querying {len(models)} models in parallel{timeout_str}...")

    # Launch all model queries in parallel (rate-limited per provider)
    tasks = []
    for model in models:
        sem = rate_limiter.get_semaphore(model.provider_cls)
        tasks.append(query_model_async(model, prompt, sem, timeout=timeout))

    results = await asyncio.gather(*tasks)

    # Build predictions dict
    predictions = {}
    for result in results:
        predictions[result.model_id] = result.to_dict()

    return {
        "question_id": question["question_id"],
        "game_id": question["game_id"],
        "template_id": question.get("template_id"),
        "question_text": question["question_text"],
        "difficulty": question.get("difficulty", {}),
        "ground_truth": question["ground_truth"],
        "predictions": predictions,
    }


def save_checkpoint(
    checkpoint_file: Path,
    results: list[dict],
    metadata: dict,
    completed_batches: set[int] | None = None,
) -> None:
    """Save evaluation progress to checkpoint file."""
    checkpoint = {
        "metadata": metadata,
        "results": results,
        "completed_count": len(results),
    }
    if completed_batches is not None:
        checkpoint["completed_batches"] = sorted(completed_batches)
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    logger.debug(f"Checkpoint saved: {len(results)} questions completed")


def load_checkpoint(checkpoint_file: Path) -> tuple[list[dict], dict, set[int] | None] | None:
    """
    Load checkpoint if it exists.

    Returns:
        Tuple of (results, metadata, completed_batches) or None if no checkpoint.
        completed_batches may be None for legacy checkpoints.
    """
    if not checkpoint_file.exists():
        return None

    with open(checkpoint_file) as f:
        checkpoint = json.load(f)

    completed_batches = checkpoint.get("completed_batches")
    if completed_batches is not None:
        completed_batches = set(completed_batches)

    return (
        checkpoint.get("results", []),
        checkpoint.get("metadata", {}),
        completed_batches,
    )


async def _evaluate_single_question(
    idx: int,
    total: int,
    question: dict,
    models: list[Any],
    rate_limiter: ProviderRateLimiter,
    world_reports_cache: dict[str, str],
    data_dir: Path,
    timeout: float | None = None,
) -> dict | None:
    """Helper to evaluate a single question with logging."""
    difficulty = question.get("difficulty", {}).get("composite", "?")
    logger.info(
        f"\n[{idx+1}/{total}] {question['game_id']}/{question['question_id']} (difficulty={difficulty})"
    )
    logger.info(f"  Q: {question['question_text'][:70]}...")
    logger.info(f"  Ground truth: {question['ground_truth']}")

    # Load world report (cached)
    game_id = question["game_id"]
    if game_id not in world_reports_cache:
        logger.debug(f"  Loading world report for {game_id}...")
        world_reports_cache[game_id] = load_world_report(data_dir, game_id)

    world_report = world_reports_cache[game_id]
    if not world_report:
        logger.warning(f"  No world report found for {game_id}, skipping")
        return None

    # Evaluate question
    result = await evaluate_question(question, models, rate_limiter, world_report, timeout=timeout)

    # Log progress summary
    success_count = sum(
        1 for p in result["predictions"].values()
        if p.get("probability") is not None
    )
    logger.info(
        f"  Completed: {success_count}/{len(models)} successful predictions"
    )

    return result


async def run_evaluation(
    questions: list[dict],
    models: list[Any],
    rate_limiter: ProviderRateLimiter,
    data_dir: Path,
    checkpoint_file: Path | None = None,
    checkpoint_interval: int = 10,
    metadata: dict | None = None,
    concurrent_questions: int = 1,
    timeout: float | None = None,
) -> list[dict]:
    """
    Run parallel evaluation with checkpointing.

    Iterates through questions, querying all models in parallel for each.
    Saves checkpoints periodically to allow resuming interrupted runs.

    Args:
        questions: List of questions to evaluate
        models: List of model objects
        rate_limiter: ProviderRateLimiter for managing concurrency
        data_dir: Directory containing game data
        checkpoint_file: Path to save/load checkpoints (optional)
        checkpoint_interval: Save checkpoint every N questions
        metadata: Evaluation metadata to include in checkpoint
        concurrent_questions: Number of questions to evaluate concurrently
        timeout: Optional timeout in seconds per model query

    Returns:
        List of result dicts for all evaluated questions
    """
    results = []
    world_reports_cache: dict[str, str] = {}
    start_idx = 0

    # Load checkpoint if exists
    if checkpoint_file:
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data:
            results, _, _ = checkpoint_data
            start_idx = len(results)
            logger.info(f"Resuming from checkpoint at question {start_idx}")

    remaining_questions = questions[start_idx:]
    total = len(questions)

    if concurrent_questions > 1:
        logger.info(f"Processing {len(remaining_questions)} questions with concurrency={concurrent_questions}")

        # Process in batches of concurrent_questions
        for batch_start in range(0, len(remaining_questions), concurrent_questions):
            batch = remaining_questions[batch_start:batch_start + concurrent_questions]
            batch_indices = range(start_idx + batch_start, start_idx + batch_start + len(batch))

            # Launch batch concurrently
            tasks = [
                _evaluate_single_question(
                    idx, total, q, models, rate_limiter, world_reports_cache, data_dir, timeout
                )
                for idx, q in zip(batch_indices, batch)
            ]

            batch_results = await asyncio.gather(*tasks)

            # Collect non-None results
            for result in batch_results:
                if result is not None:
                    results.append(result)

            # Checkpoint after batch
            if checkpoint_file and len(results) % checkpoint_interval < concurrent_questions:
                save_checkpoint(checkpoint_file, results, metadata or {})
    else:
        # Sequential processing (original behavior)
        for i, q in enumerate(remaining_questions, start=start_idx):
            result = await _evaluate_single_question(
                i, total, q, models, rate_limiter, world_reports_cache, data_dir, timeout
            )
            if result is not None:
                results.append(result)

            # Checkpoint
            if checkpoint_file and (i + 1) % checkpoint_interval == 0:
                save_checkpoint(checkpoint_file, results, metadata or {})

    # Final checkpoint
    if checkpoint_file:
        save_checkpoint(checkpoint_file, results, metadata or {})

    return results


async def _process_single_batch(
    batch_idx: int,
    batch: list[dict],
    models: list[Any],
    rate_limiter: "ProviderRateLimiter",
    data_dir: Path,
    timeout: float | None,
    verbose: bool,
    semaphore: asyncio.Semaphore,
    total_batches: int,
) -> tuple[int, list[dict] | None]:
    """
    Process a single batch with semaphore-based concurrency control.

    Returns:
        Tuple of (batch_idx, results) where results may be None if skipped.
    """
    async with semaphore:
        if not batch:
            return (batch_idx, None)

        game_id = batch[0]["game_id"]
        templates = set(q.get("template_id", "?") for q in batch)

        logger.info(f"\n[Batch {batch_idx + 1}/{total_batches}] Game: {game_id}")
        logger.info(f"  Questions: {len(batch)}, templates: {len(templates)} unique")

        # Load world report
        world_report = load_world_report(data_dir, game_id)
        if not world_report:
            logger.warning(f"  No world report found for {game_id}, skipping batch")
            return (batch_idx, None)

        # Evaluate batch
        batch_results = await evaluate_question_batch(
            batch, models, rate_limiter, world_report, timeout=timeout, verbose=verbose
        )

        # Log batch summary
        total_preds = sum(
            1 for r in batch_results
            for p in r["predictions"].values()
            if p.get("probability") is not None
        )
        expected_preds = len(batch) * len(models)
        logger.info(f"  Batch {batch_idx + 1} complete: {total_preds}/{expected_preds} predictions successful")

        return (batch_idx, batch_results)


async def run_batch_evaluation(
    question_batches: list[list[dict]],
    models: list[Any],
    rate_limiter: "ProviderRateLimiter",
    data_dir: Path,
    checkpoint_file: Path | None = None,
    checkpoint_interval: int = 5,
    metadata: dict | None = None,
    timeout: float | None = None,
    verbose: bool = False,
    concurrent_batches: int = 5,
) -> list[dict]:
    """
    Run batched evaluation where each batch contains questions from the same game.

    Batches are processed concurrently (up to concurrent_batches at a time) for
    improved throughput. This is more cost-efficient than per-question evaluation
    because the world report is only included once per batch.

    Args:
        question_batches: List of question batches, one per game
        models: List of model objects
        rate_limiter: ProviderRateLimiter for managing concurrency
        data_dir: Directory containing game data
        checkpoint_file: Path to save/load checkpoints (optional)
        checkpoint_interval: Save checkpoint every N completed batches
        metadata: Evaluation metadata to include in checkpoint
        timeout: Optional timeout in seconds per model query
        verbose: If True, log full prompts and responses
        concurrent_batches: Number of batches to process concurrently (default: 5)

    Returns:
        Flat list of result dicts for all evaluated questions
    """
    # Track results by batch index for ordering, and completed batch indices
    results_by_batch: dict[int, list[dict]] = {}
    completed_batches: set[int] = set()

    # Load checkpoint if exists
    if checkpoint_file:
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data:
            existing_results, _, saved_completed = checkpoint_data

            if saved_completed is not None:
                # New checkpoint format: use completed batch indices directly
                completed_batches = saved_completed
                # Reconstruct results_by_batch from existing results
                # Group by batch index (we need to infer from question order)
                q_idx = 0
                for batch_idx, batch in enumerate(question_batches):
                    if batch_idx in completed_batches:
                        results_by_batch[batch_idx] = existing_results[q_idx:q_idx + len(batch)]
                        q_idx += len(batch)
                logger.info(
                    f"Resuming from checkpoint: {len(completed_batches)} batches completed "
                    f"({len(existing_results)} questions)"
                )
            else:
                # Legacy checkpoint format: infer completed batches from question count
                completed_questions = len(existing_results)
                cumulative = 0
                q_idx = 0
                for batch_idx, batch in enumerate(question_batches):
                    if cumulative + len(batch) <= completed_questions:
                        completed_batches.add(batch_idx)
                        results_by_batch[batch_idx] = existing_results[q_idx:q_idx + len(batch)]
                        q_idx += len(batch)
                        cumulative += len(batch)
                    else:
                        break
                logger.info(
                    f"Resuming from legacy checkpoint: {len(completed_batches)} batches completed"
                )

    total_batches = len(question_batches)
    total_questions = sum(len(b) for b in question_batches)
    remaining_batches = [
        (idx, batch) for idx, batch in enumerate(question_batches)
        if idx not in completed_batches
    ]

    logger.info(
        f"Processing {total_batches} game batches ({total_questions} total questions), "
        f"{len(remaining_batches)} remaining, concurrency={concurrent_batches}"
    )

    if not remaining_batches:
        # All done - just flatten and return
        return [r for idx in sorted(results_by_batch.keys()) for r in results_by_batch[idx]]

    # Create semaphore for batch-level concurrency
    semaphore = asyncio.Semaphore(concurrent_batches)

    # Launch all remaining batches (semaphore controls concurrency)
    tasks = [
        asyncio.create_task(
            _process_single_batch(
                batch_idx, batch, models, rate_limiter, data_dir,
                timeout, verbose, semaphore, total_batches
            )
        )
        for batch_idx, batch in remaining_batches
    ]

    # Process as batches complete
    batches_since_checkpoint = 0
    for coro in asyncio.as_completed(tasks):
        batch_idx, batch_results = await coro

        if batch_results is not None:
            results_by_batch[batch_idx] = batch_results
            completed_batches.add(batch_idx)
            batches_since_checkpoint += 1

            # Checkpoint periodically
            if checkpoint_file and batches_since_checkpoint >= checkpoint_interval:
                # Flatten results in order for checkpoint
                ordered_results = [
                    r for idx in sorted(results_by_batch.keys())
                    for r in results_by_batch[idx]
                ]
                save_checkpoint(checkpoint_file, ordered_results, metadata or {}, completed_batches)
                batches_since_checkpoint = 0
                logger.info(f"  Checkpoint saved: {len(completed_batches)}/{total_batches} batches")

    # Final checkpoint
    ordered_results = [
        r for idx in sorted(results_by_batch.keys())
        for r in results_by_batch[idx]
    ]
    if checkpoint_file:
        save_checkpoint(checkpoint_file, ordered_results, metadata or {}, completed_batches)

    return ordered_results
