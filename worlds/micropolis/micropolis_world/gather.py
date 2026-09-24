"""Prompt gathering shared by the continuous and binary evals.

Both evals ask the same thing of the models — batch a corpus by game report,
send the uncached batches concurrently, cache what comes back — and differ
only in how the prompt is built and how the reply is read. This module owns
everything up to the raw response text; each eval keeps its own parsing and
its own response type, which is exactly where the two genuinely diverge.
"""

import asyncio
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fbsim_core.evaluation.models import get_models

from . import messages as msg
from . import module_globals as g
from .cache_keys import prompt_hash
from .model_ids import filename_slug
from .prompting import (
    PromptJob,
    PromptResult,
    format_eta,
    format_latency,
    run_prompts,
)
from .usage import save_usage

# Raw response text per (batch_id, model_name). A key that is absent means the
# call failed this run: downstream that reads as "never gathered; re-run",
# which is right, since re-running retries exactly the failures.
RawResponses = dict[tuple[str, str], str | None]

# The cache file each (batch_id, model_name) response was read from or written
# to. Every key is present, cached or not: a parse warning names the file the
# text would be in, which is also where a miss would land.
ResponsePaths = dict[tuple[str, str], Path]


def batch_id_for(question: dict) -> str:
    """The batch a corpus question is prompted in.

    Questions sharing a scenario and snapshot turn share a game report — the
    bulk of the prompt — so they are asked together in one numbered prompt.
    """
    return f"{question['scenario_id']}_T{question['snapshot_turn']}"


def _split_evenly(questions: list[dict], limit: int) -> list[list[dict]]:
    """`questions` cut into consecutive chunks of at most `limit` each.

    The chunk count is what `limit` really fixes: enough chunks that none
    exceeds it. Their sizes are then evened out rather than filling each chunk
    before starting the next, so 20 questions at a limit of 12 come out 10 and
    10 instead of 12 and 8 — no prompt is left with a lopsided tail asking
    about only a question or two.
    """
    nchunks = -(-len(questions) // limit)  # ceiling division
    base, extra = divmod(len(questions), nchunks)
    chunks = []
    start = 0
    for i in range(nchunks):
        # The first `extra` chunks take one more, so sizes differ by at most 1.
        stop = start + base + (1 if i < extra else 0)
        chunks.append(questions[start:stop])
        start = stop
    return chunks


def group_into_batches(
    corpus: list[dict], questions_per_prompt: int = -1
) -> dict[str, list[dict]]:
    """Group corpus questions by batch_id, preserving corpus order.

    `questions_per_prompt` caps how many questions one prompt may ask. The
    default -1 means no cap: a scenario's whole snapshot goes in one prompt,
    which is the cheapest way to ask them since they share a game report. A
    positive value splits a batch that would exceed it into consecutive chunks
    (see _split_evenly), each becoming its own batch — its own prompt repeating
    the report, its own cache directory, and its own response — so a split
    batch stays one hash per batch id for analyze_usage, and a chunk that fails
    is retried on its own. Chunk ids are suffixed "_c{i}of{n}"; an unsplit
    batch keeps the plain id, so existing caches stay addressable.
    """
    batches: dict[str, list[dict]] = {}
    for c in corpus:
        batches.setdefault(batch_id_for(c), []).append(c)
    if questions_per_prompt < 0:
        return batches

    if questions_per_prompt == 0:
        raise ValueError("questions_per_prompt must be -1 or a positive integer")

    split: dict[str, list[dict]] = {}
    for bid, questions in batches.items():
        if len(questions) <= questions_per_prompt:
            split[bid] = questions
            continue
        chunks = _split_evenly(questions, questions_per_prompt)
        for i, chunk in enumerate(chunks, 1):
            split[f"{bid}_c{i}of{len(chunks)}"] = chunk
    return split


def write_dataset(
    corpus: list[dict],
    forecasts: list[dict],
    model_names: list[str],
    path: Path,
) -> Path:
    """Write the corpus and this run's forecasts as one self-contained file.

    The analysis and plotting scripts read only this, so it carries everything
    they need: the questions with their resolved answers, and each model's
    forecast per question — `forecasts` rows as the caller's eval shapes them,
    since a percentile set and a probability are the one thing the two evals
    do not share. The full response text is deliberately left out — it stays
    in the response cache, and including it here would multiply the file size
    for something no downstream script reads.
    """
    # Prompts are megabytes of world report repeated per question; the
    # downstream scripts want the question, not the prompt that produced it.
    dropped = {"context"}
    questions = [{k: v for k, v in c.items() if k not in dropped} for c in corpus]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"models": model_names, "questions": questions, "forecasts": forecasts},
            indent=2,
        )
    )
    return path


@dataclass(frozen=True)
class EvalPaths:
    """Where one eval caches its prompts, responses and usage sidecars.

    Batch prompts and raw model responses are cached under `out_dir`, shared
    across every label: the cache filename already carries the prompt's hash,
    so two labels asking an identical prompt reuse the same cached response
    rather than paying for it twice. The hash is the prompt's content, so a
    batch directory can hold more than one prompt variant — a template or
    history_freq change simply adds new files alongside the old ones instead
    of colliding with or invalidating them.

    Holds the eval's subdirectory name, not its path: the data directory it
    sits under is fixed when a config is loaded, after this object was built
    at import, so out_dir is resolved against g.DATA_DIR on every read. That
    is what lets a config's 'data_dir' give a run a cache of its own.
    """

    subdir: str

    @property
    def out_dir(self) -> Path:
        return g.DATA_DIR / self.subdir

    @property
    def cache_dir(self) -> Path:
        return self.out_dir / "cache"

    def batch_dir(self, batch_id: str) -> Path:
        return self.cache_dir / batch_id

    def cache_relative(self, path: Path) -> str:
        """`path` as written in a dataset: relative to the cache root, POSIX."""
        return path.relative_to(self.cache_dir).as_posix()

    def prompt_path(self, batch_id: str, phash: str) -> Path:
        return self.batch_dir(batch_id) / f"prompt-{phash}.txt"

    def response_path(self, batch_id: str, model_id: str, phash: str) -> Path:
        return (
            self.batch_dir(batch_id) / f"response-{filename_slug(model_id)}-{phash}.txt"
        )

    def usage_path(self, batch_id: str, model_id: str, phash: str) -> Path:
        """Tokens and cost of the call that produced the matching response.

        A cached response is never re-fetched, so what it cost has to be
        recorded when it is first paid or it is lost on every later run.
        Written beside the response and keyed the same way, so the pair stays
        together — the slug must match response_path's exactly, or the sidecar
        lands next to nothing.
        """
        return (
            self.batch_dir(batch_id) / f"usage-{filename_slug(model_id)}-{phash}.json"
        )


def gather_raw_responses(
    corpus: list[dict],
    model_names: list[str],
    *,
    paths: EvalPaths,
    build_prompt: Callable[[str, list[dict]], str],
    concurrency: int | None = None,
    questions_per_prompt: int = -1,
    cache_only: bool = False,
) -> tuple[dict[str, list[dict]], RawResponses, ResponsePaths]:
    """Prompt each model on each batch of questions, reusing cached responses.

    Returns the batches (so the caller can parse each one's answers back onto
    its questions), the raw text each (batch, model) produced, and the cache
    file that text lives in — which the caller names in its parse warnings, so
    an unusable forecast can be traced to the response that caused it.

    `build_prompt` turns one batch's shared game report and its questions into
    the prompt asking them. Cache files are named with that prompt's hash, so a
    response is only ever reused when it was gathered under the exact prompt
    being asked now — any config change that alters the prompt changes the
    hash, which simply misses the cache rather than risking a stale match.
    An empty reply — a reasoning model can burn the whole token budget
    thinking — is not cached, so the next run retries it; a non-empty reply is
    cached even when unparseable, since retrying greedy decoding would return
    the same text.

    All uncached (batch, model) calls run concurrently under run_prompts's
    global cap, and each response is written to its cache file the moment it
    lands, so an interrupted run keeps what it already paid for. A failed call
    is reported and skipped rather than aborting the run — nothing is cached
    for it, so re-running the script retries exactly the failures.

    `cache_only` makes no calls at all: the cache hits are returned and the
    misses are left absent, exactly as a failed call would be, so the caller
    builds a dataset from whatever was already gathered. Prompts are still
    built (their hashes are what finds the cache) and still written, so the
    misses are inspectable.
    """
    batches = group_into_batches(corpus, questions_per_prompt)
    prompts = {
        bid: build_prompt(questions[0]["context"], questions)
        for bid, questions in batches.items()
    }
    phashes = {bid: prompt_hash(prompt) for bid, prompt in prompts.items()}
    ppaths = {bid: paths.prompt_path(bid, phashes[bid]) for bid in prompts}
    for bid, prompt in prompts.items():
        if not ppaths[bid].exists():
            paths.batch_dir(bid).mkdir(parents=True, exist_ok=True)
            ppaths[bid].write_text(prompt)

    models = get_models(model_names)
    rpaths = {
        (bid, model_name): paths.response_path(bid, model_name, phashes[bid])
        for bid in batches
        for model_name in model_names
    }
    ncached = sum(1 for p in rpaths.values() if p.exists())
    nmissing = len(rpaths) - ncached
    if cache_only:
        print(f"{ncached} of {len(rpaths)} batch responses cached; cache-only run")
        if nmissing:
            msg.warn(f"{nmissing} response(s) not cached; omitted from the dataset")
    else:
        print(
            f"{ncached} of {len(rpaths)} batch responses cached; generating {nmissing}"
        )

    # Split cache hits from the calls still to make. raws holds the text to
    # parse per (batch_id, model_name); a key that is still absent at parse
    # time means the call failed this run.
    raws: RawResponses = {}
    jobs: list[PromptJob] = []
    for model_name, model in zip(model_names, models):
        for bid in batches:
            key = (bid, model_name)
            if rpaths[key].exists():
                raws[key] = rpaths[key].read_text()
            elif not cache_only:
                # prompt_model_async rather than model.get_response, because
                # the finish reason is what distinguishes a model that answered
                # badly from one that never got to answer at all, and the usage
                # is what it cost either way.
                jobs.append(
                    PromptJob(
                        key=key,
                        model=model,
                        model_name=model_name,
                        messages=[{"role": "user", "content": prompts[bid]}],
                    )
                )

    model_cost: dict[str, float] = defaultdict(float)
    nunpriced: Counter = Counter()
    failures: list[PromptResult] = []

    async def consume() -> None:
        # The workers only make API calls; every print, dict update and disk
        # write happens here in the single consumer task, so nothing needs a
        # lock. One complete line per completed call — completions from
        # different providers interleave, so a line can't be left dangling for
        # its cost the way the serial version's was.
        done = 0
        start = time.perf_counter()
        async for result in run_prompts(jobs, limit=concurrency):
            done += 1
            eta = format_eta(start, done, len(jobs))
            bid, model_name = result.job.key
            prefix = f"[{done}/{len(jobs)}] {model_name} <- {ppaths[bid]}"
            if not result.ok:
                failures.append(result)
                err = result.error
                msg.error(f"{prefix}  FAILED: {type(err).__name__}: {err}{eta}")
                continue
            resp = result.response
            # How long this one call took, next to what it cost: a batch that
            # is slow and a batch that is expensive are different problems.
            # Retries are called out too — they are why a call can sit for
            # minutes behind a time that reads as seconds.
            took = format_latency(resp.usage.latency_ms, resp.retries)
            cost = resp.usage.cost_usd
            if cost is None:
                # Reported, not counted: a call the backend couldn't price
                # would otherwise be summed into the total as free.
                nunpriced[model_name] += 1
                print(
                    f"{prefix}  cost unknown, {resp.usage.tokens()}{took}{eta}",
                    flush=True,
                )
            else:
                # Cents: a single batch is a fraction of a cent to a few
                # cents, which dollars would print as 0.00.
                model_cost[model_name] += cost
                print(
                    f"{prefix}  {cost * 100:.3f}c, {resp.usage.tokens()}{took}{eta}",
                    flush=True,
                )
            g.warn_if_truncated(model_name, resp.finish_reason)
            raws[result.job.key] = resp.text
            if resp.text:
                # Saved as soon as it lands, so an interrupted run keeps what it
                # already paid for. Usage is written only alongside a kept
                # response, so the pair never disagrees about whether the call
                # needs paying for again.
                rpaths[result.job.key].write_text(resp.text)
                save_usage(paths.usage_path(bid, model_name, phashes[bid]), resp.usage)

    if jobs:
        asyncio.run(consume())

    # What this run paid, per model. Cached batches cost nothing, so a fully
    # cached model reports $0.00 rather than what it originally cost. A
    # cache-only run pays for nothing at all, so it prints no totals.
    if cache_only:
        return batches, raws, rpaths

    print("Per-model totals for this run:")
    for model_name in model_names:
        total = f"${model_cost[model_name]:.2f}"
        if nunpriced[model_name]:
            total += f" + {nunpriced[model_name]} unpriced call(s)"
        print(f"  {model_name}: {total}")
    if failures:
        msg.error(
            f"{len(failures)} call(s) failed (not cached; "
            "re-run this script to retry them):"
        )
        for result in failures:
            bid, model_name = result.job.key
            msg.plain(f"  {model_name} <- {ppaths[bid]}")
            msg.plain(f"    {type(result.error).__name__}: {result.error}")
    return batches, raws, rpaths
