"""Elicitation for the natural-conditional prompts, after FreeCiv's elicit_natcond_v1.py.

Turn 1 for every question A, then turn 2 for every cell, continuing the
conversation with that model's own turn-1 reply. One results.jsonl row per
call (append-only); a rerun resumes from it. An unparsed reply is asked again
up to `reask` times, as FreeCiv's run 1 did. Kept backend-free at import:
prompting.py loads the LLM client on the first call.
"""

import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from .natcond_prompts import conversation
from .prompting import PromptJob, run_prompts


def _prob(v: str) -> float | None:
    x = float(v)
    x = x / 100 if x > 1 else x
    return x if 0 <= x <= 1 else None


def parse_prob(text: str | None) -> tuple[float | None, str | None]:
    """FreeCiv's run-1 parser: strict = the last <probability> tag; lenient =
    'probability ... 0.xx', a bare '<0.3>' or a final bare number."""
    m = re.findall(r"<probability>\s*([0-9]*\.?[0-9]+)\s*%?\s*</probability>", text or "")
    if m:
        return _prob(m[-1]), "strict"
    tail = (text or "")[-400:]
    t = re.sub(r"(?i)\s*</?probability>\s*$", "", tail.strip()).rstrip(".* ")
    m = (re.findall(r"<\s*([0-9]*\.?[0-9]+)\s*%?\s*>\s*$", t)
         or re.findall(r"(?i)probability[^0-9]{0,40}([0-9]*\.?[0-9]+)\s*%?", t)
         or re.findall(r"(?:^|[\s:=~≈])(0?\.[0-9]+|1\.0+|1|0)\s*$", t))
    v = _prob(m[-1]) if m else None
    return v, ("lenient" if v is not None else None)


def messages_sha(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]


class Results:
    """The rows of results.jsonl, by (model, arm, item). The row a key resolves
    to is its first parsed one, else its latest. Given a messages hash, only
    rows asked with exactly those messages count, so a changed prompt is asked
    again and the old rows stay on file."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: dict[tuple, list[dict]] = defaultdict(list)
        if path.exists():
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    self.rows[r["model"], r["arm"], r["item"]].append(r)

    def _rows(self, key: tuple, sha: str | None) -> list[dict]:
        rows = self.rows.get(key, [])
        return [r for r in rows if r["messages_sha"] == sha] if sha else rows

    def best(self, key: tuple, sha: str | None = None) -> dict | None:
        rows = self._rows(key, sha)
        return next((r for r in rows if r["value"] is not None), rows[-1] if rows else None)

    def needs(self, key: tuple, reask: int, sha: str | None = None) -> bool:
        """Not yet answered: no row, or no parsed value with re-asks left.
        A call that errored is not an answer and does not use up a re-ask."""
        rows = self._rows(key, sha)
        if any(r["value"] is not None for r in rows):
            return False
        return sum(r["error"] is None for r in rows) <= reask

    def add(self, row: dict) -> None:
        self.rows[row["model"], row["arm"], row["item"]].append(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")


def _row(job: PromptJob, meta: dict, result) -> dict:
    r = result.response
    text = r.text if r else None
    value, mode = parse_prob(text)
    return {
        "model": job.model_name, **meta, "value": value, "parse_mode": mode, "text": text,
        "finish_reason": r.finish_reason if r else None, "retries": r.retries if r else None,
        "usage": r.usage.to_dict() if r else None,
        "error": None if result.ok else f"{type(result.error).__name__}: {result.error}",
        "messages_sha": messages_sha(job.messages), "ts": time.time(),
    }


async def elicit(jobs: dict[str, list[dict]], models: list[str], results: Results, *,
                 reask: int = 1, concurrency: int | None = None, retries: int = 8, log=print) -> tuple[int, int]:
    """Turn 1, then turn 2, for every model; each arm re-asks unparsed replies.
    Returns (prompts with a parsed answer, prompts), counting only answers to
    the current messages."""
    t1_by_q = {j["question_id"]: j for j in jobs["t1"]}

    async def run_arm(arm: str, build) -> None:
        for attempt in range(reask + 1):
            todo, meta = [], {}
            for m in models:
                for j in jobs[arm]:
                    key = (m, arm, j["item"])
                    msgs, extra = build(m, j)
                    if msgs is None or not results.needs(key, reask, messages_sha(msgs)):
                        continue
                    todo.append(PromptJob(key=key, model=SimpleNamespace(id=m), model_name=m, messages=msgs))
                    meta[key] = {"arm": arm, "item": j["item"], "question_id": j["question_id"],
                                 "cell_id": j.get("cell_id"), "block": j.get("block"), **extra}
            if not todo:
                return
            log(f"{arm}: {len(todo)} calls" + (f" (re-ask {attempt})" if attempt else ""))
            done = 0
            async for res in run_prompts(todo, limit=concurrency, num_retries=retries):
                row = _row(res.job, meta[res.job.key], res)
                results.add(row)
                done += 1
                cost = (row["usage"] or {}).get("cost_usd")
                log(f"  [{done}/{len(todo)}] {res.job.model_name} {row['item']}: "
                    + (row["error"] or f"p={row['value']} ({row['parse_mode']})"
                       + (f", ${cost:.4f}" if cost is not None else "")))

    def build_t1(m, j):
        return conversation(j), {}

    def build_t2(m, j):
        t1_job = t1_by_q[j["question_id"]]
        t1 = results.best((m, "t1", j["question_id"]), messages_sha(conversation(t1_job)))
        if not t1 or not t1["text"]:
            return None, {}  # FreeCiv: no turn 2 without a turn-1 reply
        return conversation(t1_job, t1["text"], j), {"t1_value": t1["value"]}

    await run_arm("t1", build_t1)
    await run_arm("t2", build_t2)

    answered = total = 0
    for arm, build in (("t1", build_t1), ("t2", build_t2)):
        for m in models:
            for j in jobs[arm]:
                total += 1
                msgs, _ = build(m, j)
                best = results.best((m, arm, j["item"]), messages_sha(msgs)) if msgs else None
                answered += bool(best and best["value"] is not None)
    return answered, total
