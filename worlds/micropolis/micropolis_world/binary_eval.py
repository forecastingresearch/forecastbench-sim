"""Output paths + dataset IO for the binary yes/no eval.

The binary counterpart of continuous_eval.py's cache/path/dataset half, rooted
at data/micropolis/binary/ instead of continuous/. The cache layout, batching,
hashing, gathering and dataset writing all come from gather.py; what lives
here is the binary response type and the shape its forecasts take on disk.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from . import module_globals as g
from .continuous_eval import ResponseId
from .gather import EvalPaths, write_dataset

SUBDIR = "binary"

# The binary eval's cache layout — same content-addressing as the continuous
# eval's, under the binary root. The helpers below are kept as module-level
# functions because the script and tests import them by name. The root is
# read through g.DATA_DIR at call time, never copied here: a config's
# 'data_dir' rebinds it after import.
PATHS = EvalPaths(SUBDIR)


def out_dir() -> Path:
    return g.DATA_DIR / SUBDIR


def label_dir(label: str) -> Path:
    return out_dir() / label


def data_path(label: str) -> Path:
    return label_dir(label) / "data.json"


def batch_dir(batch_id: str) -> Path:
    return PATHS.batch_dir(batch_id)


def prompt_path(batch_id: str, phash: str) -> Path:
    return PATHS.prompt_path(batch_id, phash)


def response_path(batch_id: str, model_id: str, phash: str) -> Path:
    return PATHS.response_path(batch_id, model_id, phash)


def usage_path(batch_id: str, model_id: str, phash: str) -> Path:
    return PATHS.usage_path(batch_id, model_id, phash)


@dataclass(frozen=True)
class BinaryResponse:
    """One model's answer to one question.

    `source` is the cached response file, relative to the cache root
    (PATHS.cache_relative), and `line` the 1-based line of it the probability
    was read from; both None when it did not parse or the dataset predates them.
    """

    actual: bool
    probability: float | None
    response_text: str | None = None
    source: str | None = None
    line: int | None = None


BinaryResponses = dict[ResponseId, BinaryResponse]


def save_dataset_binary(
    corpus: list[dict],
    responses: BinaryResponses,
    model_names: list[str],
    path: Path,
) -> Path:
    """Write the corpus and this run's probability forecasts to `path`.

    The binary eval's shape of gather.write_dataset: questions keep their
    resolved bool "answer", and each gathered (question, model) pair carries a
    "probability" (null = answered unusably; an absent row = never gathered),
    the cache-relative response file it was read from and the line within it.
    """
    forecasts = [
        {
            "model_id": model_id,
            "question_id": c["question_id"],
            "probability": r.probability,
            "source": r.source,
            "line": r.line,
        }
        for c in corpus
        for model_id in model_names
        for r in [responses.get(ResponseId(model_id, c["question_id"]))]
        if r is not None
    ]
    return write_dataset(corpus, forecasts, model_names, path)


def load_dataset_binary(path: Path) -> tuple[list[dict], BinaryResponses, list[str]]:
    """Read back what save_dataset_binary wrote, as (corpus, responses, models)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/run_eval_binary.py first"
        )
    data = json.loads(path.read_text())
    corpus = data["questions"]
    actual = {c["question_id"]: c["answer"] for c in corpus}
    responses: BinaryResponses = {
        ResponseId(f["model_id"], f["question_id"]): BinaryResponse(
            actual=actual[f["question_id"]],
            probability=f["probability"],
            # .get: datasets written before these fields existed lack them.
            source=f.get("source"),
            line=f.get("line"),
        )
        for f in data["forecasts"]
    }
    return corpus, responses, data["models"]
