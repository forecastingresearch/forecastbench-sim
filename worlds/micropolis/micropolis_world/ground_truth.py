"""Ground truth from reseeded continuations of a scenario.

Drives the engine's run_continuations.js — one trunk run to the snapshot, then
many continuations from a byte copy of the engine state, each with its own RNG
seed — and reads its JSONL stream to tally, per horizon, how many
continuations resolve each binary question Yes and what value each continuous
metric takes at the resolution turn. Output is one JSONL file per
(scenario, snapshot turn) under data/micropolis/ground_truth/, one line per
horizon.
"""

import json
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from . import module_globals as g
from .binary_questions import (
    QUESTION_IDS,
    RunIndex,
    check_structural_constraints,
    messages_from_events,
    resolve_all,
)
from .city_sim import CitySimulation, _load_jsonl, turn_of

SUBDIR = "ground_truth"


def out_dir() -> Path:
    """Where the ground-truth files live: under the process's data directory."""
    return g.DATA_DIR / SUBDIR


# No package.json alias exists for the script, unlike run-sim, so tsx is
# invoked directly from the engine's app directory.
PRODUCER = ["pnpm", "tsx", "cli/run_continuations.js"]

# What a stream row carries on top of the run_sim.js row it otherwise equals.
STREAM_FIELDS = ("kind", "seed")


class StreamError(ValueError):
    """The producer's stream is not what the reader was told to expect."""


def output_path(sim: CitySimulation, snapshot_turn: int) -> Path:
    return output_path_for(sim.get_id_str(), snapshot_turn)


def output_path_for(scenario_id: str, snapshot_turn: int) -> Path:
    return out_dir() / f"{scenario_id}_T{snapshot_turn}.jsonl"


def seed_shards(nseeds: int, jobs: int) -> list[tuple[int, int]]:
    """Branch seeds 1..nseeds as at most `jobs` contiguous inclusive ranges."""
    if nseeds < 1:
        raise ValueError(f"need at least one branch seed, got {nseeds}")
    jobs = max(1, min(jobs, nseeds))
    base, extra = divmod(nseeds, jobs)
    shards, lo = [], 1
    for i in range(jobs):
        hi = lo + base + (1 if i < extra else 0) - 1
        shards.append((lo, hi))
        lo = hi + 1
    return shards


def producer_command(
    city: str,
    seed: int,
    disasters: bool,
    snapshot_turn: int,
    horizon: int,
    seed_lo: int,
    seed_hi: int,
) -> list[str]:
    """The engine invocation for one shard of continuations.

    Row T of a log is taken at tick 16T+15, the end of turn T, and the report
    the models forecast from shows row S. Branching at S+1 puts the branch at
    tick 16(S+1), so every continuation starts from exactly that reported state
    and its rows are turns S+1..S+H, the window (S, S+H] the questions resolve
    over.
    """
    return [
        *PRODUCER,
        "--city",
        city,
        "--seed",
        str(seed),
        "--branch-at",
        str(snapshot_turn + 1),
        "--horizon",
        str(horizon),
        "--branch-seeds",
        f"{seed_lo}-{seed_hi}",
        "--disasters" if disasters else "--no-disasters",
    ]


@dataclass
class ShardResult:
    """Tallies over one set of continuations, all from the same trunk."""

    trunk_log: list[dict]
    seeds: list[int]
    yes: dict[int, Counter]  # horizon -> qid -> continuations resolving Yes
    values: dict[int, dict[str, list]]  # horizon -> metric -> value per continuation

    @property
    def n(self) -> int:
        return len(self.seeds)


def _strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in STREAM_FIELDS}


def _check_header(
    rec: dict, city: str, disasters: bool, snapshot_turn: int, hmax: int
) -> None:
    expected = {
        "city": city,
        "disasters": disasters,
        "branchAt": snapshot_turn + 1,
        "trunkOutput": True,
    }
    for key, value in expected.items():
        if rec.get(key) != value:
            raise StreamError(f"header {key}={rec.get(key)!r}, expected {value!r}")
    if rec["horizon"] < hmax:
        raise StreamError(
            f"producer horizon {rec['horizon']} is shorter than the longest "
            f"requested horizon {hmax}"
        )


def _check_turns(rows: list[dict], first: int, last: int, what: str) -> None:
    """Rows must be exactly turns first..last, one each, in order."""
    turns = [turn_of(r) for r in rows]
    if turns != list(range(first, last + 1)):
        got = f"{turns[0]}..{turns[-1]} ({len(turns)} rows)" if turns else "no rows"
        raise StreamError(f"{what}: expected turns {first}..{last}, got {got}")


def consume_stream(
    lines: Iterable[bytes | str],
    city: str,
    disasters: bool,
    snapshot_turn: int,
    horizons: list[int],
) -> ShardResult:
    """Tally one producer's stream (header, trunk, continuations).

    The trunk must be in the stream: the resolver needs the state at the
    snapshot and the whole population history (B8), and the trunk's last tick
    already belongs to turn S+1 under turn_of, so its messages are part of
    every continuation's window.
    """
    S = snapshot_turn
    header: dict | None = None
    trunk: RunIndex | None = None
    trunk_rows: list[dict] = []
    trunk_events: list[dict] = []
    rows: list[dict] = []
    events: list[dict] = []
    result = ShardResult(
        trunk_log=[],
        seeds=[],
        yes={h: Counter() for h in horizons},
        values={h: {m: [] for m in g.METRICS} for h in horizons},
    )

    for raw in lines:
        rec = json.loads(raw)
        kind = rec["kind"]
        if kind == "stats":
            (trunk_rows if rec["seed"] is None else rows).append(_strip(rec))
        elif kind == "event":
            (trunk_events if rec["seed"] is None else events).append(_strip(rec))
        elif kind == "header":
            _check_header(rec, city, disasters, S, max(horizons))
            header = rec
        elif kind == "begin":
            if header is None:
                raise StreamError("run begins before the header")
            if rec["seed"] is None:
                trunk_rows, trunk_events = [], []
            else:
                if trunk is None:
                    raise StreamError("continuation before the trunk finished")
                rows, events = [], []
        elif kind == "end":
            if rec["seed"] is None:
                _check_turns(trunk_rows, 0, S, f"{city} trunk")
                trunk = RunIndex(
                    trunk_rows, messages_from_events(trunk_events, f"{city} trunk")
                )
                result.trunk_log = trunk_rows
                continue
            assert header is not None and trunk is not None
            seed = rec["seed"]
            what = f"{city} branch seed {seed}"
            _check_turns(rows, S + 1, S + header["horizon"], what)
            run = RunIndex(
                trunk.log_data + rows,
                trunk.msgs + messages_from_events(events, what),
            )
            for h in horizons:
                answers = resolve_all(run, S, S + h)
                check_structural_constraints(city, answers)
                result.yes[h].update(qid for qid, yes in answers.items() if yes)
                state = run.state_at(S + h)
                for metric in g.METRICS:
                    result.values[h][metric].append(state[metric])
            result.seeds.append(seed)
        else:
            raise StreamError(f"unknown line kind {kind!r}")

    if header is None:
        raise StreamError("empty stream")
    if trunk is None:
        raise StreamError("stream ended before the trunk finished")
    expected = list(range(header["seedLo"], header["seedHi"] + 1))
    if result.seeds != expected:
        raise StreamError(
            f"stream ended after {len(result.seeds)} of {len(expected)} continuations"
        )
    return result


def run_shard(
    city: str,
    seed: int,
    disasters: bool,
    snapshot_turn: int,
    horizons: list[int],
    seed_lo: int,
    seed_hi: int,
) -> ShardResult:
    """Run one producer and tally its stream as it arrives.

    stderr goes to a temp file rather than a pipe: the producer only writes
    progress and timing there, and a full pipe would stall it.
    """
    cmd = producer_command(
        city, seed, disasters, snapshot_turn, max(horizons), seed_lo, seed_hi
    )
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            cmd, cwd=g.engine_app_path(continuations=True), stdout=subprocess.PIPE, stderr=err
        )
        assert proc.stdout is not None
        try:
            result = consume_stream(
                proc.stdout, city, disasters, snapshot_turn, horizons
            )
        finally:
            proc.stdout.close()
            returncode = proc.wait()
        if returncode != 0:
            err.seek(0)
            tail = err.read().decode(errors="replace")[-2000:]
            raise RuntimeError(
                f"{' '.join(cmd)} exited with code {returncode}:\n{tail}"
            )
    return result


def merge_shards(shards: list[ShardResult]) -> ShardResult:
    """Combine shards of one (scenario, snapshot): counts add, values concatenate."""
    if not shards:
        raise ValueError("no shards to merge")
    first = shards[0]
    for other in shards[1:]:
        if other.trunk_log != first.trunk_log:
            raise StreamError(
                "shards disagree on the trunk; the engine is not deterministic"
            )
    merged = ShardResult(
        trunk_log=first.trunk_log,
        seeds=[],
        yes={h: Counter() for h in first.yes},
        values={h: {m: [] for m in vs} for h, vs in first.values.items()},
    )
    for shard in shards:
        merged.seeds.extend(shard.seeds)
        for h in merged.yes:
            merged.yes[h].update(shard.yes[h])
            for metric, values in shard.values[h].items():
                merged.values[h][metric].extend(values)
    return merged


def cross_check_trunk(trunk_log: list[dict], sim: CitySimulation) -> str:
    """Compare the trunk against the cached run_sim.js log the eval was built on.

    Raises on a mismatch: the continuations would then be conditioned on a
    state other than the one the models saw. Returns a one-line note otherwise.
    """
    path = sim.get_data_file_path("log")
    if not path.exists():
        return f"no cached run at {path} to compare the trunk against"
    cached = _load_jsonl(path)
    n = min(len(cached), len(trunk_log))
    for t in range(n):
        if cached[t] != trunk_log[t]:
            key = next(
                k
                for k in list(cached[t]) + list(trunk_log[t])
                if cached[t].get(k) != trunk_log[t].get(k)
            )
            raise RuntimeError(
                f"{sim.get_id_str()}: trunk differs from the cached run {path} at "
                f"turn {t}, field {key!r}: cached {cached[t].get(key)!r}, "
                f"stream {trunk_log[t].get(key)!r} — the continuations would not "
                "start from the state the eval reported"
            )
    if len(cached) < len(trunk_log):
        return f"trunk matches the cached run over turns 0..{n - 1} (cached run ends there)"
    return f"trunk matches the cached run over turns 0..{n - 1}"


def _averages(values: dict[str, list]) -> dict[str, float | None]:
    """Mean over continuations per metric; None for a metric with no values."""
    return {
        metric: (sum(vs) / len(vs) if vs else None) for metric, vs in values.items()
    }


def ground_truth_lines(
    sim: CitySimulation,
    snapshot_turn: int,
    horizons: list[int],
    merged: ShardResult,
    fbsim_commit: str,
    engine_commit: str,
) -> list[dict]:
    """One output record per horizon."""
    return [
        {
            "scenario_id": sim.get_id_str(),
            "city": sim.city_name,
            "seed": sim.seed,
            "disasters": sim.disasters,
            "snapshot_turn": snapshot_turn,
            "horizon": h,
            "resolution_turn": snapshot_turn + h,
            "n_continuations": merged.n,
            "branch_seeds": [min(merged.seeds), max(merged.seeds)],
            "counts": {qid: merged.yes[h][qid] for qid in QUESTION_IDS},
            "values": merged.values[h],
            "averages": _averages(merged.values[h]),
            "fbsim_commit": fbsim_commit,
            "engine_commit": engine_commit,
        }
        for h in horizons
    ]


def write_lines(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(line) + "\n" for line in lines)


def load_lines(path: Path) -> list[dict]:
    return _load_jsonl(path)


class Truth(NamedTuple):
    """Ground-truth P(Yes) for one binary question and the continuations behind it."""

    p: float
    n: int


def load_truths(corpus: list[dict]) -> dict[str, Truth]:
    """Ground-truth P(Yes) per question_id for a binary corpus.

    Reads the tally file of each (scenario, snapshot) the corpus touches and
    errors on anything missing, so a report never quietly covers less than its
    config asks for.
    """
    files: dict[Path, dict[int, dict]] = {}
    truths: dict[str, Truth] = {}
    for c in corpus:
        path = output_path_for(c["scenario_id"], c["snapshot_turn"])
        if path not in files:
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found — run scripts/extract_ground_truth.py first"
                )
            files[path] = {line["horizon"]: line for line in load_lines(path)}
        line = files[path].get(c["horizon"])
        if line is None:
            raise FileNotFoundError(
                f"{path} has no horizon {c['horizon']} — rerun "
                "scripts/extract_ground_truth.py for this config"
            )
        n = line["n_continuations"]
        truths[c["question_id"]] = Truth(line["counts"][c["qid"]] / n, n)
    return truths


def lines_for(corpus: list[dict]) -> dict[str, dict]:
    """The tally line each corpus question resolves against, per question_id.

    Reads each (scenario, snapshot) file once and errors on a missing file or
    horizon the way load_truths does, so a report never quietly covers less
    than its config asks for.
    """
    files: dict[Path, dict[int, dict]] = {}
    out: dict[str, dict] = {}
    for c in corpus:
        path = output_path_for(c["scenario_id"], c["snapshot_turn"])
        if path not in files:
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found — run scripts/extract_ground_truth.py first"
                )
            files[path] = {line["horizon"]: line for line in load_lines(path)}
        line = files[path].get(c["horizon"])
        if line is None:
            raise FileNotFoundError(
                f"{path} has no horizon {c['horizon']} — rerun "
                "scripts/extract_ground_truth.py for this config"
            )
        out[c["question_id"]] = line
    return out


def load_averages(corpus: list[dict]) -> dict[str, float | None]:
    """Mean outcome per question_id: the "averages" entry, by metric.

    The mean over the reseeded continuations of what the metric read at the
    resolution turn. None where the file has no value for that metric.
    """
    return {
        qid: line["averages"].get(c["metric"])
        for c, (qid, line) in zip(corpus, lines_for(corpus).items(), strict=True)
    }


def load_outcomes(corpus: list[dict]) -> dict[str, list[float] | None]:
    """Every continuation's outcome per question_id: the "values" entry, by metric.

    The replay distribution excess CRPS is scored against. None where the file
    has no values for that metric.
    """
    lines = lines_for(corpus)
    return {
        c["question_id"]: lines[c["question_id"]]["values"].get(c["metric"]) or None
        for c in corpus
    }


def load_expected_persistence(
    corpus: list[dict], snapshots: dict[tuple[str, int], dict | None]
) -> dict[str, float | None]:
    """Expected CRPS of the persistence forecast, per question_id.

    Persistence puts all five percentiles on the snapshot value, so its CRPS
    against one outcome is |snapshot - outcome|; averaged over every
    continuation this is the mean absolute deviation of the outcome
    distribution about the snapshot — how far the metric was going to move,
    measured without reference to the single realized future, so no model is
    flattered or punished by a lucky draw. It is 0 only where every
    continuation equals the snapshot, i.e. where the metric provably could not
    move.

    `snapshots` maps (scenario_id, snapshot_turn) to that scenario's log row at
    the snapshot turn, or None where the run is not cached; the caller supplies
    it because reading run logs belongs to the analysis, not here. A question
    whose row or metric values are missing gets None.
    """
    lines = lines_for(corpus)
    out: dict[str, float | None] = {}
    for c in corpus:
        values = lines[c["question_id"]]["values"].get(c["metric"])
        row = snapshots.get((c["scenario_id"], c["snapshot_turn"]))
        if not values or row is None or c["metric"] not in row:
            out[c["question_id"]] = None
            continue
        snapshot = row[c["metric"]]
        out[c["question_id"]] = sum(abs(snapshot - v) for v in values) / len(values)
    return out


def covers(lines: list[dict], horizons: list[int], nseeds: int) -> bool:
    """Whether an existing file has every horizon at >= nseeds continuations.

    The question set is part of coverage: a file tallied before a question was
    added, removed or relabelled holds counts under the wrong ids, and skipping
    it would silently resolve against stale labels.
    """
    by_horizon = {line["horizon"]: line for line in lines}
    return all(
        h in by_horizon
        and by_horizon[h]["n_continuations"] >= nseeds
        and set(by_horizon[h]["counts"]) == set(QUESTION_IDS)
        for h in horizons
    )
