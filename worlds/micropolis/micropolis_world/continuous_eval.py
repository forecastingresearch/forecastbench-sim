"""Shared pieces of the continuous eval: response cache, dataset, scoring.

The eval is split across three scripts — run_eval_continuous.py gathers model
responses, analyze_continuous.py scores them, plot_forecasts.py draws them —
and this module holds what more than one of them needs. The dataset written by
the first is the only thing the other two read, so they never re-simulate or
re-prompt.
"""

import json
import os
import re
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fbsim_core.metrics import compute_crps

from . import messages as msg
from . import module_globals as g
from .city_sim import CitySimulation
from .config import Config, scenarios_from
from .gather import (  # noqa: F401 - re-exported for the scripts and tests
    EvalPaths,
    batch_id_for,
    group_into_batches,
    write_dataset,
)
from .cache_keys import prompt_hash  # noqa: F401 - re-exported
from .usage import load_usage, save_usage  # noqa: F401 - re-exported for the scripts

# Batch prompts and raw model responses are cached here, shared across every
# label: the cache filename already carries the prompt's hash (see
# batch_dir), so two labels asking an identical prompt reuse the same cached
# response rather than paying for it twice. The root is read through
# g.DATA_DIR at call time, never copied at import: a config's 'data_dir'
# rebinds it after every import has run.
SUBDIR = "continuous"


def out_dir() -> Path:
    return g.DATA_DIR / SUBDIR


def label_dir(label: str) -> Path:
    """Where one label's dataset, plots and reports are written.

    Kept apart per label — unlike out_dir()'s shared cache — so runs made under
    different prompt variants never overwrite each other's output.
    """
    return out_dir() / label


def data_path(label: str) -> Path:
    return label_dir(label) / "data.json"


def plots_path(label: str) -> Path:
    return label_dir(label) / "plots"


from fbsim_benchmark.scoring.micropolis import (
    PERCENTILE_LEVELS, quantile_array, crps_distribution, crps_floor,
)

# What each metric can be, from the engine. The four averages are means over
# byte-valued map cells (0-255; scan.cpp), traffic then scaled by 2.4
# (evaluate.cpp); population cannot be negative. A forecast whose median lies
# outside is not a forecast of the metric at all — a decimal written as
# thousands, a population copied onto every answer line — and out_of_range
# lets the dataset builder treat it as unparseable. None is no upper bound.
METRIC_RANGES: dict[str, tuple[float, float | None]] = {
    "cityPop": (0.0, None),
    "trafficAverage": (0.0, 255 * 2.4),
    "pollutionAverage": (0.0, 255.0),
    "crimeAverage": (0.0, 255.0),
    "landValueAverage": (0.0, 255.0),
}


def out_of_range(metric: str, percentiles: dict[str, float]) -> str | None:
    """Why the forecast's median cannot be a value of `metric`; None if it can.

    Only the median is checked: a tail percentile a little past a bound is a
    wide interval, not a wrong quantity, and CRPS already charges for it.
    """
    bounds = METRIC_RANGES.get(metric)
    if bounds is None:
        return None
    lo, hi = bounds
    p50 = percentiles["p50"]
    if p50 < lo or (hi is not None and p50 > hi):
        top = f"{hi:g}" if hi is not None else "inf"
        return f"median {p50:g} outside the metric's range [{lo:g}, {top}]"
    return None


def attach_outcomes(corpus: list[dict]) -> int:
    """Put each question's continuation outcomes and CRPS floor on its dict.

    Sets "outcomes" (an array) and "crps_floor" on every corpus question the
    ground truth covers, which is what score_forecasts reads to compute the
    excess CRPS; a question without them gets no excess score. Returns how
    many questions were left without. Errors, like every ground-truth loader,
    when a file the corpus needs is missing.
    """
    import numpy as np

    from .ground_truth import load_outcomes

    missing = 0
    for c, values in zip(corpus, load_outcomes(corpus).values(), strict=True):
        if values is None:
            missing += 1
            continue
        c["outcomes"] = np.asarray(values, dtype=float)
        c["crps_floor"] = crps_floor(c["outcomes"])
    return missing


@dataclass(frozen=True)
class ResponseId:
    model_id: str
    question_id: str


@dataclass(frozen=True)
class Response:
    actual: float
    percentiles: dict[str, float] | None
    response_text: str | None = None


Responses = dict[ResponseId, Response]


# The continuous eval's cache layout. The helpers below are kept as
# module-level functions because the scripts and tests import them by name;
# PATHS is what the shared gather machinery is handed.
PATHS = EvalPaths(SUBDIR)


def batch_dir(batch_id: str) -> Path:
    """Where a batch's prompt and raw model responses are cached.

    Holds prompt-{hash}.txt plus one response-{model}-{hash}.txt per model that
    has answered it (see prompt_path and response_path), named the same way as
    the knowledge eval's cache. The hash is the prompt's content, so a batch
    directory can hold more than one prompt variant — a template or history_freq
    change simply adds new files alongside the old ones instead of colliding
    with or invalidating them. Deleting the directory re-gathers the batch from
    scratch on the next run.
    """
    return PATHS.batch_dir(batch_id)


def prompt_path(batch_id: str, phash: str) -> Path:
    return PATHS.prompt_path(batch_id, phash)


def response_path(batch_id: str, model_id: str, phash: str) -> Path:
    return PATHS.response_path(batch_id, model_id, phash)


def usage_path(batch_id: str, model_id: str, phash: str) -> Path:
    """Tokens and cost of the call that produced the matching response file."""
    return PATHS.usage_path(batch_id, model_id, phash)


def save_dataset(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    path: Path,
) -> Path:
    """Write the corpus and this run's percentile forecasts to `path`.

    The continuous eval's shape of gather.write_dataset: one "percentiles"
    entry per (question, model) that was gathered.
    """
    forecasts = [
        {
            "model_id": model_id,
            "question_id": c["question_id"],
            "percentiles": r.percentiles,
        }
        for c in corpus
        for model_id in model_names
        for r in [responses.get(ResponseId(model_id, c["question_id"]))]
        if r is not None
    ]
    return write_dataset(corpus, forecasts, model_names, path)


def load_dataset(path: Path) -> tuple[list[dict], Responses, list[str]]:
    """Read back what save_dataset wrote, as (corpus, responses, model_names).

    Returns the same shapes the gathering script works with, so the analysis and
    plotting code is identical whether it was handed live results or a file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/run_eval_continuous.py first"
        )
    data = json.loads(path.read_text())
    corpus = data["questions"]
    actual = {c["question_id"]: c["value"] for c in corpus}
    responses: Responses = {
        ResponseId(f["model_id"], f["question_id"]): Response(
            actual=actual[f["question_id"]],
            percentiles=f["percentiles"],
        )
        for f in data["forecasts"]
    }
    return corpus, responses, data["models"]


def scenario_history(scenario_id: str, seed: int) -> list[dict] | None:
    """The cached log rows of a scenario run, or None if it isn't on disk.

    Indexed by turn: row i is turn i, matching how build_corpus resolves a
    question at snapshot_turn + horizon. Reads the cache only — never simulates —
    so the analysis scripts stay offline and cost nothing.

    The scenario id encodes city, disasters and seed, which is what names the log
    file, so a run can be found from a corpus entry alone. `seed` is taken as an
    argument rather than parsed back out of the id: the id is built by
    CitySimulation and this should not depend on how it is spelled.
    """
    city, _, rest = scenario_id.partition("_")
    if not rest:
        return None
    sim = CitySimulation(
        city_name=city, seed=seed, disasters=rest.startswith("disasters")
    )
    try:
        sim.load_from_disk()
    except FileNotFoundError:
        return None
    return sim.log_data


def git_commit_note(repo: Path | None = None, count_untracked: bool = True) -> str:
    """The short commit hash HEAD is at, flagged '(dirty)' if the tree differs.

    Recorded at the top of every analysis report: the tables and figures are
    computed from code, and a report generated mid-edit should say so rather
    than imply it came from a clean, identifiable commit.

    `repo` is the working directory to ask about, defaulting to this process's.
    `count_untracked` treats untracked files as making the tree dirty, which is
    right for this repo — a new, not-yet-added source file can change the
    numbers — and wrong for the engine checkout, where build and run artifacts
    sit untracked permanently and would pin the flag on forever.
    """
    status = ["git", "status", "--porcelain"]
    if not count_untracked:
        status.append("--untracked-files=no")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                status, capture_output=True, text=True, check=False, cwd=repo
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return "unknown (not a git checkout, or git is unavailable)"
    return f"{commit} (dirty)" if dirty else commit


def engine_commit_note() -> str:
    """git_commit_note for the MicropolisCore checkout the sims are run with.

    The engine decides the trajectories every question resolves against, so it
    is as much a part of a result's provenance as this repo is. Untracked files
    are ignored: the checkout carries build output and sim-runs directories that
    are not part of the engine's source.
    """
    return git_commit_note(g.engine_app_path(), count_untracked=False)


class MdReport:
    """Accumulates one script run's tables, figures and commentary as Markdown.

    Every print_*/plot_* function that used to write straight to stdout instead
    appends to a report instance passed in as an argument, so a run's whole
    output ends up in one .md file rather than scattered across the console.
    Tables are kept as the fixed-width text they were already formatted into —
    reformatting them as native Markdown tables would mean redoing the column
    alignment logic for no reader benefit — wrapped in a code fence so a
    Markdown viewer renders the alignment as written.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []

    def heading(self, text: str, level: int = 2) -> None:
        self._parts.append(f"{'#' * level} {text}")

    def text(self, text: str = "") -> None:
        self._parts.append(text)

    def table(self, text: str) -> None:
        """A preformatted, fixed-width table or block, wrapped in a code fence."""
        self._parts.append(f"```\n{text}\n```")

    def image(self, path: Path, caption: str = "") -> None:
        """Embed a figure, linked relative to wherever write() ends up putting it.

        `path` is stored absolute and resolved to a relative link in write(),
        rather than assumed to be one fixed number of directories below the
        report — analyze_continuous.py's figures sit directly under
        plots/, but analyze_baseline_skill.py's sit one level deeper, under
        plots/with_baseline/, so a hardcoded "plots/{name}" link would be wrong
        for the second caller.
        """
        alt = caption or path.stem
        self._parts.append(f"![{alt}]({path.resolve()})")
        if caption:
            self._parts.append(f"*{caption}*")

    def render(self, base_dir: Path) -> str:
        """Join the accumulated parts, rewriting image links relative to base_dir."""
        text = "\n\n".join(self._parts) + "\n"
        return re.sub(
            r"!\[([^\]]*)\]\(([^)]+)\)",
            lambda m: f"![{m.group(1)}]({os.path.relpath(m.group(2), base_dir)})",
            text,
        )

    def write(self, path: Path, title: str) -> Path:
        """Write the accumulated report to `path`, with a title and commit notes.

        Both repos are named: this one produced the tables, and the engine
        checkout produced the trajectories they score against, so neither alone
        identifies what a report came from.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# {title}\n\n"
            f"generated at forecastbench-sim commit {git_commit_note()}\n"
            f"Micropolis engine commit {engine_commit_note()}"
        )
        path.write_text(header + "\n\n" + self.render(path.parent))
        return path


class DatasetError(Exception):
    """The dataset on disk doesn't cover what the config asked for."""


def scenario_ids_from(
    cfg: Config,
    seed: int,
    cities: list[str] | None = None,
    disasters: list[bool] | None = None,
) -> list[str]:
    """The scenario ids a config's cities x disasters cross product names.

    Built through CitySimulation so the ids match the ones build_corpus wrote;
    constructing one runs nothing, it only holds the parameters. `cities` and
    `disasters` override the config's lists, for the matching command-line flags.
    """
    return [
        CitySimulation(city_name=city, seed=seed, disasters=dis).get_id_str()
        for city, dis in scenarios_from(cfg, cities, disasters)
    ]


def select_for_config(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    cfg: Config,
    seed: int,
    *,
    cities: list[str] | None = None,
    disasters: list[bool] | None = None,
    models: list[str] | None = None,
    rerun_hint: str = "scripts/run_eval_continuous.py",
    incomplete: bool = False,
) -> tuple[list[dict], Responses, list[str]]:
    """Narrow a dataset to what `cfg` asks for, or fail saying what is missing.

    Lets one gathered dataset serve many views — a subset of models, cities or
    horizons — without re-prompting. Anything the config names that the dataset
    lacks is an error rather than a silently smaller table, since a missing
    model or city would otherwise look like a legitimately empty result.
    `cities`, `disasters` and `models` override the config's lists, for the
    matching command-line flags; keyword-only, since three same-shaped list
    arguments in a row are easy to pass in the wrong order. `rerun_hint` names
    the gathering script in the error messages — the binary eval reuses this
    with its own script.

    `incomplete` downgrades the never-gathered rows from an error to a warning
    and keeps every (question, model) pair that *was* gathered, for testing the
    analysis scripts against a run still missing batches. The selection is then
    ragged: each model is scored on the questions it answered, so per-model
    aggregates cover different question sets and are not strictly comparable
    with one another. The warning says so and lists each model's coverage. This
    is a testing aid, not a reporting mode.

    It does not relax the coverage check above: a model, city or horizon the
    dataset knows nothing about is a config/dataset mismatch rather than sparse
    data. A selected model with no gathered forecast at all is likewise an
    error, since every figure it appears in would be empty.
    """
    wanted_models = cfg.get_models(models)
    wanted_scenarios = scenario_ids_from(cfg, seed, cities, disasters)
    wanted_snapshots = cfg.get_int_list("snapshot_turns")
    wanted_horizons = cfg.get_int_list("horizons")

    missing = []
    for name, wanted, present in [
        ("models", wanted_models, set(model_names)),
        ("cities/disasters", wanted_scenarios, {c["scenario_id"] for c in corpus}),
        ("snapshot_turns", wanted_snapshots, {c["snapshot_turn"] for c in corpus}),
        ("horizons", wanted_horizons, {c["horizon"] for c in corpus}),
    ]:
        absent = [w for w in wanted if w not in present]
        if absent:
            missing.append(
                f"  {name}: {', '.join(str(a) for a in absent)}\n"
                f"    dataset has: {', '.join(str(p) for p in sorted(present, key=str))}"
            )
    if missing:
        raise DatasetError(
            "the dataset does not cover this config:\n"
            + "\n".join(missing)
            + f"\n  re-run {rerun_hint} with this config to gather it"
        )

    selected_scenarios = set(wanted_scenarios)
    selected_snapshots = set(wanted_snapshots)
    selected_horizons = set(wanted_horizons)
    selected_corpus = [
        c
        for c in corpus
        if c["scenario_id"] in selected_scenarios
        and c["snapshot_turn"] in selected_snapshots
        and c["horizon"] in selected_horizons
    ]

    # Every selected question needs a row for every selected model. A model that
    # answered unusably still has a row, with null percentiles, so a genuinely
    # absent row means that pair was never gathered.
    ungathered = [
        (model_id, c["question_id"])
        for c in selected_corpus
        for model_id in wanted_models
        if ResponseId(model_id, c["question_id"]) not in responses
    ]
    if ungathered and not incomplete:
        shown = ", ".join(f"{m} / {q}" for m, q in ungathered[:3])
        more = f" (+{len(ungathered) - 3} more)" if len(ungathered) > 3 else ""
        raise DatasetError(
            f"the dataset is missing {len(ungathered)} forecast(s) the config asks "
            f"for: {shown}{more}\n"
            f"  re-run {rerun_hint} with this config to gather them\n"
            f"  or pass --incomplete to score only the fully gathered questions"
        )
    if ungathered:
        # Every pair that was gathered is kept, so the selection is ragged:
        # each model is scored on the questions it actually answered. That is
        # what makes this a testing aid and not a reporting mode — per-model
        # aggregates are then means over different question sets, so the
        # columns are not strictly comparable with each other. The per-model
        # counts below are printed for exactly that reason.
        by_model = Counter(m for m, _ in ungathered)
        nq = len(selected_corpus)
        msg.warn(
            f"--incomplete: {len(ungathered)} of {nq * len(wanted_models)} "
            f"(question, model) pair(s) were never gathered; scoring the rest. "
            "Per-model figures cover different question sets and are not "
            "directly comparable."
        )
        for model_id in wanted_models:
            n = by_model.get(model_id, 0)
            if n:
                msg.plain(
                    f"  {model_id}: {nq - n} of {nq} question(s)", color=msg.YELLOW
                )
        msg.plain(
            f"  re-run {rerun_hint} with this config to gather them",
            color=msg.YELLOW,
        )
        starved = [m for m in wanted_models if by_model.get(m, 0) >= nq]
        if starved:
            raise DatasetError(
                "no forecast at all was gathered for: " + ", ".join(starved) + "\n"
                f"  re-run {rerun_hint} with this config, or drop them from --models"
            )

    # Absent pairs are skipped rather than indexed, so an --incomplete
    # selection can be ragged; without the flag the loop above has already
    # proved every pair is present.
    selected_responses = {
        rid: responses[rid]
        for c in selected_corpus
        for model_id in wanted_models
        if (rid := ResponseId(model_id, c["question_id"])) in responses
    }
    return selected_corpus, selected_responses, wanted_models
