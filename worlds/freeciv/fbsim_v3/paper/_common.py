"""Shared paths, names and helpers for the FreeCiv generators of the ICLR paper.

Two roots:
  * this repository (dev): worlds/freeciv/fbsim_v3/ holds the run, the sets and the results.  Inputs
    are read from results/<run>/ (run = $FBSIM_RUN, default run1_2026-09-09).
  * the paper checkout (prod, the Overleaf mirror): $FBSIM_PAPER_ROOT, the folder that holds main.tex.
    Outputs go to its data/appendix_tables/ (tables), figures/ (figures) and data/freeciv/ (backing
    CSVs and the JSON side files that record every number).  Nothing else in the paper is touched,
    and no cell owned by another world is written: Micropolis tables, figures and macros come from
    Fabio's analyze_paper.py, StarSim tables and figures from data/starsim/scripts/.  The shared tables
    (hosting_cost_table, cost_per_item, roster_table, validation_table*) are edited in place, FreeCiv
    cells only, by update_shared_tables.py.

    python worlds/freeciv/fbsim_v3/paper/<script>.py --paper-root /path/to/paper   (or export FBSIM_PAPER_ROOT)
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FBSIM_V3 = Path(__file__).resolve().parents[1]           # worlds/freeciv/fbsim_v3
REPO_DEV = FBSIM_V3.parents[1]                             # the forecastbench-sim checkout
RUN = os.environ.get("FBSIM_RUN", "run1_2026-09-09")
RESULTS = FBSIM_V3 / "results" / RUN


def _paper_root():
    for i, a in enumerate(sys.argv[1:]):
        if a == "--paper-root" and i + 2 < len(sys.argv) + 0:
            return Path(sys.argv[i + 2]).expanduser().resolve()
        if a.startswith("--paper-root="):
            return Path(a.split("=", 1)[1]).expanduser().resolve()
    env = os.environ.get("FBSIM_PAPER_ROOT")
    if not env:
        sys.exit("paper checkout not given: pass --paper-root DIR or export FBSIM_PAPER_ROOT (the folder that holds main.tex)")
    return Path(env).expanduser().resolve()


REPO = _paper_root()                                        # kept under this name: the scripts print paths relative to it
if not (REPO / "main.tex").exists():
    sys.exit(f"{REPO} has no main.tex; is it the paper checkout?")
DATA = REPO / "data" / "freeciv"                           # backing files and JSON side files in the paper
TABLES = REPO / "data" / "appendix_tables"
FIGURES = REPO / "figures"
PAPER_DATA = REPO / "data"
DATA.mkdir(parents=True, exist_ok=True)

# Inputs (dev).  score_items.csv.gz holds the per-item rows of the scored run; pandas reads the gzip directly.
WIDE = RESULTS / "results_v1" / "freeciv_results_wide.csv"
RESULTS_MD = RESULTS / "results_v1" / "freeciv_results_table.md"
SCORES_MD = RESULTS / "scores_v1" / "SCORES.md"
SCORE_ITEMS = RESULTS / "scores_v1" / "score_items.csv.gz"
FAMILY_HORIZON = RESULTS / "family_horizon_scores.csv"
RELIABILITY = RESULTS / "reliability_bands.csv"
MODEL_SCORES = RESULTS / "model_scores.csv"               # 24 shared models: ECI, ForecastBench overall
SLUG_MAP = RESULTS / "model_scores_with_slugs.csv"        # Fabio's copy with OpenRouter slugs (26 rows)
MODELS_V1 = FBSIM_V3 / "run" / "models_v1.csv"            # the run's per-model settings and provider pins
NORM_CONSTANTS = FBSIM_V3 / "sets" / "draw_v1" / "continuous_norm_constants.json"
COMPOSITION = FBSIM_V3 / "sets" / "draw_v1" / "COMPOSITION.md"

# The backing files copied into the paper's data/freeciv/ so that every number there traces to a file.
BACKING_FILES = [WIDE, RESULTS_MD, SCORES_MD, FAMILY_HORIZON, RELIABILITY, MODEL_SCORES, MODELS_V1, NORM_CONSTANTS, COMPOSITION]

SEED = 2026
N_BOOT = 10_000

SHORT = {
    "anthropic/claude-fable-5": "Fable",
    "anthropic/claude-opus-5": "Opus 5",
    "openai/gpt-5.6-sol": "GPT-5.6 Sol",
    "openai/gpt-5.5": "GPT-5.5",
    "google/gemini-3.7-flash": "Gemini 3.7 Flash",
    "openai/gpt-5.6-luna": "GPT-5.6 Luna",
    "anthropic/claude-sonnet-5": "Sonnet 5",
    "deepseek/deepseek-v4-flash-0731": "DeepSeek V4 Flash",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "openai/gpt-5": "GPT-5",
    "openai/o3": "o3",
    "openai/gpt-5.4-nano": "GPT-5.4 Nano",
    "openai/o4-mini": "o4-mini",
    "openai/gpt-5-mini": "GPT-5 Mini",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "qwen/qwen3.5-flash-02-23": "Qwen3.5 Flash",
    "anthropic/claude-haiku-4.5": "Haiku 4.5",
    "openai/gpt-5-nano": "GPT-5 Nano",
    "moonshotai/kimi-k2": "Kimi K2",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "qwen/qwen3-235b-a22b": "Qwen3 235B",
    "openai/gpt-4.1": "GPT-4.1",
    "deepseek/deepseek-chat": "DeepSeek V3",
    "meta-llama/llama-4-scout": "Llama 4 Scout",
    "openai/gpt-4.1-mini": "GPT-4.1 Mini",
    "openai/gpt-4.1-nano": "GPT-4.1 Nano",
}


def rel(p):
    """A path as the paper or this repository would print it."""
    p = Path(p)
    for root in (REPO, REPO_DEV):
        try:
            return str(p.relative_to(root))
        except ValueError:
            continue
    return str(p)


def display_name(openrouter_name: str) -> str:
    """The name used in the shared tables: the OpenRouter display name without its vendor prefix."""
    return openrouter_name.split(": ", 1)[-1].strip()


def name(slug, dagger=False):
    return SHORT[slug] + ("$^{\\dagger}$" if dagger else "")


def tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt(x, dp=3, pct=False):
    """Format a number for a LaTeX cell; math-mode minus; '--' for missing."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if pct:
        return f"{100 * x:.{dp}f}"
    s = f"{x:.{dp}f}"
    return s.replace("-", "$-$")


def write_tabular(path, colspec, header_lines, rows, tabcolsep="3pt", comment="", script="make_freeciv_tables.py"):
    """Write a bare booktabs tabular. header_lines: list of LaTeX header rows (without \\\\)."""
    lines = [f"% Generated by worlds/freeciv/fbsim_v3/paper/{script} (forecastbench-sim, branch freeciv-v3) from "
             f"results/{RUN}; do not edit by hand."]
    if comment:
        for c in comment.strip().splitlines():
            lines.append("% " + c)
    lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    for h in header_lines:
        lines.append(h if h.strip().startswith("\\cmidrule") else h + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"wrote {rel(path)} ({len(rows)} rows)")


def parse_md_tables(text):
    """Return {heading: DataFrame} for every pipe table in a Markdown file, keyed by the nearest heading."""
    tables, cur, rows = {}, None, []

    def flush():
        if cur is not None and rows:
            hdr = rows[0]
            body = [r for r in rows[1:] if len(r) == len(hdr)]
            tables[cur] = pd.DataFrame(body, columns=hdr)

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            cur, rows = line.lstrip("#").strip(), []
        elif line.startswith("|"):
            if re.match(r"^\|\s*-", line):
                continue
            cells = [c.strip().replace("**", "").replace("⚠", "").strip() for c in line.strip().strip("|").split("|")]
            rows.append(cells)
        else:
            if rows:
                flush()
                rows = []
    flush()
    return tables


def spearman_signed(x, y, lower_is_better=True, seed=SEED, n_boot=N_BOOT):
    """Spearman rho of capability x against score y, signed so positive = more capable models score better,
    with a percentile bootstrap over the paired (model) observations."""
    y = -np.asarray(y, float) if lower_is_better else np.asarray(y, float)
    x = np.asarray(x, float)
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([spearmanr(x[i], y[i])[0] for i in idx])
    boots = boots[~np.isnan(boots)]
    return float(rho), float(p), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], int(len(boots))


def load_capability():
    """ECI and ForecastBench overall for the 24 shared models, indexed by OpenRouter slug.

    model_scores.csv (24 rows) carries the values; Fabio's copy carries the slugs.  The two must agree."""
    ms = pd.read_csv(MODEL_SCORES)
    fab = pd.read_csv(SLUG_MAP)
    m = fab.merge(ms, left_on="Name", right_on="OpenRouterName", how="inner", suffixes=("_fab", "_dl"))
    assert len(m) == 24, len(m)
    assert (m.ECI_fab == m.ECI_dl).all()
    assert (m.FBOverall_fab.fillna(-1) == m.FBOverall_dl.fillna(-1)).all()
    cap = m[["slug", "Name", "ECI_dl", "FBOverall_dl"]].rename(
        columns={"slug": "model", "Name": "openrouter_name", "ECI_dl": "eci", "FBOverall_dl": "fb"}).set_index("model")
    assert cap.fb.notna().sum() == 17
    return cap


def load_wide():
    return pd.read_csv(WIDE).set_index("model")


def load_items():
    return pd.read_csv(SCORE_ITEMS, low_memory=False)


def models_by_eci(fw):
    """The 24 models in descending ECI, as the tables list them."""
    return sorted(fw.index, key=lambda s: -fw.loc[s, "eci"])
