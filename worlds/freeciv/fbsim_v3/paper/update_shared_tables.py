#!/usr/bin/env python
"""Edit the FreeCiv cells of the tables that three worlds share, in place.

    python data/freeciv/scripts/update_shared_tables.py [--final] [--check]

Touches, FreeCiv cells only, leaving every other cell as it is:
  data/hosting_cost_table.tex            per model: FreeCiv host, calls, cost; the Total row's FreeCiv cells
  data/appendix_tables/cost_per_item.tex the FreeCiv block (items, calls per model, cost per model, cents per item)
  data/roster_table.tex                  the FreeCiv reasoning-setting column
  data/validation_table.tex              the four FreeCiv rows (rho vs ECI, CI over models, p, CI over the 8 games)
  data/validation_table_forecastbench.tex the same rows against ForecastBench overall
  data/validation_table_main.tex         the four FreeCiv rows with the eight-game interval as the one CI
and writes data/freeciv/freeciv_validation_stats.json with every statistic and its method.

Fabio's analyze_paper.py and Nick's build_metadata.py edit their own cells of the same files the same
way, so the three generators compose in any order.  Row and column positions are located by label,
never by index alone, and the script refuses to write if a table's shape is not what it expects.

--final drops the "(provisional)" qualifier from the FreeCiv row label and the comment lines (for the
rerun at 50 questions per prompt).  --check recomputes and reports differences without writing.

Sources (data/freeciv/): freeciv_results_wide.csv (headline scores, calls, cost), freeciv_results_table.md
(calls and cost per model as the run log printed them), models_v1.csv (reasoning setting, provider pin),
score_items.csv.gz (per-item rows for the cluster bootstrap over the eight anchor games), model_scores.csv.
The FreeCiv statistics reproduce the retired data/compute_validation_stats.py (same seed, same resample
count, same bootstrap order).
"""
import json
import re
import sys
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import (DATA, MODELS_FILE, N_BOOT, RUN, PAPER_DATA, REPO, RESULTS_MD, SCORE_ITEMS, SEED, TABLES, WIDE,
                     display_name, load_capability, load_items, load_wide, tex, rel)

FINAL = "--final" in sys.argv[1:]
CHECK = "--check" in sys.argv[1:]

# The four FreeCiv headline rows: (table label, wide column, score_items set, score_items column).
FREECIV_SETS = [
    ("nCRPS", "continuous_all_ncrps_global", "continuous", "ncrps_global"),
    ("Tail excess bits", "tails_all_excess_bits", "tails", "excess_bits"),
    ("Natural-conditional excess Brier", "natcond_all_excess_t2", "natcond", "excess_t2"),
    ("Bank excess Brier", "bank_all_excess_brier", "bank", "excess_brier"),
]
WORLD_LABEL = "FreeCiv" if FINAL else "\\shortstack[l]{FreeCiv\\\\(provisional)}"
FC_SETS_COST = [("Binary bank", 750), ("Tail set", 300), ("Mirror set", 50), ("Extra value questions (seed conditionals)", 124),
                ("Continuous set", 300), ("Natural conditionals, turn 2", 400), ("No-news control", 99)]


def read_lines(path):
    return path.read_text().splitlines()


def write_lines(path, lines, old):
    new = "\n".join(lines) + "\n"
    if new == old:
        print(f"unchanged  {rel(path)}")
        return
    if CHECK:
        print(f"WOULD EDIT {rel(path)}")
        return
    path.write_text(new)
    print(f"edited     {rel(path)}")


def cells_of(line):
    body = line.rstrip()
    assert body.endswith("\\\\"), line
    return [c.strip() for c in body[:-2].split("&")]


def join_cells(cells):
    return " & ".join(cells) + " \\\\"


# ---------------------------------------------------------------- statistics
def spearman_stats(x, y, rng):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    rho, p = spearmanr(x, y)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boots[i] = spearmanr(x[idx[i]], y[idx[i]])[0]
    ok = ~np.isnan(boots)
    lo, hi = np.percentile(boots[ok], [2.5, 97.5])
    return dict(n=int(n), rho=float(rho), p=float(p), ci_lo=float(lo), ci_hi=float(hi), n_boot_valid=int(ok.sum()))


def cluster_tables(items):
    tabs = {}
    for label, wcol, iset, icol in FREECIV_SETS:
        sub = items[items["set"] == iset]
        g = sub.groupby(["model", "world"])[icol].agg(["sum", "count"])
        tabs[wcol] = dict(sum=g["sum"].unstack("world"), cnt=g["count"].unstack("world"),
                          items_per_world={w: int(n) for w, n in sub.groupby("world")["item"].nunique().items()})
    return tabs


def cluster_bootstrap(tab, cap_axis, sign, rng):
    S, C = tab["sum"], tab["cnt"]
    worlds, models = list(S.columns), list(S.index)
    axis = cap_axis.reindex(models)
    keep = axis.notna().values
    Sv, Cv, ax = S.values[keep], C.values[keep], axis.values[keep]
    k = len(worlds)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draw = rng.integers(0, k, size=k)
        mult = np.bincount(draw, minlength=k).astype(float)
        boots[i] = spearmanr(ax, sign * ((Sv @ mult) / (Cv @ mult)))[0]
    ok = ~np.isnan(boots)
    lo, hi = np.percentile(boots[ok], [2.5, 97.5])
    full = Sv.sum(1) / Cv.sum(1)
    return dict(ci_lo=float(lo), ci_hi=float(hi), n_boot_valid=int(ok.sum()), n_clusters=k, clusters=worlds,
                items_per_world=tab["items_per_world"], full_sample_means=dict(zip(np.array(models)[keep], map(float, full))))


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def fmt_ci(lo, hi):
    return f"$[{lo:.2f}, {hi:.2f}]$"


def validation_rows():
    cap = load_capability()
    w = load_wide()
    items = load_items()
    tabs = cluster_tables(items)
    out = []
    for label, wcol, iset, icol in FREECIV_SETS:
        vals = w[wcol].dropna()
        vals = vals[vals.index.isin(cap.index)]
        rec = dict(score=label, column=wcol, items_set=iset, items_column=icol, lower_is_better=True,
                   per_model_score={m: float(v) for m, v in vals.sort_values().items()},
                   best_model=vals.idxmin(), best_score=float(vals.min()))
        for axis_key in ("eci", "fb"):
            ax = cap[axis_key].reindex(vals.index)
            keep = ax.notna()
            rng = np.random.default_rng(SEED)
            st = spearman_stats(ax[keep].values, -vals[keep].values, rng)
            st["models"] = sorted(vals.index[keep])
            rec[axis_key] = st
            rng = np.random.default_rng(SEED)
            cl = cluster_bootstrap(tabs[wcol], cap[axis_key], -1.0, rng)
            for m, v in cl["full_sample_means"].items():
                assert abs(v - float(vals[m])) < 1e-9, (wcol, m, v, vals[m])
            del cl["full_sample_means"]
            rec[f"{axis_key}_cluster"] = cl
        out.append(rec)
    return out


# ---------------------------------------------------------------- the tables
def update_validation(rows):
    def replace_block(path, make_cells):
        old = path.read_text()
        lines = read_lines(path)
        # locate the FreeCiv block: the row whose first cell holds a multirow with FreeCiv, plus the three rows after it
        start = next(i for i, l in enumerate(lines) if "multirow" in l and "FreeCiv" in l)
        block = lines[start:start + 4]
        assert all(l.rstrip().endswith("\\\\") for l in block), block
        assert [cells_of(l)[1] for l in block] == [r["score"] for r in rows], [cells_of(l)[1] for l in block]
        new_block = []
        for gi, r in enumerate(rows):
            cells = make_cells(r)
            cell_world = f"\\multirow{{4}}{{*}}{{{WORLD_LABEL}}}" if gi == 0 else ""
            new_block.append(join_cells([cell_world] + cells))
        lines[start:start + 4] = new_block
        if FINAL:
            lines = [re.sub(r"; FreeCiv is provisional \(one question per prompt\)", "", l) if l.startswith("%") else l for l in lines]
        write_lines(path, lines, old)

    def full(axis):
        def make(r):
            st, cl = r[axis], r[f"{axis}_cluster"]
            return [r["score"], str(st["n"]), f"${st['rho']:.2f}$", fmt_ci(st["ci_lo"], st["ci_hi"]), fmt_p(st["p"]),
                    fmt_ci(cl["ci_lo"], cl["ci_hi"])]
        return make

    def main_(r):
        st, cl = r["eci"], r["eci_cluster"]
        return [r["score"], str(st["n"]), f"${st['rho']:.2f}$", fmt_ci(cl["ci_lo"], cl["ci_hi"]), fmt_p(st["p"])]

    replace_block(PAPER_DATA / "validation_table.tex", full("eci"))
    replace_block(PAPER_DATA / "validation_table_forecastbench.tex", full("fb"))
    replace_block(PAPER_DATA / "validation_table_main.tex", main_)


def freeciv_run_records():
    """Per model: display name, host pin, reasoning setting, calls, cost, from models_v1.csv and the results table."""
    fc = pd.read_csv(MODELS_FILE)
    assert len(fc) == 24, len(fc)
    fc["dname"] = fc["name"].map(display_name)
    if "provider" not in fc.columns:      # run 2: Micropolis-registry endpoint slugs
        HOST = {"anthropic": "Anthropic", "openai": "OpenAI", "openai/default": "OpenAI", "google-ai-studio": "Google AI Studio",
                "alibaba": "Alibaba", "deepinfra": "DeepInfra", "novita": "Novita", "streamlake": "StreamLake", "deepseek": "DeepSeek"}
        fc["provider"] = [HOST.get(e, e) + (f" ({q})" if isinstance(q, str) and q else "") for e, q in zip(fc["endpoint"], fc.get("quantizations", [""] * len(fc)))]

    def setting(row):
        mode = row["reasoning_mode"]
        if mode == "effort":
            return row["effort"]
        if mode == "budget":
            return f"budget {int(row['budget']):,}"
        if mode in ("none", "disabled"):
            return "none" if mode == "none" else "disabled"
        raise ValueError(mode)

    fc["setting"] = fc.apply(setting, axis=1)
    fc["host"] = (fc["provider"].fillna("").astype(str).str.replace("|", " / ", regex=False)
                  .str.replace("Google AI Studio / Google", "Google AI Studio"))
    calls, cost = {}, {}
    for line in RESULTS_MD.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > 3 and "/" in cells[2] and cells[2] in set(fc.openrouter_id):
            calls[cells[2]] = int(cells[-2])
            cost[cells[2]] = float(cells[-1])
    assert len(calls) == 24, len(calls)
    fc["calls"] = fc.openrouter_id.map(calls)
    fc["cost"] = fc.openrouter_id.map(cost)
    return fc.set_index("dname")


def update_hosting(fc):
    path = PAPER_DATA / "hosting_cost_table.tex"
    old = path.read_text()
    lines = read_lines(path)
    hdr = next(l for l in lines if l.startswith("Model &"))
    hcells = cells_of(hdr)
    assert hcells[1] == "FreeCiv host" and hcells[2] == "Calls" and hcells[3].startswith("Cost"), hcells
    seen = 0
    for i, l in enumerate(lines):
        if not l.rstrip().endswith("\\\\") or l.startswith("Model &") or "&" not in l:
            continue
        c = cells_of(l)
        nm = c[0].replace("\\_", "_")
        if nm in fc.index:
            r = fc.loc[nm]
            c[1], c[2], c[3] = tex(r["host"]), f"{int(r['calls']):,}", f"{r['cost']:.2f}"
            lines[i] = join_cells(c)
            seen += 1
        elif c[0] == "Total":
            c[2], c[3] = f"{int(fc['calls'].sum()):,}", f"{fc['cost'].sum():.2f}"
            lines[i] = join_cells(c)
    assert seen == 24, seen
    write_lines(path, lines, old)


def update_roster(fc):
    path = PAPER_DATA / "roster_table.tex"
    old = path.read_text()
    lines = read_lines(path)
    hcells = cells_of(next(l for l in lines if l.startswith("Model &")))
    col = hcells.index("FreeCiv")
    seen = 0
    for i, l in enumerate(lines):
        if not l.rstrip().endswith("\\\\") or l.startswith("Model &"):
            continue
        c = cells_of(l)
        nm = c[0].replace("\\_", "_")
        if nm in fc.index:
            c[col] = tex(fc.loc[nm, "setting"])
            lines[i] = join_cells(c)
            seen += 1
    assert seen == 24, seen
    write_lines(path, lines, old)


RUN2_ROWS = [("Binary bank", 750, "bank"), ("Tail set", 300, "tails"), ("Mirror set", 50, "mirrors"), ("Continuous set", 300, "continuous"),
             ("Natural conditionals, turn 1 (one question per prompt)", 355, "t1nc"), ("Natural conditionals, turn 2", 400, "t2"), ("No-news control", 99, "nonews")]


def update_cost_per_item(fc, w):
    """The FreeCiv block of cost_per_item.tex.

    Run 1 (no per-set cost columns): every row is the run's mean cost per call times the set's calls, the two estimated
    cells in dark yellow.  Run 2 (cost_bank_usd etc. present): exact per-set costs; a batched call's cost is split pro rata
    over the questions it asked, so a set's cost is its share of the prompts that carried its questions, and "Calls/model"
    for a batched set is the number of prompts that carried at least one of its questions.  The natural-conditional rows
    of the 22 models that reuse run 1 carry run 1's costs (results_table_v2.py --natcond-cost-from)."""
    path = TABLES / "cost_per_item.tex"
    old = path.read_text()
    lines = read_lines(path)
    start = next(i for i, l in enumerate(lines) if l.startswith("FreeCiv &"))
    end = start
    while not lines[end].startswith("\\midrule") and not lines[end].startswith("\\bottomrule"):
        end += 1
    n_models = len(w)
    total_calls, total_cost = int(w["total_calls"].sum()), float(w["total_cost_usd"].sum())
    new = []
    if "cost_bank_usd" in w.columns:
        for k, (lab, n, key) in enumerate(RUN2_ROWS):
            ccol = f"cost_{key}_usd"
            cpm = float(w[ccol].sum()) / n_models
            calls = int(round(w[f"calls_{key}"].sum() / n_models))
            new.append(join_cells(["FreeCiv" if k == 0 else "", lab, f"{n:,}", f"{calls:,}", f"{cpm:.2f}", f"{100 * cpm / n:.1f}"]))
        items = sum(n for _, n, _ in RUN2_ROWS)
        new.append(join_cells(["", "\\emph{all sets}", f"{items:,}", f"{int(round(total_calls / n_models)):,}", f"{total_cost / n_models:.2f}", f"{100 * total_cost / n_models / items:.1f}"]))
    else:
        per_call = total_cost / total_calls
        for k, (lab, n) in enumerate(FC_SETS_COST):
            cpm = n * per_call
            new.append(join_cells(["FreeCiv" if k == 0 else "", lab, f"{n:,}", f"{n:,}", f"\\yellow{{{cpm:.2f}}}", f"\\yellow{{{100 * per_call:.1f}}}"]))
        design = sum(n for _, n in FC_SETS_COST)
        new.append(join_cells(["", "\\emph{all sets}", f"{design:,}", f"{design:,}", f"{total_cost / n_models:.2f}", f"{100 * per_call:.1f}"]))
    lines[start:end] = new
    write_lines(path, lines, old)


def main():
    rows = validation_rows()
    fc = freeciv_run_records()
    w = load_wide()
    update_validation(rows)
    update_hosting(fc)
    update_roster(fc)
    update_cost_per_item(fc, w)
    cap = load_capability()
    both = cap.dropna()
    rng = np.random.default_rng(SEED)
    out = dict(generated=str(date.today()), script="data/freeciv/scripts/update_shared_tables.py", seed=SEED, n_boot=N_BOOT,
               final=FINAL, sources={"wide": rel(WIDE), "score_items": rel(SCORE_ITEMS),
                                     "results_md": rel(RESULTS_MD), "models_file": rel(MODELS_FILE), "run": RUN},
               sign_convention="rho between the capability score and (-1 x score); positive = more capable models score better",
               model_bootstrap=f"percentile bootstrap over the (capability, score) model pairs, {N_BOOT:,} resamples, "
                               f"numpy default_rng({SEED}) re-seeded per row and axis",
               cluster_bootstrap="resample the 8 anchor worlds with replacement, recompute every model's mean score from "
                                 "score_items over the items of the drawn worlds (with multiplicity), then Spearman rho",
               eci_vs_fb_spearman=spearman_stats(both.eci.values, both.fb.values, rng),
               rows=rows,
               run=dict(models=24, total_calls=int(w["total_calls"].sum()), total_cost_usd=float(w["total_cost_usd"].sum()),
                        per_model={n: dict(host=r["host"], setting=r["setting"], calls=int(r["calls"]), cost=float(r["cost"]))
                                   for n, r in fc.iterrows()}))
    if not CHECK:
        (DATA / "freeciv_validation_stats.json").write_text(json.dumps(out, indent=1, default=float))
        print("wrote", rel(DATA / "freeciv_validation_stats.json"))
    for r in rows:
        e, c = r["eci"], r["eci_cluster"]
        print(f"  {r['score']:36s} rho {e['rho']:5.2f} {fmt_ci(e['ci_lo'], e['ci_hi'])} p {e['p']:.3f} games {fmt_ci(c['ci_lo'], c['ci_hi'])}")


if __name__ == "__main__":
    main()
