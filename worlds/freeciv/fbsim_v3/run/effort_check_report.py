#!/usr/bin/env python3
"""effort_check_report.py RUN2_PAPER EFFORT_DIR BATCHED_DIR --out JSON [--tex FILE]

The grouped-prompt reasoning-effort check (effort_check_v2.py) against the main run on the same 200 bank questions:
per model, excess Brier, bias (mean forecast minus truth), slope of forecast on truth, discrimination (Pearson),
reasoning tokens per question and per prompt, cost and unparsed answers at the run's lowest level (RUN2_PAPER rows,
tokens from BATCHED_DIR) and at the next level (EFFORT_DIR); the constant-0.5 excess on these questions; and the
Spearman correlation of each statistic with ECI across the compared models at both levels.  Writes a JSON of every
number and, with --tex, the Appendix D table in the paper's green style.
"""
import argparse, glob, gzip, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("run2_paper"); ap.add_argument("effort_dir"); ap.add_argument("batched_dir")
ap.add_argument("--out", required=True); ap.add_argument("--tex", default="")
ap.add_argument("--run1", default="", help="results/run1_2026-09-09, with scores_v1/ and elicit_medium/ (the check of 9 September)")
a = ap.parse_args()
P, E, B = Path(a.run2_paper), Path(a.effort_dir), Path(a.batched_dir)
NAMES = {"anthropic/claude-fable-5": "Fable", "anthropic/claude-sonnet-5": "Sonnet 5", "qwen/qwen3-235b-a22b": "Qwen3 235B", "openai/gpt-5": "GPT-5",
         "anthropic/claude-haiku-4.5": "Haiku 4.5", "openai/o3": "o3", "openai/gpt-5.6-luna": "GPT-5.6 Luna"}
LOW = {"anthropic/claude-fable-5": "low", "anthropic/claude-sonnet-5": "low", "qwen/qwen3-235b-a22b": "budget 1,024", "openai/gpt-5": "minimal",
       "anthropic/claude-haiku-4.5": "budget 1,024", "openai/o3": "low", "openai/gpt-5.6-luna": "low"}
NEXT = {"qwen/qwen3-235b-a22b": "budget 2,048", "anthropic/claude-haiku-4.5": "budget 2,048"}


def read_rows(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        return [json.loads(l) for l in fh if l.strip()]


si = pd.read_csv(P / "scores_v1" / "score_items.csv.gz", low_memory=False)
w = pd.read_csv(P / "results_v1" / "freeciv_results_wide.csv").set_index("model")
items = json.load(open(next(iter(glob.glob(str(E / "*" / "items.json"))))))
S = si[(si.set == "bank") & si.item.isin(items)].set_index(["model", "item"])
q_of = si[(si.set == "bank") & si.item.isin(items)].drop_duplicates("item").set_index("item").q
flat = float(((0.5 - q_of) ** 2).mean())


def stats(pairs, rows):
    """pairs: list of (q, p or None); rows: the result rows behind them (for tokens and cost)."""
    g = [(q, p) for q, p in pairs if p is not None]
    q = np.array([x[0] for x in g]); p = np.array([x[1] for x in g])
    per_q = [(r.get("tokens_reasoning") or 0) / (r.get("n_in_call") or 1) for r in rows]
    per_call = {}
    for r in rows:
        per_call[r["call_id"]] = r.get("tokens_reasoning") or 0
    return dict(n=len(g), unparsed=len(pairs) - len(g), excess=float(np.mean((p - q) ** 2)), bias=float(np.mean(p - q)), slope=float(np.polyfit(q, p, 1)[0]),
                disc=float(pearsonr(p, q)[0]), reas_per_q=float(np.mean(per_q)) if per_q else None, reas_per_call=float(np.mean(list(per_call.values()))) if per_call else None,
                cost=float(sum(r.get("cost_share") or 0 for r in rows)), calls=len(per_call))


out = {"items": len(items), "flat_excess": flat, "models": {}}
for d in sorted(E.glob("*/results.jsonl")):
    rows_n = [r for r in read_rows(d) if r["item"] in items]
    if not rows_n:
        continue
    mid = rows_n[0]["model"]
    bdir = B / d.parent.name
    if not bdir.exists():
        bdir = B / d.parent.name.replace(".", "_")   # the batched run's slugs replace dots as well as slashes
    bfile = next(iter(glob.glob(str(bdir / "results.jsonl*"))), None)
    assert bfile, f"no batched rows for {mid} under {bdir}"
    rows_l = [r for r in read_rows(bfile) if r["item"] in items and r.get("arm") == "t1"] if bfile else []
    low_pairs = [(q_of[i], S.loc[(mid, i)].p if (mid, i) in S.index and pd.notna(S.loc[(mid, i)].p) else None) for i in items]
    val = {r["item"]: r.get("value") for r in rows_n}
    next_pairs = [(q_of[i], val.get(i)) for i in items]
    out["models"][mid] = dict(name=NAMES.get(mid, mid), eci=float(w.loc[mid, "eci"]), low_level=LOW.get(mid, "low"), next_level=NEXT.get(mid, "medium"),
                              low=stats(low_pairs, rows_l), next=stats(next_pairs, rows_n))
M = out["models"]
if len(M) >= 4:
    e = [M[m]["eci"] for m in M]
    out["rho_eci"] = {k: {lvl: float(spearmanr(e, [sgn * M[m][lvl][k] for m in M])[0]) for lvl in ("low", "next")} for k, sgn in [("excess", -1), ("slope", 1), ("disc", 1), ("bias", 1)]}
# the check of 9 September on the one-question run (results/run1_2026-09-09: scores_v1 for the lowest level, elicit_medium for the next)
if a.run1:
    R1 = Path(a.run1)
    si1 = pd.read_csv(next(R1.glob("scores_v1/score_items.csv*")), low_memory=False)
    med_rows = {}
    for f in R1.glob("elicit_medium/*/results.jsonl"):
        for r in read_rows(f):
            med_rows.setdefault(r["model"], {})[r["item"]] = r
    items1 = sorted({i for rows in med_rows.values() for i in rows})
    b1 = si1[(si1.set == "bank") & si1.item.isin(items1)].set_index(["model", "item"])
    q1 = si1[(si1.set == "bank") & si1.item.isin(items1)].drop_duplicates("item").set_index("item").q
    T1 = si1[(si1.set == "bank") & si1.item.isin(items1)].drop_duplicates("item").set_index("item")["T"]
    r1 = {"items": len(items1), "in_common": len(set(items1) & set(items)), "per_horizon": {int(k): int(v) for k, v in T1.value_counts().sort_index().items()},
          "flat_excess": float(((0.5 - q1) ** 2).mean()), "models": {}}
    for mid, rows in med_rows.items():
        low = [(q1[i], b1.loc[(mid, i)].p if (mid, i) in b1.index and pd.notna(b1.loc[(mid, i)].p) else None) for i in items1]
        nxt = [(q1[i], rows[i].get("value")) for i in items1 if i in rows]
        def st1(pairs, rr):
            g = [(q, p) for q, p in pairs if p is not None]; q = np.array([x[0] for x in g]); p = np.array([x[1] for x in g])
            return dict(n=len(g), excess=float(np.mean((p - q) ** 2)), bias=float(np.mean(p - q)), slope=float(np.polyfit(q, p, 1)[0]), disc=float(pearsonr(p, q)[0]),
                        reas_per_q=float(np.mean([r.get("tokens_reasoning") or 0 for r in rr])) if rr else None, cost=float(sum(r.get("cost") or 0 for r in rr)))
        r1["models"][mid] = dict(name=NAMES.get(mid, mid), eci=float(w.loc[mid, "eci"]), low=st1(low, []), next=st1(nxt, list(rows.values())))
    M1 = r1["models"]; e1 = [M1[m]["eci"] for m in M1]
    r1["rho_eci"] = {k: {lvl: float(spearmanr(e1, [sgn * M1[m][lvl][k] for m in M1])[0]) for lvl in ("low", "next")} for k, sgn in [("excess", -1), ("disc", 1)]}
    out["run1_check"] = r1
    print(f"9 September check: {r1['items']} questions ({r1['in_common']} in common with this sample), per horizon {r1['per_horizon']}, constant 0.5 {r1['flat_excess']:.4f}")
    for m, r in sorted(M1.items(), key=lambda kv: -kv[1]["eci"]):
        print(f"  {r['name']:14s} excess {r['low']['excess']:.3f} -> {r['next']['excess']:.3f} ({r['next']['excess'] - r['low']['excess']:+.3f}); bias {r['low']['bias']:+.2f} -> {r['next']['bias']:+.2f}; reasoning/q {r['next']['reas_per_q']:.0f}; ${r['next']['cost']:.2f}")
json.dump(out, open(a.out, "w"), indent=1)
print(f"{len(items)} bank questions; constant 0.5 scores {flat:.4f}")
print(f"{'model':14s} {'ECI':>6s} | {'excess L':>8s} {'excess N':>8s} {'chg':>7s} | {'bias L':>7s} {'bias N':>7s} | {'slope L':>7s} {'slope N':>7s} | {'disc L':>6s} {'disc N':>6s} | {'reas/q L':>8s} {'reas/q N':>8s} | {'$ N':>5s} {'unp':>3s}")
for m, r in sorted(M.items(), key=lambda kv: -kv[1]["eci"]):
    L, N = r["low"], r["next"]
    print(f"{r['name']:14s} {r['eci']:6.1f} | {L['excess']:8.3f} {N['excess']:8.3f} {N['excess'] - L['excess']:+7.3f} | {L['bias']:+7.2f} {N['bias']:+7.2f} | {L['slope']:7.2f} {N['slope']:7.2f} | "
          f"{L['disc']:6.2f} {N['disc']:6.2f} | {L['reas_per_q'] or 0:8.0f} {N['reas_per_q'] or 0:8.0f} | {N['cost']:5.2f} {N['unparsed']:3d}")
if "rho_eci" in out:
    for k, v in out["rho_eci"].items():
        print(f"  Spearman with ECI across {len(M)} models, {k:6s}: low {v['low']:+.2f}  next {v['next']:+.2f}")
if a.tex:
    L = [r"% Generated by worlds/freeciv/fbsim_v3/run/effort_check_report.py (forecastbench-sim, branch freeciv-v3) from results/run2_paper and results/run2_effort; do not edit by hand.",
         r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
         r"\green{Model} & \green{ECI} & \multicolumn{2}{c}{\green{Excess Brier}} & \multicolumn{2}{c}{\green{Bias $f-q$}} & \multicolumn{2}{c}{\green{Reasoning tokens per question}} & \green{Parsed} & \green{Cost (\$)} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r" & & \green{lowest} & \green{next} & \green{lowest} & \green{next} & \green{lowest} & \green{next} & \green{next} & \green{next} \\", r"\midrule"]
    for m, r in sorted(M.items(), key=lambda kv: -kv[1]["eci"]):
        Lw, N = r["low"], r["next"]
        L.append(f"\\green{{{r['name']}}} & \\green{{{r['eci']:.1f}}} & \\green{{{Lw['excess']:.3f}}} & \\green{{{N['excess']:.3f}}} & \\green{{${Lw['bias']:+.2f}$}} & \\green{{${N['bias']:+.2f}$}} & "
                 f"\\green{{{Lw['reas_per_q'] or 0:,.0f}}} & \\green{{{N['reas_per_q'] or 0:,.0f}}} & \\green{{{100 * N['n'] / len(items):.0f}\\%}} & \\green{{{N['cost']:.2f}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    Path(a.tex).write_text("\n".join(L) + "\n"); print("wrote", a.tex)
