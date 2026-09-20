#!/usr/bin/env python3
"""results_table_v2.py RESULTS_DIR [RESULTS_DIR ...] --out OUT_DIR — the FreeCiv results in the shapes the paper reads.

Run-2 version of results_table.py: scores with score_v2 (arm t1nc for the unbatched natural-conditional turn 1),
takes ECI from models_v2.csv and ForecastBench overall from results/run1_2026-09-09/model_scores.csv, and accounts
calls and cost per API call rather than per row, since one batched call answers up to 50 questions (each row carries
the call's cost divided by the questions asked, `cost_share`).

  OUT_DIR/freeciv_results_table.md       headline table per model and Spearman rho vs ECI per column
  OUT_DIR/freeciv_results_wide.csv       one row per model: every set x horizon x metric, n_items, n_valid, calls and
                                         cost per arm (t1, t1nc, t2, nonews) and per set (cost_<set>_usd, calls_<set>)
  OUT_DIR/freeciv_results_long.csv       one row per model x set x family/block x horizon
  OUT_DIR/freeciv_results_binary.csv     Fabio's binary schema
  OUT_DIR/freeciv_results_continuous.csv Fabio's continuous schema
"""
import argparse, collections, csv, datetime, math, os, sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_v2 as sv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--models-file", default=os.path.join(HERE, "models_v2.csv"))
ap.add_argument("--capability", default=os.path.join(HERE, "..", "results", "run1_2026-09-09", "model_scores.csv"))
ap.add_argument("--impute-binary", default="0.5")
ap.add_argument("--natcond-cost-from", default="", help="run-1 wide table: natural-conditional costs for models whose natcond rows carry no cost (reused run-1 forecasts)")
a = ap.parse_args()
out = a.out
os.makedirs(out, exist_ok=True)
impute = None if a.impute_binary == "none" else float(a.impute_binary)

S, C, N = sv.load_sets()
R = sv.load_results(a.results)
rows = sv.score_items(R, S, C, N, impute=impute)
models_meta = {m["openrouter_id"]: m for m in csv.DictReader(open(a.models_file))}
import pandas as pd
run1 = pd.read_csv(a.natcond_cost_from).set_index("model") if a.natcond_cost_from else None
try:
    fb = {r["OpenRouterName"]: r for r in csv.DictReader(open(a.capability))}
    fb_by_id = {mid: fb.get(m["name"]) for mid, m in models_meta.items()}
except FileNotFoundError:
    fb_by_id = {}

# ---- calls, cost and reasoning tokens per model and arm, counted per API call ------------------------------------
set_of = {iid: it["set"] for iid, it in S.items()}
set_of.update({iid: "continuous" for iid in C})
calls = collections.defaultdict(collections.Counter)
cost = collections.defaultdict(collections.Counter)
set_calls = collections.defaultdict(lambda: collections.defaultdict(set))
set_cost = collections.defaultdict(collections.Counter)
rtok = collections.defaultdict(dict)
seen_calls = set()
for (m, item, arm), r in R.items():
    cid = r.get("call_id") or f"{arm}:{item}"           # v1-format rows are one call each
    share = r.get("cost_share") if r.get("call_id") else r.get("cost")
    cost[m][arm] += share or 0
    if (m, cid) not in seen_calls:
        seen_calls.add((m, cid))
        calls[m][arm] += 1
        if r.get("tokens_reasoning") is not None:
            rtok[m][cid] = r["tokens_reasoning"]
    s = set_of.get(item) if arm in ("t1", "t1nc") else None
    if s and arm == "t1":
        set_calls[m][s].add(cid)
        set_cost[m][s] += share or 0
    if arm == "t1nc":
        set_calls[m]["natcond_t1"].add(cid)
        set_cost[m]["natcond_t1"] += share or 0

models = sorted({r["model"] for r in rows}, key=lambda m: -float(models_meta.get(m, {}).get("eci") or 0))
HZ = {"bank": [90, 120, 150, 180, 210], "tails": [90, 120, 150, 180, 210], "mirrors": [90, 120, 150, 180, 210],
      "continuous": [90, 120, 150, 180, 210], "natcond": [120, 150, 180, 210]}


def mean(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def recov(rs):
    """Nick's share of recoverable Brier: 1 - excess(model) / excess(always 0.5)."""
    ex = mean([r["excess_brier"] for r in rs])
    base = mean([(0.5 - r["q"]) ** 2 for r in rs])
    return 1 - ex / base if base else float("nan")


wide, long_rows = [], []
for m in models:
    W = {"model": m, "eci": models_meta.get(m, {}).get("eci", ""), "fb_overall": (fb_by_id.get(m) or {}).get("FBOverall", "")}
    byset = collections.defaultdict(list)
    for r in rows:
        if r["model"] == m:
            byset[r["set"]].append(r)
    for s, hz in HZ.items():
        rs = byset.get(s, [])
        for T in hz + ["all"]:
            g = [r for r in rs if T == "all" or r["T"] == T]
            key = f"{s}_{T}"
            W[f"{key}_n_items"] = len(g)
            W[f"{key}_n_valid"] = sum(r["parsed"] for r in g)
            if s in ("bank", "tails", "mirrors"):
                W[f"{key}_excess_brier"] = mean([r["excess_brier"] for r in g])
                W[f"{key}_brier"] = mean([r["brier"] for r in g])
                W[f"{key}_recov"] = recov(g)
                if s == "tails":
                    W[f"{key}_excess_bits"] = mean([r["excess_bits"] for r in g])
                    W[f"{key}_logloss_bits"] = mean([r["logloss_bits"] for r in g])
            elif s == "continuous":
                for met in ("ncrps_global", "excess_ncrps_global", "excess_crps_norm", "crps5", "cov50", "cov90"):
                    W[f"{key}_{met}"] = mean([r.get(met) for r in g])
            else:
                for met in ("excess_t2", "stay", "gain", "excess_nonews", "move", "drift"):
                    W[f"{key}_{met}"] = mean([r.get(met) for r in g])
        subkey = "block" if s == "natcond" else "family"
        for sub in sorted({r[subkey] for r in rs}):
            for T in hz + ["all"]:
                g = [r for r in rs if r[subkey] == sub and (T == "all" or r["T"] == T)]
                if not g:
                    continue
                L = dict(model=m, eci=W["eci"], fb_overall=W["fb_overall"], question_type=s, group=sub, horizon=T, n_items=len(g), n_valid=sum(r["parsed"] for r in g))
                if s in ("bank", "tails", "mirrors"):
                    L.update(excess_brier=mean([r["excess_brier"] for r in g]), brier=mean([r["brier"] for r in g]), recov=recov(g), excess_bits=mean([r["excess_bits"] for r in g]))
                elif s == "continuous":
                    L.update(ncrps_global=mean([r.get("ncrps_global") for r in g]), excess_ncrps_global=mean([r.get("excess_ncrps_global") for r in g]), crps=mean([r.get("crps5") for r in g]))
                else:
                    L.update(excess_t2=mean([r.get("excess_t2") for r in g]), stay=mean([r.get("stay") for r in g]), gain=mean([r.get("gain") for r in g]), excess_nonews=mean([r.get("excess_nonews") for r in g]))
                long_rows.append(L)
    # the extra questions (natural-conditional value questions), scored from the unbatched rows
    ex = byset.get("extra", [])
    W["extra_all_n_items"], W["extra_all_n_valid"], W["extra_all_excess_brier"] = len(ex), sum(r["parsed"] for r in ex), mean([r["excess_brier"] for r in ex])
    W["total_calls"] = sum(calls[m].values())
    W["total_cost_usd"] = round(sum(cost[m].values()), 4)
    W["reasoning_tokens_mean"] = mean(list(rtok[m].values()))
    for arm in ("t1", "t1nc", "t2", "nonews"):
        W[f"calls_{arm}"] = calls[m][arm]
        W[f"cost_{arm}_usd"] = round(cost[m][arm], 4)
    if run1 is not None and m in run1.index and cost[m]["t2"] == 0 and calls[m]["t2"] > 0:
        # reused run-1 natural conditionals: their cost is run 1's (turn-1 share = 355 of 1,524 turn-1 calls, an estimate)
        W["cost_t2_usd"] = round(float(run1.loc[m, "cost_t2_usd"]), 4)
        W["cost_nonews_usd"] = round(float(run1.loc[m, "cost_nonews_usd"]), 4)
        W["cost_t1nc_usd"] = round(float(run1.loc[m, "cost_t1_usd"]) * 355 / 1524, 4)
        W["cost_natcond_t1_usd"] = W["cost_t1nc_usd"]
        W["natcond_cost_source"] = "run1"
        W["total_cost_usd"] = round(W["total_cost_usd"] + W["cost_t2_usd"] + W["cost_nonews_usd"] + W["cost_t1nc_usd"], 4)
    else:
        W["natcond_cost_source"] = "run2" if calls[m]["t2"] else ""
    for s in ("bank", "tails", "mirrors", "continuous", "natcond_t1"):
        W[f"calls_{s}"] = len(set_calls[m][s])          # distinct calls that carried at least one question of the set
        W[f"cost_{s}_usd"] = round(set_cost[m][s], 4)  # the set's share of those calls' cost
    wide.append(W)

keys = list(dict.fromkeys(k for w in wide for k in w))
with open(f"{out}/freeciv_results_wide.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=keys)
    wr.writeheader()
    wr.writerows(wide)
lkeys = list(dict.fromkeys(k for l in long_rows for k in l))
with open(f"{out}/freeciv_results_long.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=lkeys)
    wr.writeheader()
    wr.writerows(long_rows)
with open(f"{out}/freeciv_results_binary.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["model", "question_type", "horizon", "nforecasts", "nvalid", "brier", "expected_brier", "calibration", "excess_bits"])
    for m in models:
        for s, qt in [("bank", "regular"), ("tails", "tail"), ("mirrors", "mirror")]:
            for T in HZ[s] + ["all"]:
                g = [r for r in rows if r["model"] == m and r["set"] == s and (T == "all" or r["T"] == T)]
                if g:
                    wr.writerow([m, qt, f"T{T}" if T != "all" else "all", len(g), sum(r["parsed"] for r in g), mean([r["brier"] for r in g]), mean([r["brier"] for r in g]), mean([r["excess_brier"] for r in g]), mean([r["excess_bits"] for r in g])])
with open(f"{out}/freeciv_results_continuous.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["model", "metric", "horizon", "nforecasts", "nvalid", "CRPS", "nCRPS_global", "excess_nCRPS_global", "excess_CRPS_over_IQR"])
    for m in models:
        fams = sorted({r["family"] for r in rows if r["set"] == "continuous"}) + ["all"]
        for fam in fams:
            for T in HZ["continuous"] + ["all"]:
                g = [r for r in rows if r["model"] == m and r["set"] == "continuous" and (fam == "all" or r["family"] == fam) and (T == "all" or r["T"] == T)]
                if g:
                    wr.writerow([m, fam, f"T{T}" if T != "all" else "all", len(g), sum(r["parsed"] for r in g), mean([r.get("crps5") for r in g]) if fam != "all" else "",
                                 mean([r.get("ncrps_global") for r in g]), mean([r.get("excess_ncrps_global") for r in g]), mean([r.get("excess_crps_norm") for r in g])])

f3 = lambda x: "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"  # noqa: E731
L = ["# FreeCiv results table, run 2 (built " + datetime.date.today().isoformat() + ")", "",
     "Per model. Bank, tails, mirrors and continuous: batched at up to 50 questions per prompt (run 2). Natural conditionals: one question per "
     "prompt, turn 2 continues the conversation (run 1 rows for unchanged models, re-elicited for re-pinned ones). Truth = frequency/distribution over "
     "1,000 replays. Binary: `recov` = share of the recoverable Brier score (1 = matches the truth probabilities, 0 = always 0.5) and excess Brier (p-q)^2. "
     "Tails (q <= 0.05): excess bits = KL(q||p). Continuous: nCRPS_global = CRPS / fixed per-family constant and the excess over the replay floor. "
     "Natcond: turn-2 excess Brier vs p(Y|X), `gain` = improvement over not updating. Cost = OpenRouter USD of the calls scored here.", "",
     "| ECI | FB | model | bin T90 | bin T150 | bin T210 | **binary** (recov) | excess Brier | tails bits | mirrors recov | cont T90 | cont T210 | "
     "**continuous** (nCRPS) | excess nCRPS | **natcond** excess_t2 | gain | valid % | calls | cost $ |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for W in wide:
    valid = (W["bank_all_n_valid"] + W["tails_all_n_valid"] + W["mirrors_all_n_valid"] + W["continuous_all_n_valid"]) / max(1, W["bank_all_n_items"] + W["tails_all_n_items"] + W["mirrors_all_n_items"] + W["continuous_all_n_items"])
    L.append(f"| {W['eci']} | {W['fb_overall']} | {W['model']} | {f3(W['bank_90_recov'])} | {f3(W['bank_150_recov'])} | {f3(W['bank_210_recov'])} | **{f3(W['bank_all_recov'])}** | "
             f"{f3(W['bank_all_excess_brier'])} | {f3(W['tails_all_excess_bits'])} | {f3(W['mirrors_all_recov'])} | {f3(W['continuous_90_ncrps_global'])} | {f3(W['continuous_210_ncrps_global'])} | "
             f"**{f3(W['continuous_all_ncrps_global'])}** | {f3(W['continuous_all_excess_ncrps_global'])} | **{f3(W['natcond_all_excess_t2'])}** | {f3(W['natcond_all_gain'])} | "
             f"{100 * valid:.1f} | {W['total_calls']} | {W['total_cost_usd']:.2f} |")
L += ["", "## Capability gradient (Spearman rho vs ECI, sign-adjusted so higher = better)", "", "| column | metric | rho | p | n |", "|---|---|---|---|---|"]
for col, met, sign in [("binary", "bank_all_recov", 1), ("binary excess Brier", "bank_all_excess_brier", -1), ("tails", "tails_all_excess_bits", -1), ("mirrors", "mirrors_all_recov", 1),
                       ("continuous", "continuous_all_ncrps_global", -1), ("continuous excess", "continuous_all_excess_ncrps_global", -1), ("natcond excess_t2", "natcond_all_excess_t2", -1), ("natcond gain", "natcond_all_gain", 1)]:
    pts = [(float(W["eci"]), W[met]) for W in wide if W.get("eci") and W.get(met) == W.get(met)]
    if len(pts) > 3:
        rho, p = spearmanr([x for x, _ in pts], [y * sign for _, y in pts])
        L.append(f"| {col} | {met} | {rho:.2f} | {p:.3f} | {len(pts)} |")
tot = sum(W["total_cost_usd"] for W in wide)
L += ["", f"Total OpenRouter cost for the rows scored here: ${tot:.2f} over {len(wide)} models."]
open(f"{out}/freeciv_results_table.md", "w").write("\n".join(L) + "\n")
print("written", out, "models", len(wide))
