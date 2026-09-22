#!/usr/bin/env python
"""make_freeciv_forecast_files.py --results RUN_DIR --sets SETS_DIR --out DIR

Per-forecast files for FreeCiv in the shape of Micropolis's data/micropolis/binary_forecasts.csv (Fabio's request of
20 September 2026), from the scored rows of the run the paper reports:

  freeciv_binary_forecasts.csv   one row per model and binary question: the grouped-prompt forecast of the bank, tail
                                 and mirror sets, the single-prompt forecast of the 124 extra questions, and, for the
                                 231 bank questions that also anchor natural-conditional cells, the single-prompt
                                 forecast as well (section "bank", prompt "single"), so the two protocols can be compared
                                 on the same questions.
  freeciv_natcond_forecasts.csv  one row per model and cell: turn-1 forecast, turn-2 forecast after the revealed fact,
                                 the no-news forecast where the cell is in the control, p(Y), p(Y|X) and the scores.
  freeciv_questions.csv          one row per binary question and one per cell: text, resolution criteria, revealed fact.

Columns follow Micropolis's names where the meaning is the same (model, question_id, section, horizon, forecast,
real_prob, excess_brier, excess_bits); expected_brier is (f-q)^2 + q(1-q), the Brier score expected over the replays.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--sets", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
RES, SETS, OUT = Path(a.results), Path(a.sets), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)

si = pd.read_csv(RES / "scores_v1" / "score_items.csv.gz", low_memory=False)
SECTION = {"bank": "mid-range", "tails": "tail", "mirrors": "mirror", "extra": "extra"}
b = si[si.set.isin(SECTION)].copy()
b["section"] = b.set.map(SECTION)
b["prompt"] = np.where(b.set == "extra", "single", "grouped")
bin_rows = b[["model", "item", "world", "family", "section", "prompt", "T", "p", "q", "expected_brier", "excess_brier", "excess_bits", "parsed"]].rename(
    columns={"item": "question_id", "T": "horizon", "p": "forecast", "q": "real_prob"})

# single-prompt turn-1 forecasts of the bank questions that anchor cells (p1 of the cell rows, one per model and question)
nc = si[si.set == "natcond"].copy()
nc["question_id"] = nc.item.str.split("|").str[0]
bank_ids = set(b[b.set == "bank"].item)
single = nc[nc.question_id.isin(bank_ids)].drop_duplicates(["model", "question_id"])[["model", "question_id", "world", "family", "T", "p1", "q"]].copy()
single = single.rename(columns={"T": "horizon", "p1": "forecast", "q": "real_prob"})
single["section"] = "mid-range"; single["prompt"] = "single"; single["parsed"] = single.forecast.notna().astype(int)
f, q = single.forecast, single.real_prob
single["expected_brier"] = (f - q) ** 2 + q * (1 - q); single["excess_brier"] = (f - q) ** 2
fc, qc = f.clip(0.001, 0.999), q
single["excess_bits"] = np.where(qc > 0, qc * np.log2(qc / fc), 0) + np.where(qc < 1, (1 - qc) * np.log2((1 - qc) / (1 - fc)), 0)
bin_rows = pd.concat([bin_rows, single[bin_rows.columns]], ignore_index=True)
bin_rows = bin_rows.sort_values(["section", "prompt", "question_id", "model"]).reset_index(drop=True)
bin_rows.to_csv(OUT / "freeciv_binary_forecasts.csv", index=False, float_format="%.6g")

# natural-conditional cells
cells = {c["qid"] + "|" + c["rev_id"]: c for c in json.load(open(SETS / "natcond_600.json"))}
nat = nc[["model", "item", "question_id", "world", "family", "block", "rev_kind", "T", "p1", "p2", "p_nonews", "q", "p_given", "target", "move", "excess_t1_uncond", "stay", "excess_t2", "gain", "excess_nonews", "parsed"]].rename(
    columns={"item": "cell_id", "T": "horizon", "q": "real_prob", "p_given": "real_prob_given", "target": "true_shift", "p1": "forecast_turn1", "p2": "forecast_turn2", "p_nonews": "forecast_no_news",
             "excess_t1_uncond": "excess_brier_turn1", "stay": "excess_brier_turn1_vs_given", "excess_t2": "excess_brier_turn2", "excess_nonews": "excess_brier_no_news"})
nat = nat.sort_values(["cell_id", "model"]).reset_index(drop=True)
nat.to_csv(OUT / "freeciv_natcond_forecasts.csv", index=False, float_format="%.6g")

# question text
rows = []
for fname, section in (("bank_750.json", "mid-range"), ("tails_300.json", "tail"), ("mirrors_50.json", "mirror"), ("natcond_extra_turn1.json", "extra")):
    for r in json.load(open(SETS / fname)):
        rows.append(dict(question_id=r["id"], cell_id="", world=r["world"], family=r["family"], section=section, horizon=r["T"], real_prob=r["qAll"], real_prob_given="", revealed_fact="", question=r["text"], criteria=r["criteria"]))
for cid, c in cells.items():
    rows.append(dict(question_id=c["qid"], cell_id=cid, world=c["world"], family=c["family"], section="natcond", horizon=c["T"], real_prob=c["p"], real_prob_given=c["p_given"], revealed_fact=c["reveal"], question=c["question"], criteria=c["criteria"]))
pd.DataFrame(rows).to_csv(OUT / "freeciv_questions.csv", index=False)

# checks against the paper's wide file
w = pd.read_csv(RES / "results_v1" / "freeciv_results_wide.csv").set_index("model")
g = bin_rows[bin_rows.prompt == "grouped"].groupby(["model", "section"])
chk = {("mid-range", "bank_all_excess_brier", "excess_brier"), ("tail", "tails_all_excess_bits", "excess_bits"), ("mirror", "mirrors_all_excess_brier", "excess_brier")}
for sec, col, val in chk:
    mine = bin_rows[(bin_rows.prompt == "grouped") & (bin_rows.section == sec)].groupby("model")[val].mean().reindex(w.index)
    assert np.allclose(mine, w[col].astype(float), atol=1e-6), sec
nat_chk = nat.groupby("model").excess_brier_turn2.mean().reindex(w.index); assert np.allclose(nat_chk, w["natcond_all_excess_t2"].astype(float), atol=1e-6)
print("binary rows", len(bin_rows), bin_rows.groupby(["section", "prompt"]).question_id.nunique().to_dict())
print("natcond rows", len(nat), "cells", nat.cell_id.nunique(), "; questions file rows", len(rows))
print("per-model means reproduce freeciv_results_wide.csv for the bank, tail, mirror and natural-conditional sets")
