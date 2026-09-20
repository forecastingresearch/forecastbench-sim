#!/usr/bin/env python3
"""rebuild_natcond_rows.py — recreate run 1's natural-conditional rows from the per-item score file.

  python rebuild_natcond_rows.py --score-items ../results/run1_2026-09-09/scores_v1/score_items.csv.gz
                                 --out ../results/run2_natcond/from_run1 [--exclude id1,id2]

Run 2 batches the bank, tail, mirror and continuous questions but keeps the natural-conditional arm exactly as
run 1 asked it: one question per prompt at turn 1, then the revealed fact, then the question again.  For the
models whose provider did not change, run 1's natural-conditional forecasts are therefore reused as they are.
The raw responses of 23 models are in the Drive archive, not on this machine, but every forecast the scorer
needs is in score_items.csv: p1 (the turn-1 forecast of the cell's question, identical to the bank or extra
row's p for that question), p2 (turn 2) and p_nonews (the control).  This script writes those forecasts back
as results.jsonl rows in the harness format, one per (model, item, arm), so score_v1.py scores them unchanged:
  t1nc    item = question id (355 per model), value = p of the bank/extra row (the arm name keeps these
          unbatched turn-1 forecasts apart from the batched turn-1 row of the same bank question in run 2)
  t2      item = qid|rev_id (400 per model), value = p2
  nonews  item = question id (99 per model), value = p_nonews
Rows carry source="run1:score_items" and no response text.  Models named in --exclude (the re-pinned ones,
whose natural conditionals are re-elicited) are skipped.
"""
import argparse, json, os
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--score-items", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--exclude", default="", help="comma list of OpenRouter ids to leave out")
ap.add_argument("--sets", default=str(Path(__file__).resolve().parent.parent / "sets" / "draw_v1"))
a = ap.parse_args()
excl = {x.strip() for x in a.exclude.split(",") if x.strip()}
si = pd.read_csv(a.score_items, low_memory=False)
nc = json.load(open(os.path.join(a.sets, "natcond_600.json")))
cells = {c["qid"] + "|" + c["rev_id"]: c for c in nc}
qids = {c["qid"] for c in nc}
nn_q = {c["qid"] for c in nc if c["control_no_news"]}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
n_rows, n_models = 0, 0
with open(out / "results.jsonl", "w") as f:
    for model, g in si.groupby("model"):
        if model in excl:
            continue
        n_models += 1
        t1 = g[g["set"].isin(["bank", "extra"]) & g["item"].isin(qids)]
        assert len(t1) == len(qids) == 355, (model, len(t1))
        ncg = g[g["set"] == "natcond"]
        assert len(ncg) == 400, (model, len(ncg))
        # p1 in the cell rows equals the bank/extra p of the question wherever both parsed
        chk = ncg.assign(qid=ncg["item"].str.split("|").str[0]).merge(t1[["item", "p"]].rename(columns={"item": "qid", "p": "p_t1"}), on="qid")
        both = chk["p1"].notna() & chk["p_t1"].notna()
        assert ((chk.loc[both, "p1"] - chk.loc[both, "p_t1"]).abs() < 1e-9).all(), model
        def row(item, arm, value, world, extra=None):
            r = dict(model=model, item=item, arm=arm, kind="bin", world=world, reasoning="low",
                     value=None if pd.isna(value) else float(value), parse_mode="run1" if not pd.isna(value) else None,
                     source="run1:score_items", **(extra or {}))
            f.write(json.dumps(r) + "\n")
        for _, r in t1.iterrows():
            row(r["item"], "t1nc", r["p"], r["world"])
        nn_done = set()
        for _, r in ncg.iterrows():
            c = cells[r["item"]]
            row(r["item"], "t2", r["p2"], r["world"], {"t1_value": None if pd.isna(r["p1"]) else float(r["p1"])})
            if c["control_no_news"] and c["qid"] not in nn_done:
                nn_done.add(c["qid"])
                row(c["qid"], "nonews", r["p_nonews"], r["world"], {"t1_value": None if pd.isna(r["p1"]) else float(r["p1"])})
        assert nn_done == nn_q, (model, len(nn_done))
        n_rows += len(t1) + len(ncg) + len(nn_done)
print(f"wrote {out / 'results.jsonl'}: {n_rows} rows for {n_models} models (355 t1 + 400 t2 + 99 nonews each); excluded {sorted(excl)}")
