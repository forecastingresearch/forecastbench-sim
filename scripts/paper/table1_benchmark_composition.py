"""Table 1 — Benchmark composition.

Counts per question_type x horizon. Writes a detailed CSV plus a compact
LaTeX table for the main paper.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.paper._data import (
    H0_CONT_RESULTS,
    RUN_H0_BINARY,
    RUN_NONH0,
    infer_horizon,
    load_run,
)
from scripts.paper._style import PLOTS_DIR


def _row_for_run(run, horizon_label_overrides=None) -> dict[str, dict]:
    horizon_label_overrides = horizon_label_overrides or {}
    by = defaultdict(lambda: {"templates": set(), "games": set(), "count": 0})
    for q in run["questions"]:
        qtype = q.get("question_type")
        _turn, horizon = infer_horizon(q["question_text"])
        horizon = horizon_label_overrides.get(qtype) or horizon
        key = (qtype, horizon)
        by[key]["templates"].add(q.get("template_id", ""))
        by[key]["games"].add(q.get("game_id", ""))
        by[key]["count"] += 1
    return by


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    nonh0 = load_run(RUN_NONH0)
    h0_bin = load_run(RUN_H0_BINARY)

    for (qtype, horizon), info in sorted(_row_for_run(nonh0).items()):
        rows.append({
            "source": "main run (uncond non-H0)",
            "question_type": qtype,
            "horizon": horizon or "?",
            "n_questions": info["count"],
            "n_templates": len(info["templates"]),
            "n_games": len(info["games"]),
        })
    for (qtype, horizon), info in sorted(_row_for_run(h0_bin, {"binary": "H0"}).items()):
        rows.append({
            "source": "main run (uncond H0 binary)",
            "question_type": qtype,
            "horizon": horizon or "H0",
            "n_questions": info["count"],
            "n_templates": len(info["templates"]),
            "n_games": len(info["games"]),
        })

    if H0_CONT_RESULTS.exists():
        h0_cont = load_run(H0_CONT_RESULTS)
        # H0 continuous file has its own structure; reuse infer_horizon.
        bag = defaultdict(lambda: {"templates": set(), "games": set(), "count": 0})
        for q in h0_cont.get("questions", []):
            tid = q.get("source_template_id") or q.get("template_id", "").removeprefix("h0_")
            bag[("continuous", "H0")]["templates"].add(tid)
            bag[("continuous", "H0")]["games"].add(q.get("game_id", ""))
            bag[("continuous", "H0")]["count"] += 1
        for (qtype, horizon), info in bag.items():
            rows.append({
                "source": "H0 continuous (commit 3.5)",
                "question_type": qtype,
                "horizon": horizon,
                "n_questions": info["count"],
                "n_templates": len(info["templates"]),
                "n_games": len(info["games"]),
            })

    out = PLOTS_DIR / "table1_benchmark_composition.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "question_type", "horizon", "n_questions", "n_templates", "n_games"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")

    nonh0_binary = [r for r in rows if r["source"] == "main run (uncond non-H0)" and r["question_type"] == "binary"]
    nonh0_cont = [r for r in rows if r["source"] == "main run (uncond non-H0)" and r["question_type"] == "continuous"]
    h0_binary = [r for r in rows if r["source"] == "main run (uncond H0 binary)" and r["question_type"] == "binary"]
    h0_cont = [r for r in rows if r["source"] == "H0 continuous (commit 3.5)" and r["question_type"] == "continuous"]

    compact_rows = [
        ("Binary", "H1--H7", sum(r["n_questions"] for r in nonh0_binary), "10", "11"),
        ("Continuous", "H1--H7", sum(r["n_questions"] for r in nonh0_cont), "6", "11"),
        ("H0 binary", "H0", sum(r["n_questions"] for r in h0_binary), "10", "11"),
        ("H0 continuous", "H0", sum(r["n_questions"] for r in h0_cont), "3", "11"),
    ]
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{ForecastBench-Sim composition. H0 rows are report-reading checks; H1--H7 rows are forecasting tasks. T/W = templates/worlds.}",
        r"\label{tab:benchmark-composition}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Family & Horiz. & Qs. & T/W \\",
        r"\midrule",
    ]
    for family, horizons, questions, templates, worlds in compact_rows:
        tex.append(f"{family} & {horizons} & {questions:,} & {templates} / {worlds} \\\\")
    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    tex_out = PLOTS_DIR / "table1_benchmark_composition.tex"
    tex_out.write_text("\n".join(tex), encoding="utf-8")
    print(f"wrote {tex_out}")


if __name__ == "__main__":
    main()
