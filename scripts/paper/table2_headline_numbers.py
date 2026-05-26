"""Table 2 — Headline per-model numbers.

One row per curated 9 model with: rank-out-of-N (binary), Brier overall,
rank-out-of-N (continuous), normalized CRPS overall, intervention-gain
on republic, intervention-gain on gold500. CSV for Overleaf.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.paper._data import (
    ARCHIVE_GOLD500_BASELINE_BIN,
    ARCHIVE_GOLD500_CONDITIONAL_BIN,
    ARCHIVE_REPUBLIC_BASELINE_BIN,
    ARCHIVE_REPUBLIC_CONDITIONAL_BIN,
    RUN_NONH0,
    load_run,
    mean_brier_by_model,
    mean_normalized_crps_by_model,
    rank_models,
    score_binary,
    score_continuous_with_opus,
)
from scripts.paper._style import CURATED_MODELS, PLOTS_DIR, display_name


def _intervention_gain(baseline_run: dict, conditional_run: dict) -> dict[str, float]:
    base_by_qid = {q["question_id"]: q for q in baseline_run["questions"]}
    per_model: dict[str, list[float]] = {m: [] for m in CURATED_MODELS}
    for q_cond in conditional_run["questions"]:
        bid = q_cond["question_id"].removesuffix("_intervention")
        q_base = base_by_qid.get(bid)
        if q_base is None:
            continue
        gt_intervention = int(bool(q_cond["ground_truth"]))
        for m in CURATED_MODELS:
            cp = q_cond["predictions"].get(m, {})
            bp = q_base["predictions"].get(m, {})
            if cp.get("error") or bp.get("error"):
                continue
            p_cond, p_base = cp.get("probability"), bp.get("probability")
            if p_cond is None or p_base is None:
                continue
            try:
                p_cond, p_base = float(p_cond), float(p_base)
            except (TypeError, ValueError):
                continue
            naive = (p_base - gt_intervention) ** 2
            updated = (p_cond - gt_intervention) ** 2
            per_model[m].append(naive - updated)
    return {m: mean(v) for m, v in per_model.items() if v}


def main() -> None:
    main_run = load_run(RUN_NONH0)
    bin_means = mean_brier_by_model(score_binary(main_run))
    cont_means = mean_normalized_crps_by_model(score_continuous_with_opus())

    bin_rank = {m: r for r, m, _s in rank_models(bin_means)}
    cont_rank = {m: r for r, m, _s in rank_models(cont_means)}
    n_bin = len(bin_means)
    n_cont = len(cont_means)

    rep_gain = _intervention_gain(
        load_run(ARCHIVE_REPUBLIC_BASELINE_BIN),
        load_run(ARCHIVE_REPUBLIC_CONDITIONAL_BIN),
    )
    gold_gain = _intervention_gain(
        load_run(ARCHIVE_GOLD500_BASELINE_BIN),
        load_run(ARCHIVE_GOLD500_CONDITIONAL_BIN),
    )

    rows: list[dict] = []
    for m in CURATED_MODELS:
        rows.append({
            "model": display_name(m),
            "model_id": m,
            "binary_brier": f"{bin_means.get(m, ''):.4f}" if m in bin_means else "",
            "binary_rank": f"{bin_rank.get(m, '')}/{n_bin}" if m in bin_rank else "",
            "continuous_norm_crps": f"{cont_means.get(m, ''):.4f}" if m in cont_means else "",
            "continuous_rank": f"{cont_rank.get(m, '')}/{n_cont}" if m in cont_rank else "",
            "republic_intervention_gain": f"{rep_gain.get(m, ''):.4f}" if m in rep_gain else "",
            "gold500_intervention_gain": f"{gold_gain.get(m, ''):.4f}" if m in gold_gain else "",
        })

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "table2_headline_numbers.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "model_id",
                "binary_brier", "binary_rank",
                "continuous_norm_crps", "continuous_rank",
                "republic_intervention_gain", "gold500_intervention_gain",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
