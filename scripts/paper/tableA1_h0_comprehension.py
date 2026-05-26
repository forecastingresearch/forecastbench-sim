"""Table A1 — H0 comprehension checks as LaTeX.

Writes a compact appendix table with binary H0 Brier and H0 continuous
normalized CRPS for the curated models. H0 questions are answerable from the
turn-60 report, so this table is a report-reading sanity check rather than a
forecasting result.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.paper._data import H0_CONT_DIR, RUN_H0_BINARY, load_run
from scripts.paper._style import CURATED_MODELS, PLOTS_DIR, display_name


def _escape_tex(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _binary_h0() -> dict[str, dict]:
    run = load_run(RUN_H0_BINARY)
    out = {}
    for model_id, result in run["model_results"].items():
        binary = result.get("binary", {})
        out[model_id] = {
            "brier": binary.get("brier_score"),
            "n": binary.get("num_predictions"),
            "failures": binary.get("num_failures"),
        }
    return out


def _continuous_h0() -> dict[str, dict]:
    path = H0_CONT_DIR / "h0_model_summary.csv"
    if not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["group"] != "overall":
                continue
            out[row["model"]] = {
                "norm_crps": row["normalized_crps"],
                "n": row["n_valid_predictions"],
                "total": row["n_questions"],
            }
    return out


def _fmt(value, ndigits=4) -> str:
    if value in ("", None):
        return "--"
    try:
        return f"{float(value):.{ndigits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    binary = _binary_h0()
    continuous = _continuous_h0()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{H0 comprehension checks. H0 questions are answerable from the turn-60 report and test report reading rather than future forecasting. Lower is better.}",
        r"\label{tab:h0-comprehension}",
        r"\small",
        r"\resizebox{0.78\linewidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Bin. Brier & Bin. $n$ & Cont. nCRPS & Cont. $n$ \\",
        r"\midrule",
    ]

    for model_id in CURATED_MODELS:
        b = binary.get(model_id, {})
        c = continuous.get(model_id, {})
        cont_n = "--"
        if c:
            cont_n = f"{c.get('n', '--')}/{c.get('total', '--')}"
        lines.append(
            f"{_escape_tex(display_name(model_id))} & "
            f"{_fmt(b.get('brier'))} & "
            f"{b.get('n', '--')} & "
            f"{_fmt(c.get('norm_crps'))} & "
            f"{cont_n} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ])

    out = PLOTS_DIR / "tableA1_h0_comprehension.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
