#!/usr/bin/env python
"""update_captions.py --paper-root DIR [--check] — refresh the numbers quoted in the FreeCiv figure captions of Appendix C and
Section 4 from the current run's freeciv_summary.json and figure JSON, which the generators wrote for that run.

Three captions quote numbers (the by-horizon means, the natural-conditional move statistics, the reliability bands) and one
states how far the figure's 5,000-resample intervals lie from the validation table's.  Each numeric sentence is matched by a
pattern anchored on its fixed words, so the surrounding green prose is untouched.  Every pattern must match exactly once.
"""
import argparse, json, re, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--paper-root", required=True)
ap.add_argument("--check", action="store_true")
a = ap.parse_args()
P = Path(a.paper_root).resolve()
S = json.load(open(P / "data" / "freeciv" / "freeciv_summary.json"))
CAP = json.load(open(P / "figures" / "fig_freeciv_capability.json"))
VAL = json.load(open(P / "data" / "freeciv" / "freeciv_validation_stats.json"))

def f2(x):
    return f"{x:.2f}".replace("-", "$-$")

def f3(x):
    return f"{x:.3f}"

H = S["fig_freeciv_horizon"]
hz = lambda k: dict(zip(H[k]["horizons"], H[k]["mean_over_models"]))  # noqa: E731
c, t, b, n = hz("continuous"), hz("tails"), hz("bank"), hz("natcond")
horizon_sentence = (f"Mean excess nCRPS goes from {c[90]:.2f} at $T = 90$ to {c[210]:.2f} at $T = 210$, mean tail bits from {t[90]:.2f} to {t[210]:.2f}, "
                    f"the bank excess Brier score from {b[90]:.3f} to {b[210]:.3f}, and the natural-conditional excess Brier score from {n[120]:.3f} at $T = 120$ to {n[210]:.3f}.")

U = S["fig_freeciv_natcond_update"]["pooled"]
cells, allp = U["cells"], U["all_pairs"]
ratio = cells["mean_abs_move"] / cells["mean_abs_target"]
natcond_sentences = (f"Grey points are the {allp['n']:,} model--cell pairs. Orange is the mean of cell means by bin of true shift, with 1.96 standard errors; the dashed line marks a move equal to the true shift. "
                     f"Cell means track the true shift (Pearson $r = {cells['pearson']:.2f}$, least-squares slope {cells['ols_slope']:.2f}). "
                     f"The mean absolute move over the {cells['n']} cell means is {cells['mean_abs_move']:.2f}, {ratio:.1f} times the mean absolute true shift of {cells['mean_abs_target']:.2f}; "
                     f"over the {allp['n']:,} model--cell pairs it is {allp['mean_abs_move']:.2f}.")

Rb = S["fig_freeciv_reliability"]
q, p = Rb["pooled"]["q"], Rb["pooled"]["p"]
above = [i for i in range(len(q)) if p[i] > q[i]]
k = len(above)
W = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
if above == list(range(k)):
    where = f"they lie above the truth in the {W[k]} lowest intervals and below it in the {W[10 - k]} highest"
elif above == list(range(10 - k, 10)):
    where = f"they lie below the truth in the {W[10 - k]} lowest intervals and above it in the {W[k]} highest"
else:
    where = f"they lie above the truth in {W[k]} of the ten intervals"
bias = list(Rb["per_model_bias_bank"].values())
neg = sum(1 for x in bias if x < 0)
reliability_sentences = (f"Pooled forecasts run from {p[0]:.2f} in the lowest interval ($q = {q[0]:.2f}$) to {p[-1]:.2f} in the highest ($q = {q[-1]:.2f}$); {where}. "
                         f"The question-level mean of $f - q$ is negative for {neg} of {len(bias)} models (range {f2(min(bias))} to {'$+$' + f'{max(bias):.2f}' if max(bias) >= 0 else f2(max(bias))}).")

# figure intervals versus the validation table's
byset = {r["items_set"]: r for r in VAL["rows"]}
gap = 0.0
for s_, panel in CAP["panels"].items():
    sp = panel["spearman"]
    lo, hi = sp["ci95_models"]
    v = byset[s_]["eci"]
    gap = max(gap, abs(lo - v["ci_lo"]), abs(hi - v["ci_hi"]))
within = f"{(int(gap * 100 + 0.999) / 100):.2f}"

E = [
    ("appendix/C_full_results.tex", r"Mean excess nCRPS (?:rises|goes) from .*?Natural conditionals exist only for", horizon_sentence + " Natural conditionals exist only for"),
    ("appendix/C_full_results.tex", r"Grey points are the [\d,]+ model--cell pairs\. .*?over the [\d,]+ model--cell pairs it is [\d.]+\.", natcond_sentences),
    ("appendix/C_full_results.tex", r"Pooled forecasts run from .*?\(range \$[^)]*\)\.", reliability_sentences),
    ("sections/05_freeciv.tex", r"\(5,000 resamples, within [\d.]+ of the intervals of \\cref\{sec:validation\}\)", f"(5,000 resamples, within {within} of the intervals of \\cref{{sec:validation}})"),
]
problems, edits = [], {}
for f, pat, new in E:
    path = P / f
    text = edits.get(path, path.read_text())
    ms = list(re.finditer(pat, text, re.S))
    if len(ms) != 1:
        problems.append(f"{f}: {len(ms)} matches for {pat[:60]!r}")
        continue
    edits[path] = text[:ms[0].start()] + new + text[ms[0].end():]
if problems:
    print("NOT WRITTEN:\n  " + "\n  ".join(problems))
    sys.exit(1)
for path, text in edits.items():
    if a.check:
        print(f"would edit {path.relative_to(P)}")
    else:
        path.write_text(text)
        print(f"edited     {path.relative_to(P)}")
print("captions:", horizon_sentence, "|", natcond_sentences[-160:], "|", reliability_sentences, "| figure-vs-table gap", within, sep="\n  ")
