#!/usr/bin/env python
"""convention_2026-09-21.py --paper-root DIR [--check]

The coauthors' conventions of 21 September 2026 applied to the FreeCiv-owned prose:
  * the 750-question set is the "mid-range set" (Fabio's convention: mid-range and tail for the binary sets); file and
    column names keep "bank", and Appendix A says so once;
  * the main validation table shows one interval type, the percentile bootstrap over models (Nick, Fabio); every sentence
    that quoted FreeCiv's interval over its eight worlds now quotes the interval over models, read from
    data/freeciv/freeciv_validation_stats.json after update_shared_tables.py has run, and the claims that every interval
    excludes zero are corrected;
  * the FreeCiv continuous row is excess nCRPS;
  * the Micropolis horizon claim of the introduction is limited to its tail questions (its mid-range excess falls with horizon).
Exact matches for the sentences that change; a guarded word-level pass for the remaining "bank" wording in the appendices.
"""
import argparse, json, re, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--paper-root", required=True); ap.add_argument("--check", action="store_true")
a = ap.parse_args(); P = Path(a.paper_root).resolve()
ST = {r["column"]: r["eci"] for r in json.load(open(P / "data/freeciv/freeciv_validation_stats.json"))["rows"]}
def ci(col):
    e = ST[col]; lo, hi = e["ci_lo"], e["ci_hi"]
    f = lambda v: f"$-{abs(v):.2f}$" if v < 0 else f"{v:.2f}"
    return f"[{f(lo)}, {f(hi)}]"
def rho(col): return f"{ST[col]['rho']:.2f}"
n_excl = sum(1 for e in ST.values() if e["ci_lo"] > 0)   # FreeCiv rows whose interval over models excludes zero
n_total_excl = 6 + n_excl                                  # the six Micropolis and StarSim rows all exclude zero
WORD = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
inc = [lab for col, lab in [("tails_all_excess_bits", "tail"), ("bank_all_excess_brier", "mid-range"), ("natcond_all_excess_t2", "natural-conditional"), ("continuous_all_excess_ncrps_global", "continuous")] if ST[col]["ci_lo"] <= 0]
inc_txt = " and ".join(inc) if len(inc) <= 2 else ", ".join(inc[:-1]) + " and " + inc[-1]

R = [
    # --- Section 1
    ("sections/02_introduction.tex", r"in Micropolis, on FreeCiv's tail and continuous sets, and on StarSim's interventional questions in unnormalised units, forecasts move further from the ground truth as the horizon grows.}",
     r"on Micropolis tail questions, on FreeCiv's tail and continuous sets, and on StarSim's interventional questions in unnormalised units, forecasts move further from the ground truth as the horizon grows.}"),
    ("sections/02_introduction.tex", r"The Spearman correlation between capability and each of ten scores is positive, and its 95\% bootstrap interval excludes zero for all ten;",
     f"The Spearman correlation between capability and each of ten scores is positive, and its 95\\% bootstrap interval over models excludes zero for {WORD[n_total_excl]} of the ten;"),
    # --- Section 4
    ("sections/05_freeciv.tex", r"The binary bank holds 750 questions, 30 from each of five equal-width intervals of $q$ on $(0.05, 0.95)$ at each horizon, and the tail set 300 questions",
     r"The mid-range set holds 750 binary questions, 30 from each of five equal-width intervals of $q$ on $(0.05, 0.95)$ at each horizon, and the tail set 300 questions"),
    ("sections/05_freeciv.tex", r"(d) Binary bank: excess Brier score, 750. $\rho$ is sign-adjusted Spearman with a 95\% bootstrap over models (5,000 resamples; \cref{tab:validation} gives the interval over the eight anchor games); orange marks the best and worst model.}}",
     r"(d) Mid-range questions: excess Brier score, 750. $\rho$ is sign-adjusted Spearman with its 95\% percentile bootstrap over models (10,000 resamples), the numbers of \cref{tab:validation}; orange marks the best and worst model.}}"),
    ("sections/05_freeciv.tex", r"Their correlation with ECI is $\rho = 0.34$ [0.20, 0.42] over the eight anchor games (\cref{fig:freeciv-capability}b), against \MPDRhoTail\ on the Micropolis tail set.",
     f"Their correlation with ECI is $\\rho = {rho('tails_all_excess_bits')}$ {ci('tails_all_excess_bits')} over models (\\cref{{fig:freeciv-capability}}b), against \\MPDRhoTail\\ on the Micropolis tail set."),
    ("sections/05_freeciv.tex", r"mean 0.119, and correlates with ECI at $\rho = 0.38$ [0.18, 0.60] over games (\cref{fig:freeciv-capability}c).",
     f"mean 0.119, and correlates with ECI at $\\rho = {rho('natcond_all_excess_t2')}$ {ci('natcond_all_excess_t2')} over models (\\cref{{fig:freeciv-capability}}c)."),
    ("sections/05_freeciv.tex", r"Its correlation with ECI, $\rho = 0.59$ [0.43, 0.63] over games (\cref{fig:freeciv-capability}a), is the strongest of the four sets",
     f"Its correlation with ECI, $\\rho = {rho('continuous_all_excess_ncrps_global')}$ {ci('continuous_all_excess_ncrps_global')} over models (\\cref{{fig:freeciv-capability}}a), is the strongest of the four sets"),
    ("sections/05_freeciv.tex", r"\paragraph{Binary bank.}", r"\paragraph{Mid-range questions.}"),
    ("sections/05_freeciv.tex", r"On the 750 bank questions, whose ground truth is spread uniformly over $(0.05, 0.95)$, excess Brier runs from 0.103 (Gemini 3 Flash) to 0.193 (Llama 4 Scout), mean 0.141. It is close to unrelated to ECI, $\rho = 0.18$ [0.03, 0.23] over games (\cref{fig:freeciv-capability}d), and flat across horizons",
     f"On the 750 mid-range questions, whose ground truth is spread uniformly over $(0.05, 0.95)$, excess Brier runs from 0.103 (Gemini 3 Flash) to 0.193 (Llama 4 Scout), mean 0.141. It is close to unrelated to ECI, $\\rho = {rho('bank_all_excess_brier')}$ {ci('bank_all_excess_brier')} over models (\\cref{{fig:freeciv-capability}}d), and flat across horizons"),
    # --- Section 6
    ("sections/07_validation.tex", r"For Micropolis and StarSim the interval is a percentile bootstrap over the $n$ models, which measures how much $\rho$ would move under a different panel of models and conditions on the question sets. For FreeCiv it is a cluster bootstrap over the eight anchor games with the panel fixed (\cref{app:validation-full} gives both intervals for every row).",
     r"The interval is a percentile bootstrap over the $n$ models in every row, which measures how much $\rho$ would move under a different panel of models and conditions on the question sets; for FreeCiv, \cref{app:validation-full} adds a cluster bootstrap over its eight worlds with the panel fixed."),
    ("sections/07_validation.tex", r"Every point estimate is positive, from 0.18 on the FreeCiv bank to \blue{$\MPDRhoContinuous$ on Micropolis continuous questions}, and the interval lies above zero in every row.",
     f"Every point estimate is positive, from {rho('bank_all_excess_brier')} on the FreeCiv mid-range set to \\blue{{$\\MPDRhoContinuous$ on Micropolis continuous questions}}, and the interval lies above zero in {WORD[n_total_excl]} of the ten rows; it includes zero for the FreeCiv {inc_txt} rows, whose gradients are the weakest."),
    ("sections/07_validation.tex", r"95\% CI: for Micropolis and StarSim a percentile bootstrap over models (\blue{\MPDResamples\ resamples, at seed \MPDSeed\ for Micropolis}; seed 0 for StarSim); for FreeCiv a cluster bootstrap over the eight anchor games with the model panel fixed (\cref{app:validation-full} gives both intervals for every row); $p$: two-sided.",
     r"95\% CI: percentile bootstrap over models (\blue{\MPDResamples\ resamples, at seed \MPDSeed\ for Micropolis}; seed 0 for StarSim; seed 2026 for FreeCiv), with FreeCiv's interval over its eight worlds in \cref{app:validation-full}; $p$: two-sided."),
    # --- Appendix A: say once that the files call the set the bank
    ("appendix/A_worlds_and_protocol.tex", r"The bank holds 750 items: 30 from each of five equal-width bands of $q$ on $(0.05, 0.95)$",
     r"The mid-range set, which the released files call the bank, holds 750 items: 30 from each of five equal-width bands of $q$ on $(0.05, 0.95)$"),
    # --- data folder README
    ("data/freeciv/README.md", "one row per model and binary question (bank, tail, mirror, extra)", "one row per model and binary question (mid-range, tail, mirror, extra)"),
    ("data/freeciv/README.md", "the 231 bank questions that anchor cells appear twice", "the 231 mid-range questions that anchor cells appear twice"),
    ("data/freeciv/README.md", "| `reliability_bands.csv` | bank reliability:", "| `reliability_bands.csv` | mid-range reliability:"),
]
texts = {}
for f, old, new in R:
    s = texts.get(f) or (P / f).read_text()
    if s.count(old) != 1: print(f"ABORT: {f}: {s.count(old)} matches: {old[:80]!r}"); sys.exit(1)
    texts[f] = s.replace(old, new)

# guarded word-level pass over the FreeCiv-owned files: rename the set in prose, never inside \texttt{...}, identifiers or file names
PHRASES = [(r"\bbinary bank\b", "mid-range set"), (r"\bBinary bank\b", "Mid-range set"), (r"\bthe bank excess\b", "the mid-range excess"), (r"\bthe bank\b", "the mid-range set"),
           (r"\bbank questions\b", "mid-range questions"), (r"\bbank items\b", "mid-range items"), (r"\bbank forecasts\b", "mid-range forecasts"), (r"\bbank score\b", "mid-range score"),
           (r"\bbank excess Brier\b", "mid-range excess Brier"), (r"\bBank excess Brier\b", "Mid-range excess Brier"), (r"\bFreeCiv bank\b", "FreeCiv mid-range set"),
           (r"\bits bank\b", "its mid-range set"), (r"\bbank, tail, mirror and continuous\b", "mid-range, tail, mirror and continuous"), (r"\bbank, tail and mirror\b", "mid-range, tail and mirror"),
           (r"\bbank, the tail set\b", "mid-range set, the tail set"), (r"\(bank\)", "(mid-range)"), (r"\bmean bank\b", "mean mid-range"), (r"\bBank\b(?= &| \\\\)", "Mid-range")]
def rename(s):
    out, i = [], 0
    for m in re.finditer(r"\\texttt\{[^{}]*\}|\\(?:cref|Cref|ref|label|input|includegraphics)\{[^{}]*\}|[A-Za-z_]*bank_[A-Za-z0-9_]*|bank_750|\bbank_v1\b", s):
        seg = s[i:m.start()]
        for pat, rep in PHRASES: seg = re.sub(pat, rep, seg)
        out.append(seg); out.append(m.group(0)); i = m.end()
    seg = s[i:]
    for pat, rep in PHRASES: seg = re.sub(pat, rep, seg)
    out.append(seg); return "".join(out)
for f in ["sections/05_freeciv.tex", "sections/07_validation.tex", "sections/09_discussion.tex", "appendix/A_worlds_and_protocol.tex", "appendix/B_prompts_and_templates.tex",
          "appendix/C_full_results.tex", "appendix/D_ablations.tex", "appendix/F_release.tex", "data/freeciv/README.md"]:
    s = texts.get(f) or (P / f).read_text(); texts[f] = rename(s)
left = {}
for f, s in texts.items():
    s2 = re.sub(r"\\texttt\{[^{}]*\}|[A-Za-z_]*bank_[A-Za-z0-9_]*|bank_750|bank_v1", "", s)
    left[f] = [m.group(0) for m in re.finditer(r"[^.\n]{0,60}\b[Bb]ank\b[^.\n]{0,60}", s2)]
for f, s in texts.items():
    print("would write" if a.check else "wrote", f, "| remaining 'bank':", len(left[f]))
    for x in left[f][:6]: print("     ", x.strip())
    if not a.check: (P / f).write_text(s)
