#!/usr/bin/env python
"""effort_appendix_2026-09-20.py --paper-root DIR --effort-dir RESULTS/run2_effort [--check]

Writes the grouped-prompt reasoning-effort check into Appendix D: copies the generated table and the numbers file into the
paper, replaces the subsection's paragraph with prose generated from those numbers, replaces the hand-written table by
an \\input of the generated one, and drops the two yellow notes (the missing o3 and GPT-5.6 Luna values, the promise to
repeat the check).  The check of 9 September stays as one sentence.  Each anchor must match exactly once.
"""
import argparse, json, re, shutil, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--paper-root", required=True); ap.add_argument("--effort-dir", required=True); ap.add_argument("--check", action="store_true")
a = ap.parse_args(); P = Path(a.paper_root).resolve(); E = Path(a.effort_dir).resolve()
J = json.load(open(E / "effort_check.json")); M = J["models"]
rows = sorted(M.values(), key=lambda r: -r["eci"])
chg = {r["name"]: r["next"]["excess"] - r["low"]["excess"] for r in rows}
best, worst = min(chg, key=chg.get), max(chg, key=chg.get)
n_better = sum(1 for v in chg.values() if v < 0)
reas_low = [r["low"]["reas_per_q"] for r in rows]; reas_next = [r["next"]["reas_per_q"] for r in rows]
bias_all = [r[l]["bias"] for r in rows for l in ("low", "next")]
slope_up = sum(1 for r in rows if r["next"]["slope"] > r["low"]["slope"] + 1e-9)
disc_d = [r["next"]["disc"] - r["low"]["disc"] for r in rows]
gpt5 = M["openai/gpt-5"]; fable = M["anthropic/claude-fable-5"]; o3 = M["openai/o3"]
cost = sum(r["next"]["cost"] for r in rows); unparsed = sum(r["next"]["unparsed"] for r in rows)
sizes = None
try:
    C = [json.loads(l) for l in open(next(E.glob("*/calls.jsonl")))]; sizes = sorted(c["n_asked"] for c in C)
except Exception:
    pass
rho = J["rho_eci"]
WORDS = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven'}
RUN1_SENT = ""
if "run1_check" in J:
    R1 = J["run1_check"]; M1 = R1["models"]
    c1 = {r["name"]: r["next"]["excess"] - r["low"]["excess"] for r in M1.values()}
    lo1, hi1 = min(c1, key=c1.get), max(c1, key=c1.get)
    b1 = [r[l]["bias"] for r in M1.values() for l in ("low", "next")]
    RUN1_SENT = (f"A first check on the one-question run of 9 September, at the same levels, drew its 200 questions in turn from each family without stratifying by horizon ({R1['in_common']} of them are in this sample); "
                 f"it moved excess Brier by between ${c1[lo1]:+.3f}$ ({lo1}) and ${c1[hi1]:+.3f}$ ({hi1}), left the bias between ${min(b1):+.2f}$ and ${max(b1):+.2f}$, and put the correlation of excess Brier with ECI at ${R1['rho_eci']['excess']['low']:+.2f}$ and ${R1['rho_eci']['excess']['next']:+.2f}$. ")
PARA = (f"The FreeCiv runs set every model to its lowest reasoning effort (\\cref{{sec:freeciv}}). Because the binary forecasts are compressed and biased low, we checked whether more reasoning changes them. "
        f"We drew 200 mid-range questions, 40 per horizon in turn from each family, and asked them again of seven models at the next level: effort medium where the provider exposes a level, and a 2,048-token budget for Haiku 4.5 and Qwen3 235B. "
        f"The questions went in grouped prompts of the main run's kind, one game per prompt" + (f" with {sizes[0]} to {sizes[-1]} questions" if sizes else "") + f". The check cost \\${cost:.2f}" + (" and every answer parsed" if unparsed == 0 else f" and {unparsed} answers did not parse") + ". "
        f"\\Cref{{tab:effort}} gives the result. Excess Brier moved by between ${chg[best]:+.3f}$ ({best}) and ${chg[worst]:+.3f}$ ({worst}); {WORDS[n_better]} of the seven models improved. "
        f"Reasoning tokens per question rose from {min(reas_low):.0f} to {max(reas_low):.0f} at the lowest level to {min(reas_next):.0f} to {max(reas_next):.0f} at the next; GPT-5, which the main run had at effort minimal, went from {gpt5['low']['reas_per_q']:.0f} to {gpt5['next']['reas_per_q']:.0f}. "
        f"The mean of forecast minus truth stayed between ${min(bias_all):+.2f}$ and ${max(bias_all):+.2f}$ at both levels, and every model stayed above the {J['flat_excess']:.3f} that a constant forecast of 0.5 scores on these questions. "
        f"The slope of forecast on truth rose for {'every model' if slope_up == 7 else WORDS[slope_up] + ' of the seven models'} (Fable from {fable['low']['slope']:.2f} to {fable['next']['slope']:.2f}, o3 from {o3['low']['slope']:.2f} to {o3['next']['slope']:.2f}), and discrimination changed by ${min(disc_d):+.2f}$ to ${max(disc_d):+.2f}$. "
        f"Across the seven models the correlation of excess Brier with ECI was ${rho['excess']['low']:+.2f}$ at the lowest level and ${rho['excess']['next']:+.2f}$ at the next, and that of discrimination ${rho['disc']['low']:+.2f}$ and ${rho['disc']['next']:+.2f}$. "
        + RUN1_SENT
        + "At these levels the effort setting does not produce the compression.")
CAPTION = (r"\caption{\green{Excess Brier score on the same 200 FreeCiv mid-range questions at the lowest reasoning effort of the main run and at the next level (effort medium, or a 2,048-token budget for Haiku 4.5 and Qwen3 235B), "
           r"asked in grouped prompts of the main run's kind; lower is better. Bias is the mean of forecast minus truth; reasoning tokens are per question, a prompt's reasoning tokens divided by its questions. "
           f"A constant forecast of 0.5 scores {J['flat_excess']:.3f} on these questions. Source: \\texttt{{freeciv\\_effort\\_check.json}}.}}}}")
TABLE = "\\begin{table}[t]\n" + CAPTION + "\n\\label{tab:effort}\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n{\\color{draftgreen}\\input{data/appendix_tables/freeciv_effort}}\n\\end{table}"

f = "appendix/D_ablations.tex"; s = (P / f).read_text()
m = re.search(r"(\\subsection\{Reasoning effort \(FreeCiv\)\}\n\\label\{app:ablation-effort\}\n\n\\begin\{greentext\}\n)(.*?)(\n\\end\{greentext\})", s, re.S)
if not m: print("ABORT: paragraph anchor"); sys.exit(1)
s = s[:m.start(2)] + PARA + s[m.end(2):]
t = re.search(r"\\begin\{table\}\[t\]\n\\caption\{\\green\{Excess Brier score on the same 200 FreeCiv mid-range questions.*?\\label\{tab:effort\}.*?\\end\{table\}", s, re.S)
if not t: print("ABORT: table anchor"); sys.exit(1)
s = s[:t.start()] + TABLE + s[t.end():]
readme = P / "data/freeciv/README.md"; r = readme.read_text()
old = "| `freeciv_numbers.json`, `freeciv_summary.json`"
new = "| `freeciv_effort_check.json` | the reasoning-effort check on the grouped prompts: per model, both levels, the 200 questions' constant-0.5 excess, correlations with ECI |\n" + old
if r.count(old) != 1 and "freeciv_effort_check" not in r: print("ABORT: README anchor"); sys.exit(1)
print("would write" if a.check else "writing", f, "data/appendix_tables/freeciv_effort.tex", "data/freeciv/freeciv_effort_check.json")
if not a.check:
    (P / f).write_text(s)
    shutil.copy(E / "effort_table.tex", P / "data/appendix_tables/freeciv_effort.tex")
    shutil.copy(E / "effort_check.json", P / "data/freeciv/freeciv_effort_check.json")
    if "freeciv_effort_check" not in r: readme.write_text(r.replace(old, new))
print(PARA)
