#!/usr/bin/env python3
"""Build the FreeCiv question-family tables for appendix A.

    python data/freeciv/scripts/make_freeciv_family_table.py

Outputs (in data/, where the appendix \\inputs them; the JSON in data/freeciv/):
  freeciv_binary_families.tex      family id, resolution rule, items in bank / tails / mirrors / natcond
  freeciv_continuous_families.tex  family id, resolution rule, items, normalization constant
  freeciv_family_summary.json      the parsed counts and the sources

Sources (data/freeciv/):
  COMPOSITION.md              item counts per family and set (draw v1.8)
  continuous_norm_constants.json  per-family nCRPS constants
  fbsim_v3_run/criteria_v1.py                         the resolution rules the descriptions paraphrase
"""
import json
import re
from pathlib import Path

from _common import COMPOSITION as COMP, NORM_CONSTANTS as NORM, PAPER_DATA as HERE, SCORE_ITEMS, DATA, REPO, rel  # noqa: E402

text = COMP.read_text()


def section_totals(heading_regex: str) -> dict:
    """Return {family: All} for the markdown table under the heading."""
    m = re.search(heading_regex + r".*?\n(.*?)(?:\n## |\Z)", text, re.S)
    assert m, heading_regex
    out = {}
    for line in m.group(1).splitlines():
        parts = line.split()
        if len(parts) < 2 or not re.match(r"^[A-Z]+[0-9]*_", parts[0]):
            continue
        out[parts[0]] = int(parts[-1])
    return out


bank = section_totals(r"## Binary bank 750: family x horizon")
tails = section_totals(r"## Tails 300 \(q<=\.05\): family x horizon")
mirrors = section_totals(r"## Mirrors 50 \(q>=\.95\): family x horizon")
natcond = section_totals(r"## Natcond: question family x block")
cont = section_totals(r"## Continuous 300: family x horizon")
assert sum(bank.values()) == 750 and sum(tails.values()) == 300
assert sum(mirrors.values()) == 50 and sum(natcond.values()) == 400 and sum(cont.values()) == 300

# Paraphrases of criteria_v1.py; the window is turns 61 through T inclusive,
# "at T" means as recorded at the end of turn T.
BINARY = {
    "EX_comparative": "civilization A's quantity (score, population, cities, territory, treasury, military units or technologies) at $T$ is strictly greater than B's",
    "EX_government_at": "the civilization's government at $T$ is the named form; anarchy counts as a form",
    "EX_tech_discovered": "the civilization discovers the named technology in the window",
    "NB1_threshold": "the civilization's quantity at $T$ is at least the stated value",
    "NB1_value_threshold": "the civilization's quantity at $T$ is at least the stated value",
    "NB4_drawdown": "the civilization's quantity falls below a stated share of its turn-60 value at the end of any turn in the window",
    "NB6_event": "any city is captured (or at least $k$ captures occur in total) in the window",
    "NEW_civil_war": "the civilization splits in a civil war in the window",
    "NW1_war_at": "the pair is at war at $T$",
    "NW2_diplo": "the pair is in the named state (peace, alliance, or cease-fire or armistice) at the end of some turn in the window",
    "NW4_survival": "the civilization is eliminated in the window",
    "NW5_wonder": "any civilization completes the named wonder in the window",
    "S3_wars_at_T": "at least $k$ of the ten pairs are at war at $T$",
    "S4_gov_change_count": "the civilization's government changes at least $k$ times in the window, counting changes into and out of anarchy",
    "S5_civ_wonders": "the civilization completes at least $k$ great wonders in the window",
    "S6_city_founding": "the civilization founds at least $k$ new cities in the window",
    "S7_any_destroyed": "any city is destroyed in the window",
    "W1_wonder_race": "the named civilization completes the named wonder in the window",
    "W2_directed_conquest": "civilization A captures at least one city held by B in the window",
    "W3_capture_k": "the civilization captures at least $k$ cities in the window",
    "W4_lose_k": "the civilization loses at least $k$ cities to capture in the window",
    "W5_tech_lead": "the civilization knows strictly more technologies than each other civilization at $T$",
    "W6_peace_at": "the pair is at peace at $T$",
}
CONT = {
    "NC1_value_at_T": "the civilization's score, population, number of cities or territory size at $T$",
    "P5_techs_at_T": "the number of technologies the civilization knows at $T$",
    "P6_world_techs": "the sum over the five civilizations of technologies known at $T$",
    "P1_civ_conquests": "the number of cities the civilization captures in the window",
    "P2_civ_losses": "the number of cities the civilization loses to capture in the window",
    "P3_civ_founds": "the number of new cities the civilization founds in the window",
    "NC5_world_captures": "the total number of city captures by all players in the window",
    "NC14_world_wonders": "the number of great wonders completed by all civilizations in the window",
    "S7_world_razings": "the number of cities destroyed in the window",
}
families = sorted(set(bank) | set(tails) | set(mirrors) | set(natcond))
assert set(families) == set(BINARY), set(families) ^ set(BINARY)
assert set(cont) == set(CONT), set(cont) ^ set(CONT)

norm = json.loads(NORM.read_text())["constants"]


def tex(s: str) -> str:
    return s.replace("_", r"\_")


lines = [r"\begin{tabular}{@{}l >{\raggedright\arraybackslash}p{5.4cm} r r r r@{}}", r"\toprule",
         r"Family & Resolves YES if \ldots & Mid-range & Tail & Mirror & Nat.\ cond. \\", r"\midrule"]
for f in families:
    lines.append(
        f"\\texttt{{{tex(f)}}} & {BINARY[f]} & {bank.get(f, 0)} & {tails.get(f, 0)} & "
        f"{mirrors.get(f, 0)} & {natcond.get(f, 0)} \\\\"
    )
lines += [r"\midrule",
          f"Total ({len(families)} families) & & {sum(bank.values())} & {sum(tails.values())} & "
          f"{sum(mirrors.values())} & {sum(natcond.values())} \\\\",
          r"\bottomrule", r"\end{tabular}"]
(HERE / "freeciv_binary_families.tex").write_text("\n".join(lines) + "\n")

order = ["NC1_value_at_T", "P5_techs_at_T", "P6_world_techs", "P1_civ_conquests", "P2_civ_losses",
         "P3_civ_founds", "NC5_world_captures", "NC14_world_wonders", "S7_world_razings"]
lines = [r"\begin{tabular}{@{}l >{\raggedright\arraybackslash}p{5.8cm} r >{\raggedright\arraybackslash}p{3.1cm}@{}}", r"\toprule",
         r"Family & Resolves to \ldots & Items & Constant \\", r"\midrule"]
for f in order:
    if f == "NC1_value_at_T":
        c = "; ".join(f"{k.split('/')[1].replace('_', ' ')} {int(v['constant'])}"
                      for k, v in norm.items() if k.startswith("NC1_value_at_T/"))
        c = c.replace("cities count", "cities").replace("scores", "score").replace("territory size", "territory")
    else:
        c = str(int(norm[f]["constant"]))
    lines.append(f"\\texttt{{{tex(f)}}} & {CONT[f]} & {cont[f]} & {c} \\\\")
lines += [r"\midrule", f"Total (9 families) & & {sum(cont.values())} & \\\\", r"\bottomrule", r"\end{tabular}"]
(HERE / "freeciv_continuous_families.tex").write_text("\n".join(lines) + "\n")

# ---- items per anchor game ----------------------------------------------------
# Distinct items per (set, anchor world), counted from scores_v1/score_items.csv
# for one model (every model has the same items); cross-checked against the
# bank and tail per-world counts printed in COMPOSITION.md.
import pandas as pd
SEEDS = [7001, 7003, 7005, 7008, 7010, 7011, 7014, 7022]
si = pd.read_csv(SCORE_ITEMS, low_memory=False)
one = si[si["model"] == si["model"].iloc[0]]
counts = one.groupby(["set", "world"])["item"].nunique().unstack()
m = re.search(r"## Binary bank: world x horizon.*?\n(.*?)(?:\n## |\Z)", text, re.S)
bank_world = {}
for line in m.group(1).splitlines():
    parts = line.split()
    if parts and parts[0].startswith("seed"):
        bank_world[int(parts[0][4:])] = int(parts[-1])
m = re.search(r"per world: (\{[^}]*\})", text)
tails_world = {int(k[4:]): v for k, v in eval(m.group(1)).items()}
per_world = {k: [int(counts.loc[k, f"seed{s}"]) for s in SEEDS]
             for k in ("bank", "tails", "mirrors", "continuous", "natcond", "extra")}
assert per_world["bank"] == [bank_world[s] for s in SEEDS], (per_world["bank"], bank_world)
assert per_world["tails"] == [tails_world[s] for s in SEEDS], (per_world["tails"], tails_world)
assert sum(per_world["natcond"]) == 400 and sum(per_world["continuous"]) == 300
lines = [r"\begin{tabular}{@{}l rrrrrrrr r@{}}", r"\toprule",
         "Set & " + " & ".join(str(s) for s in SEEDS) + r" & Total \\", r"\midrule"]
for label, k in (("Binary bank", "bank"), ("Tail", "tails"), ("Mirror", "mirrors"),
                 ("Continuous", "continuous"), ("Natural-conditional cells", "natcond"),
                 ("Extra turn-1 questions", "extra")):
    v = per_world[k]
    lines.append(f"{label} & " + " & ".join(str(x) for x in v) + f" & {sum(v)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE / "freeciv_items_per_anchor.tex").write_text("\n".join(lines) + "\n")

summary = {"sources": [rel(COMP), rel(NORM), "fbsim_v3_run/criteria_v1.py (run scripts, not in this repo)", rel(SCORE_ITEMS)],
           "items_per_anchor": {k: dict(zip(SEEDS, v)) for k, v in per_world.items()},
           "n_binary_families_any_set": len(families),
           "n_binary_families_in_bank": len(bank),
           "families_outside_bank": sorted(set(families) - set(bank)),
           "bank": bank, "tails": tails, "mirrors": mirrors, "natcond": natcond, "continuous": cont,
           "norm_constants": {k: v["constant"] for k, v in norm.items()}}
(DATA / "freeciv_family_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k not in ("bank", "tails", "mirrors", "natcond", "continuous")}, indent=1))
