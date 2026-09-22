#!/usr/bin/env python
"""patch_generators_2026-09-21.py — one-off edits to the FreeCiv paper generators after the coauthors' pass of 21 September:

  * naming: Fabio's convention, "mid-range" and "tail" for the two binary sets (the files keep the name bank);
  * intervals: one interval type in the main validation table, the percentile bootstrap over models (Nick and Fabio,
    21 September); FreeCiv's interval over its eight worlds stays in Appendix C;
  * the FreeCiv continuous row of the validation tables becomes excess nCRPS, as Micropolis's row is and as the figure shows;
  * the capability figure takes its brackets from the validation statistics file, so figure and table agree exactly;
  * larger labels on the FreeCiv figures (Nick).
Each replacement must match exactly once.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = [
    ("update_shared_tables.py", '    ("nCRPS", "continuous_all_ncrps_global", "continuous", "ncrps_global"),', '    ("Excess nCRPS", "continuous_all_excess_ncrps_global", "continuous", "excess_ncrps_global"),'),
    ("update_shared_tables.py", '    ("Bank excess Brier", "bank_all_excess_brier", "bank", "excess_brier"),', '    ("Mid-range excess Brier", "bank_all_excess_brier", "bank", "excess_brier"),'),
    ("update_shared_tables.py", 'FC_SETS_COST = [("Binary bank", 750),', 'FC_SETS_COST = [("Mid-range set", 750),'),
    ("update_shared_tables.py", 'RUN2_ROWS = [("Binary bank", 750, "bank"),', 'RUN2_ROWS = [("Mid-range set", 750, "bank"),'),
    ("update_shared_tables.py",
     '        assert [cells_of(l)[1] for l in block] == [r["score"] for r in rows], [cells_of(l)[1] for l in block]',
     '        OLD_LABELS = {"nCRPS": "Excess nCRPS", "Bank excess Brier": "Mid-range excess Brier"}   # labels before 21 September 2026\n'
     '        assert [OLD_LABELS.get(cells_of(l)[1], cells_of(l)[1]) for l in block] == [r["score"] for r in rows], [cells_of(l)[1] for l in block]'),
    ("update_shared_tables.py",
     '        if FINAL:\n            lines = [re.sub(r"; FreeCiv is provisional \\(one question per prompt\\)", "", l) if l.startswith("%") else l for l in lines]',
     '        if FINAL:\n            lines = [re.sub(r"; FreeCiv is provisional \\(one question per prompt\\)", "", l) if l.startswith("%") else l for l in lines]\n'
     '        lines = [l.replace("one CI column (models for Micropolis and StarSim; eight-game cluster bootstrap for FreeCiv)", "one CI column, the bootstrap over models in every row (Appendix C adds FreeCiv\'s interval over its eight worlds)") if l.startswith("%") else l for l in lines]'),
    ("update_shared_tables.py",
     '        return [r["score"], str(st["n"]), f"${st[\'rho\']:.2f}$", fmt_ci(cl["ci_lo"], cl["ci_hi"]), fmt_p(st["p"])]',
     '        return [r["score"], str(st["n"]), f"${st[\'rho\']:.2f}$", fmt_ci(st["ci_lo"], st["ci_hi"]), fmt_p(st["p"])]   # 21 September 2026: the interval over models, as in every other row'),
    ("make_freeciv_tables.py", '"Model & ECI & Bank & Tails & ', '"Model & ECI & Mid-range & Tails & '),
    ("make_freeciv_tables.py", 'Bank excess Brier, by horizon (turn)', 'Mid-range excess Brier, by horizon (turn)'),
    ("make_freeciv_tables.py", 'comment="FreeCiv bank (150 questions per horizon)', 'comment="FreeCiv mid-range set (150 questions per horizon)'),
    ("make_freeciv_family_table.py", r'Family & Resolves YES if \ldots & Bank & Tail & Mirror & Nat.\ cond. \\', r'Family & Resolves YES if \ldots & Mid-range & Tail & Mirror & Nat.\ cond. \\'),
    ("make_freeciv_figs.py", 'N_BOOT = 5000', 'N_BOOT = 10000'),
    ("make_freeciv_figs.py", '                 title="(d) Binary bank"),', '                 title="(d) Mid-range"),'),
    ("make_freeciv_figs.py", '"natcond": "(c) Natural conditionals", "bank": "(d) Binary bank"}', '"natcond": "(c) Natural conditionals", "bank": "(d) Mid-range"}'),
    ("make_freeciv_figs.py",
     'from _common import RUN  # noqa: E402',
     'from _common import RUN, REPO  # noqa: E402\n'
     '# the validation statistics file written by update_shared_tables.py: the figure quotes the same rho and interval as the table\n'
     '_STATS_FILE = REPO / "data" / "freeciv" / "freeciv_validation_stats.json"\n'
     'STATS = {r["column"]: r["eci"] for r in json.load(open(_STATS_FILE))["rows"]} if _STATS_FILE.exists() else None\n'
     'STATS_COL = {"continuous": "continuous_all_excess_ncrps_global", "tails": "tails_all_excess_bits", "natcond": "natcond_all_excess_t2", "bank": "bank_all_excess_brier"}'),
    ("make_freeciv_figs.py",
     '    rho, p, lo, hi = spearman_boot(ECI, -score)\n    boots = cluster_boot(piv, worlds)',
     '    rho, p, lo, hi = spearman_boot(ECI, -score)\n'
     '    if STATS and STATS.get(STATS_COL[key]):   # quote the table\'s numbers where they exist\n'
     '        srow = STATS[STATS_COL[key]]\n'
     '        assert abs(srow["rho"] - rho) < 0.005, (key, srow["rho"], rho)\n'
     '        rho, p, lo, hi = srow["rho"], srow["p"], srow["ci_lo"], srow["ci_hi"]\n'
     '    boots = cluster_boot(piv, worlds)'),
    ("make_freeciv_figs.py", 'fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.05))\nfig.subplots_adjust(left=0.085, right=0.995, top=0.85, bottom=0.31, wspace=0.50)',
     'fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.4))\nfig.subplots_adjust(left=0.09, right=0.995, top=0.87, bottom=0.29, wspace=0.55)'),
    ("make_freeciv_figs.py", '    ax.set_title(spec["title"], loc="left", fontsize=7)\n    ax.set_ylabel(spec["ylabel"], fontsize=6.5)\n    ax.tick_params(labelsize=6.5)\n    ax.text(0.98, 0.875, rho_text(rho, lo, hi), transform=ax.transAxes, fontsize=6,',
     '    ax.set_title(spec["title"], loc="left", fontsize=8)\n    ax.set_ylabel(spec["ylabel"], fontsize=7.5)\n    ax.tick_params(labelsize=7.5)\n    ax.text(0.98, 0.885, rho_text(rho, lo, hi), transform=ax.transAxes, fontsize=7,'),
    ("make_freeciv_figs.py", 'for ax in axes:\n    ax.set_xlabel("ECI", fontsize=7)', 'for ax in axes:\n    ax.set_xlabel("ECI", fontsize=8)'),
    ("make_freeciv_figs.py", '         ha="center", va="bottom", fontsize=5.8, color=GREY, linespacing=1.3)', '         ha="center", va="bottom", fontsize=6.3, color=GREY, linespacing=1.3)'),
    ("make_freeciv_figs.py", 'def place_labels(fig, ax, xs, ys, names, required, fontsize=6, blocked=()):', 'def place_labels(fig, ax, xs, ys, names, required, fontsize=6.5, blocked=()):'),
    ("make_freeciv_figs.py", '    ax.set_title(short_titles[key], loc="left", fontsize=8.5)', '    ax.set_title(short_titles[key], loc="left", fontsize=9.5)'),
    ("make_freeciv_figs.py", '    ax.set_xlabel("Resolution turn $T$ (snapshot at turn 60)", fontsize=8)\n    ax.set_ylabel(short_ylabels[key], fontsize=8)', '    ax.set_xlabel("Resolution turn $T$ (snapshot at turn 60)", fontsize=9)\n    ax.set_ylabel(short_ylabels[key], fontsize=9)'),
    ("make_freeciv_figs.py", 'axes[0].legend(loc="lower right", fontsize=6.5,', 'axes[0].legend(loc="lower right", fontsize=7.5,'),
    ("make_freeciv_figs.py", '    "xtick.labelsize": 8,\n    "ytick.labelsize": 8,', '    "xtick.labelsize": 9,\n    "ytick.labelsize": 9,'),
    ("make_freeciv_forecast_files.py", 'SECTION = {"bank": "bank", "tails": "tail", "mirrors": "mirror", "extra": "extra"}', 'SECTION = {"bank": "mid-range", "tails": "tail", "mirrors": "mirror", "extra": "extra"}'),
    ("make_freeciv_forecast_files.py", 'single["section"] = "bank"; single["prompt"] = "single"', 'single["section"] = "mid-range"; single["prompt"] = "single"'),
    ("make_freeciv_forecast_files.py", '    mine = bin_rows[(bin_rows.prompt == "grouped") & (bin_rows.section == sec)]', '    mine = bin_rows[(bin_rows.prompt == "grouped") & (bin_rows.section == sec)]'),
    ("make_freeciv_forecast_files.py", 'chk = {("bank", "bank_all_excess_brier", "excess_brier"),', 'chk = {("mid-range", "bank_all_excess_brier", "excess_brier"),'),
]
for f, old, new in R:
    p = HERE / f; s = p.read_text()
    if s.count(old) != 1:
        print(f"ABORT {f}: {s.count(old)} matches for {old[:70]!r}"); sys.exit(1)
    p.write_text(s.replace(old, new)); print("patched", f, "|", old[:50].replace("\n", " "))
