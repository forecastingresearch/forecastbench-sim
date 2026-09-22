"""FreeCiv figures for the ForecastBench-Sim paper.

    python data/freeciv/scripts/make_freeciv_figs.py [--png]


Sources (read-only):
  score_items.csv        per-item scores (46,176 rows: model x item x arm)
  freeciv_results_wide.csv  ECI per model (cross-checked against model_scores.csv)
  reliability_bands.csv  bank reliability bands (10 truth bands per model)
Outputs: PDF and a per-figure JSON of every plotted number in figures/; data/freeciv/freeciv_summary.json.
Inputs are read from data/freeciv/ (see _common.py). Moved from the retired paper/data/make_freeciv_figs_v13.py on 2026-09-19.
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", message=".*cmr10.*")

from _common import DATA, FIGURES, MODEL_SCORES, RELIABILITY, SCORE_ITEMS, WIDE, rel  # noqa: E402

ITEMS = SCORE_ITEMS
RELIAB = RELIABILITY
FIGDIR = FIGURES
DATADIR = DATA
FIGDIR.mkdir(parents=True, exist_ok=True)
WRITE_PNG = "--png" in sys.argv[1:]   # PDFs go to figures/; PNG previews only on request

GREEN = "#102B23"
ORANGE = "#E8632C"
GREY = "#8C8C8C"
LGREY = "#D0D0D0"
from _common import RUN, REPO  # noqa: E402
# the validation statistics file written by update_shared_tables.py: the figure quotes the same rho and interval as the table
_STATS_FILE = REPO / "data" / "freeciv" / "freeciv_validation_stats.json"
STATS = {r["column"]: r["eci"] for r in json.load(open(_STATS_FILE))["rows"]} if _STATS_FILE.exists() else None
STATS_COL = {"continuous": "continuous_all_excess_ncrps_global", "tails": "tails_all_excess_bits", "natcond": "natcond_all_excess_t2", "bank": "bank_all_excess_brier"}
PROV = "provisional: one question per prompt; batched rerun pending" if RUN.startswith("run1") else ""
N_BOOT = 10000
RNG = np.random.default_rng(20260916)

mpl.rcParams.update({
    "font.family": "cmr10",
    "axes.formatter.use_mathtext": True,
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 200,
})

SHORT = {
    "anthropic/claude-fable-5": "Fable 5",
    "anthropic/claude-opus-5": "Opus 5",
    "anthropic/claude-sonnet-5": "Sonnet 5",
    "anthropic/claude-haiku-4.5": "Haiku 4.5",
    "openai/gpt-5.6-sol": "GPT-5.6 Sol",
    "openai/gpt-5.6-luna": "GPT-5.6 Luna",
    "openai/gpt-5.5": "GPT-5.5",
    "openai/gpt-5": "GPT-5",
    "openai/gpt-5-mini": "GPT-5 Mini",
    "openai/gpt-5-nano": "GPT-5 Nano",
    "openai/gpt-5.4-nano": "GPT-5.4 Nano",
    "openai/o3": "o3",
    "openai/o4-mini": "o4-mini",
    "openai/gpt-4.1": "GPT-4.1",
    "google/gemini-3.7-flash": "Gemini 3.7 Flash",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "deepseek/deepseek-v4-flash-0731": "DeepSeek V4 Flash",
    "deepseek/deepseek-chat": "DeepSeek V3",
    "moonshotai/kimi-k2": "Kimi K2",
    "qwen/qwen3-235b-a22b": "Qwen3 235B",
    "qwen/qwen3.5-flash-02-23": "Qwen3.5 Flash",
    "meta-llama/llama-4-scout": "Llama 4 Scout",
}

# ---------------------------------------------------------------- data
items = pd.read_csv(ITEMS, low_memory=False)
wide = pd.read_csv(WIDE)
eci = wide.set_index("model")["eci"]
ms = pd.read_csv(MODEL_SCORES)
# cross-check ECI values against model_scores.csv (joined on LiteLLM slug where present)
chk = ms.dropna(subset=["LiteLLMSlug"]).set_index("LiteLLMSlug")["ECI"]
for slug, v in chk.items():
    key = slug.replace("claude-haiku-4-5-20251001", "claude-haiku-4.5")
    if key in eci.index:
        assert abs(eci[key] - v) < 1e-6, (key, eci[key], v)
assert sorted(np.round(ms.ECI, 2)) == sorted(np.round(eci.values, 2)), "ECI sets differ"
models = list(eci.sort_values().index)
assert len(models) == 24
ECI = eci.loc[models].values

PANELS = {
    "continuous": dict(set="continuous", col="excess_ncrps_global", n_items=300,
                       ylabel="Excess nCRPS",
                       title="(a) Continuous"),
    "tails": dict(set="tails", col="excess_bits", n_items=300,
                  ylabel="Excess bits",
                  title="(b) Tails ($q \\leq 0.05$)"),
    "natcond": dict(set="natcond", col="excess_t2", n_items=400,
                    ylabel="Turn-2 excess Brier",
                    title="(c) Natural conditionals"),
    "bank": dict(set="bank", col="excess_brier", n_items=750,
                 ylabel="Excess Brier",
                 title="(d) Mid-range"),
}
HORIZONS = {"continuous": [90, 120, 150, 180, 210], "tails": [90, 120, 150, 180, 210],
            "natcond": [120, 150, 180, 210], "bank": [90, 120, 150, 180, 210]}


def per_model_matrix(setname, col):
    """Return (item list, worlds per item, matrix models x items) of per-item scores."""
    sub = items[items.set == setname]
    piv = sub.pivot(index="model", columns="item", values=col).loc[models]
    worlds = sub.drop_duplicates("item").set_index("item")["world"].loc[piv.columns].values
    Ts = sub.drop_duplicates("item").set_index("item")["T"].loc[piv.columns].values
    return piv, worlds, Ts


def spearman_boot(x, y, n_boot=N_BOOT):
    """Spearman of x vs y with percentile CI from bootstrap over the paired (model) units."""
    rho, p = spearmanr(x, y)
    n = len(x)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        reps[b] = spearmanr(x[idx], y[idx]).statistic
    return float(rho), float(p), float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))


def cluster_boot(piv, worlds, n_boot=N_BOOT):
    """Cluster bootstrap: resample the 8 anchor worlds with replacement, items nested; rho of
    per-model mean score vs ECI. Also an iid item bootstrap for reference. Sign-adjusted rho."""
    M = piv.values  # models x items
    uw = np.unique(worlds)
    S = np.stack([np.nansum(M[:, worlds == w], axis=1) for w in uw], axis=1)
    N = np.stack([np.sum(~np.isnan(M[:, worlds == w]), axis=1) for w in uw], axis=1)
    reps_c = np.empty(n_boot)
    for b in range(n_boot):
        cnt = np.bincount(RNG.integers(0, len(uw), len(uw)), minlength=len(uw))
        means = (S * cnt).sum(1) / (N * cnt).sum(1)
        reps_c[b] = spearmanr(ECI, -means).statistic
    reps_i = np.empty(n_boot)
    n_it = M.shape[1]
    for b in range(n_boot):
        idx = RNG.integers(0, n_it, n_it)
        means = np.nanmean(M[:, idx], axis=1)
        reps_i[b] = spearmanr(ECI, -means).statistic
    return dict(
        cluster_world=dict(ci95=[float(np.percentile(reps_c, 2.5)), float(np.percentile(reps_c, 97.5))],
                           n_worlds=int(len(uw)), n_boot=n_boot,
                           method="resample the 8 anchor worlds with replacement; all items of a drawn world enter (items nested in worlds); per-model mean recomputed; Spearman of ECI vs -score",
                           caveat="only 8 clusters, so the percentile interval is coarse; the model-level CI printed in the panel is the headline interval"),
        item_iid=dict(ci95=[float(np.percentile(reps_i, 2.5)), float(np.percentile(reps_i, 97.5))],
                      n_items=int(n_it), n_boot=n_boot,
                      method="resample items iid with replacement, same for all models; Spearman of ECI vs -score"),
    )


def fmt(x, nd=2):
    s = f"{abs(x):.{nd}f}"
    return ("-" if x < 0 else "") + s


def rho_text(rho, lo, hi):
    return r"$\rho = " + fmt(rho) + r"\ [" + fmt(lo) + r",\ " + fmt(hi) + r"]$"


def prov_note(ax, loc="upper left"):
    x, ha = (0.02, "left") if loc.endswith("left") else (0.98, "right")
    y, va = (0.98, "top") if loc.startswith("upper") else (0.02, "bottom")
    ax.text(x, y, PROV, transform=ax.transAxes, fontsize=5.5, color=GREY, ha=ha, va=va, zorder=1)


def place_labels(fig, ax, xs, ys, names, required, fontsize=6.2, blocked=()):
    """Greedy label placement: required indices always get a label (least-overlap slot);
    the rest are labelled only where the label overlaps nothing. `blocked` = axes-fraction
    boxes (x0, y0, x1, y1) that labels must not enter."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(rend)
    disp = ax.transData.transform(np.c_[xs, ys])
    pt_boxes = [Bbox.from_bounds(px - 2.5, py - 2.5, 5, 5) for px, py in disp]
    placed = [Bbox(ax.transAxes.transform([[b[0], b[1]], [b[2], b[3]]])) for b in blocked]
    slots = [(4, 2, "left", "bottom"), (4, -2, "left", "top"), (-4, 2, "right", "bottom"),
             (-4, -2, "right", "top"), (0, 5, "center", "bottom"), (0, -5, "center", "top"),
             (6, 0, "left", "center"), (-6, 0, "right", "center"),
             (8, 7, "left", "bottom"), (8, -7, "left", "top"), (-8, 7, "right", "bottom"), (-8, -7, "right", "top"),
             (0, 10, "center", "bottom"), (0, -10, "center", "top"),
             # farther slots: the leader line keeps the association readable
             (13, 11, "left", "bottom"), (13, -11, "left", "top"), (-13, 11, "right", "bottom"), (-13, -11, "right", "top"),
             (0, 16, "center", "bottom"), (0, -16, "center", "top"), (16, 0, "left", "center"), (-16, 0, "right", "center"),
             (20, 16, "left", "bottom"), (20, -16, "left", "top"), (-20, 16, "right", "bottom"), (-20, -16, "right", "top")]
    order = list(required) + [i for i in np.argsort(-np.abs(ys - np.median(ys))) if i not in required]
    labelled = []
    for i in order:
        best, best_ov = None, np.inf
        for dx, dy, ha, va in (slots if i in required else slots[:14]):
            t = ax.annotate(names[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, ha=ha, va=va, color="#333333", zorder=4,
                            arrowprops=dict(arrowstyle="-", lw=0.35, color="#B8B8B8", shrinkA=0, shrinkB=2.2))
            # text-only box (Annotation.get_window_extent would also include the leader line);
            # update_positions applies the offset first, otherwise the box sits on the data point
            t.update_positions(rend)
            bb = mpl.text.Text.get_window_extent(t, rend).expanded(1.08, 1.15)
            ov = 0.0
            for j, pb in enumerate(pt_boxes):
                if j != i and bb.overlaps(pb):
                    ov += 1.0
            for pb in placed:
                if bb.overlaps(pb):
                    ov += 2.0
            if not (axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1 and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1):
                ov += 10.0   # a label outside its panel is worse than one that touches a few points
            if ov < best_ov:
                if best is not None:
                    best.remove()
                best, best_ov, best_bb = t, ov, bb
            else:
                t.remove()
            if ov == 0:
                break
        if best_ov == 0 or i in required:
            placed.append(best_bb)
            labelled.append(names[i])
        else:
            best.remove()
    return labelled


def save(fig, stem, numbers):
    fig.savefig(FIGDIR / f"{stem}.pdf")
    if WRITE_PNG:
        fig.savefig(FIGDIR / f"{stem}.png", dpi=200)
    with open(FIGDIR / f"{stem}.json", "w") as f:
        json.dump(numbers, f, indent=1)
    plt.close(fig)


summary = dict(
    status="PROVISIONAL FreeCiv v3 (run 2026-09-09): one question per prompt, one sample per item, lowest reasoning effort; batched (n=50 per prompt) rerun pending",
    sources=dict(score_items=rel(ITEMS), results_wide=rel(WIDE), reliability_bands=rel(RELIAB), model_scores=rel(MODEL_SCORES)),
    conventions=dict(
        rho="Spearman rank correlation of ECI with -score (score: lower is better), so positive rho = more capable models score better; this matches the sign-adjusted rho in SCORES.md/freeciv_results_table.md",
        ci_models=f"percentile 95% CI from {N_BOOT} bootstrap resamples of the 24 models (paired ECI, score)",
        excess_ncrps_global="ncrps_global - floor/norm_c per item, i.e. nCRPS minus the simulator's own CRPS floor on the same per-family constant (score_items.csv columns)",
        truth="frequency/distribution over all 1,000 continuations per anchor world (8 worlds)",
    ),
    models={m: dict(short=SHORT[m], eci=float(eci[m])) for m in models},
)

# ================================================================ Figure 1: capability 2x2
fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.15))
fig.subplots_adjust(left=0.09, right=0.995, top=0.86, bottom=0.31, wspace=0.55)
cap = {}
for ax, (key, spec) in zip(axes.ravel(), PANELS.items()):
    piv, worlds, Ts = per_model_matrix(spec["set"], spec["col"])
    assert piv.shape[1] == spec["n_items"], (key, piv.shape)
    score = piv.mean(axis=1, skipna=True).values  # nanmean over items, as SCORES.md
    n_valid = piv.notna().sum(axis=1).values
    rho, p, lo, hi = spearman_boot(ECI, -score)
    if STATS and STATS.get(STATS_COL[key]):   # quote the table's numbers where they exist
        srow = STATS[STATS_COL[key]]
        assert abs(srow["rho"] - rho) < 0.005, (key, srow["rho"], rho)
        rho, p, lo, hi = srow["rho"], srow["p"], srow["ci_lo"], srow["ci_hi"]
    boots = cluster_boot(piv, worlds)
    order = np.argsort(score)
    required = list(order[:1]) + list(order[-1:])
    ax.scatter(ECI, score, s=14, color=GREEN, zorder=3, linewidths=0)
    ax.scatter(ECI[required], score[required], s=14, color=ORANGE, zorder=3, linewidths=0)
    names = [SHORT[m] for m in models]
    r = score.max() - score.min()
    ax.set_ylim(score.min() - 0.18 * r, score.max() + 0.42 * r)
    ax.set_xlim(ECI.min() - 2.5, ECI.max() + 8.0)   # room for the label of the best model at the right edge
    labelled = place_labels(fig, ax, ECI, score, names, required, blocked=[(0, 0.80, 1, 1)])
    ax.set_title(spec["title"], loc="left", fontsize=8)
    ax.set_ylabel(spec["ylabel"], fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    ax.text(0.98, 0.885, rho_text(rho, lo, hi), transform=ax.transAxes, fontsize=6.5,
            ha="right", va="top", color=GREEN)
    cap[key] = dict(
        set=spec["set"], metric=spec["col"], n_items=int(spec["n_items"]),
        scores={m: float(s) for m, s in zip(models, score)},
        n_valid={m: int(n) for m, n in zip(models, n_valid)},
        eci={m: float(e) for m, e in zip(models, ECI)},
        spearman=dict(rho=rho, p=p, ci95_models=[lo, hi], n_models=24),
        spearman_item_bootstraps=boots,
        best3=[models[i] for i in order[:3]], worst3=[models[i] for i in order[-3:]],
        labelled_models=labelled,
    )
    print(f"{key:11s} rho={rho:+.3f} p={p:.3f} CI_models=[{lo:+.2f},{hi:+.2f}] "
          f"CI_world=[{boots['cluster_world']['ci95'][0]:+.2f},{boots['cluster_world']['ci95'][1]:+.2f}] "
          f"CI_item=[{boots['item_iid']['ci95'][0]:+.2f},{boots['item_iid']['ci95'][1]:+.2f}] "
          f"best={SHORT[models[order[0]]]} {score[order[0]]:.4f} worst={SHORT[models[order[-1]]]} {score[order[-1]]:.4f}")
for ax in axes:
    ax.set_xlabel("ECI", fontsize=8)
fig.text(0.5, 0.008,
         r"$\rho$: Spearman rank correlation of ECI with $-$score, so positive means more capable models score better." + "\n"
         f"Brackets: 95% percentile CI from {N_BOOT:,} bootstrap resamples of the 24 models. Orange: best and worst model.",
         ha="center", va="bottom", fontsize=6.3, color=GREY, linespacing=1.3)
fig.text(0.5, 0.995, PROV, ha="center", va="top", fontsize=6, color=GREY)
save(fig, "fig_freeciv_capability", dict(source=rel(ITEMS), eci_source=rel(WIDE), panels=cap, provisional=PROV))
summary["fig_freeciv_capability"] = cap

# ================================================================ Figure 2: horizon
fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.7))
fig.subplots_adjust(left=0.11, right=0.985, top=0.94, bottom=0.12, wspace=0.34, hspace=0.55)
axes = axes.ravel()
hz = {}
uniq = items.drop_duplicates("item").set_index("item")
def kl_bits(q, p):
    q = np.clip(q, 1e-9, 1 - 1e-9)
    return q * np.log2(q / p) + (1 - q) * np.log2((1 - q) / (1 - p))
short_titles = {"continuous": "(a) Continuous", "tails": "(b) Tails ($q \\leq 0.05$)",
                "natcond": "(c) Natural conditionals", "bank": "(d) Mid-range"}
short_ylabels = {"continuous": "Excess nCRPS\n(lower is better)", "tails": "Excess bits $\\mathrm{KL}(q\\,\\|\\,p)$\n(lower is better)",
                 "natcond": "Turn-2 excess Brier\n(lower is better)", "bank": "Excess Brier\n(lower is better)"}
for ax, (key, spec) in zip(axes, PANELS.items()):
    piv, worlds, Ts = per_model_matrix(spec["set"], spec["col"])
    hs = HORIZONS[key]
    per_model = np.array([[np.nanmean(piv.values[:, Ts == h][i]) for h in hs] for i in range(len(models))])
    mean = per_model.mean(0)
    q25, q75 = np.percentile(per_model, 25, axis=0), np.percentile(per_model, 75, axis=0)
    ax.fill_between(hs, q25, q75, color=GREEN, alpha=0.18, linewidth=0, label="IQR over models")
    ax.plot(hs, mean, color=GREEN, lw=1.4, marker="o", ms=3, label="mean over models")
    ref = None
    if key == "bank":
        u = uniq[uniq.set == "bank"]
        ref = [float(np.mean((0.5 - u.q[u["T"] == h]) ** 2)) for h in hs]
        ref_label = "flat 0.5"
    elif key == "tails":
        u = uniq[uniq.set == "tails"]
        ref = [float(np.mean(kl_bits(u.q[u["T"] == h].values, 0.05))) for h in hs]
        ref_label = "flat 0.05"
    elif key == "natcond":
        piv_s, _, Ts_s = per_model_matrix("natcond", "stay")
        ref = [float(np.nanmean(piv_s.values[:, Ts_s == h])) for h in hs]
        ref_label = "no update"
    if ref is not None:
        ax.plot(hs, ref, color=GREY, lw=1.0, ls="--", dashes=(4, 2), label=ref_label)
        ax.annotate(ref_label, (hs[-1], ref[-1]), xytext=(4, -7 if key == "natcond" else 0),
                    textcoords="offset points", fontsize=6, color=GREY, va="center", ha="left")
    ax.set_title(short_titles[key], loc="left", fontsize=9.5)
    ax.set_xticks([90, 120, 150, 180, 210])
    ax.set_xticklabels(["90", "120", "150", "180", "210"])
    ax.set_xlim(82, 248)
    ax.set_xlabel("Resolution turn $T$ (snapshot at turn 60)", fontsize=9)
    ax.set_ylabel(short_ylabels[key], fontsize=9)
    top = max(q75.max(), max(ref) if ref is not None else 0)
    ax.set_ylim(0, top * 1.28)
    prov_note(ax, "upper left")
    hz[key] = dict(set=spec["set"], metric=spec["col"], horizons=hs, mean_over_models=mean.tolist(),
                   iqr_q25=q25.tolist(), iqr_q75=q75.tolist(),
                   per_model={m: per_model[i].tolist() for i, m in enumerate(models)},
                   reference=(dict(label=ref_label, values=ref) if ref is not None else None))
    print(f"{key:11s} horizon means: " + " ".join(f"{v:.3f}" for v in mean) + (f" ref: " + " ".join(f"{v:.3f}" for v in ref) if ref else ""))
axes[0].legend(loc="lower right", fontsize=7.5, handlelength=1.6, borderaxespad=0.3)
save(fig, "fig_freeciv_horizon", dict(source=rel(ITEMS), panels=hz, provisional=PROV,
                                      notes="mean and interquartile range across the 24 per-model means at each resolution turn; snapshot at turn 60; natural conditionals exist only for T120-T210; bank reference = mean (0.5-q)^2 over that horizon's items; tails reference = mean KL(q||0.05); natcond reference = mean over models of the not-updating excess (p1 - p(Y|X))^2"))
summary["fig_freeciv_horizon"] = hz

# ================================================================ Figure 3: reliability (appendix)
rb = pd.read_csv(RELIAB)
assert rb.model.nunique() == 24 and rb.band.nunique() == 10
# cross-check against score_items: band = decile of q over the 750 bank items
bank = items[items.set == "bank"].copy()
edges = np.round(np.linspace(0.05, 0.95, 11), 6)
bank["band"] = pd.cut(bank.q, edges, labels=False, right=True, include_lowest=True)
chk = bank.groupby(["model", "band"]).agg(q=("q", "mean"), p=("p", "mean"), n=("p", "size")).reset_index()
m = rb.merge(chk, on=["model", "band"], suffixes=("", "_chk"))
print("reliability cross-check max |dq|,|dp|:", (m.q - m.q_chk).abs().max(), (m.p - m.p_chk).abs().max(),
      "n equal:", (m.n == m.n_chk).all())
pooled = rb.groupby("band")[["q", "p", "n"]].apply(lambda g: pd.Series(dict(q=np.average(g.q, weights=g.n), p=np.average(g.p, weights=g.n), n=int(g.n.sum())))).reset_index()
fig, ax = plt.subplots(figsize=(3.0, 3.0))
fig.subplots_adjust(left=0.19, right=0.97, top=0.95, bottom=0.16)
ax.plot([0, 1], [0, 1], color=GREY, ls="--", dashes=(4, 2), lw=0.8, zorder=1)
for mname, g in rb.groupby("model"):
    g = g.sort_values("band")
    ax.plot(g.q, g.p, color=LGREY, lw=0.6, zorder=2)
ax.plot(pooled.q, pooled.p, color=GREEN, lw=1.6, marker="o", ms=3, zorder=3, label="pooled over 24 models")
ax.plot([], [], color=LGREY, lw=0.6, label="one line per model")
ax.plot([], [], color=GREY, ls="--", dashes=(4, 2), lw=0.8, label="perfect reliability")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("Mean truth probability $q$ in band")
ax.set_ylabel("Mean forecast $p$ in band")
ax.set_aspect("equal")
ax.legend(loc="lower right", fontsize=6.5, handlelength=1.8)
prov_note(ax, "upper left")
save(fig, "fig_freeciv_reliability", dict(
    source=rel(RELIAB), cross_check="score_items.csv bank rows; bands = ten equal-width bins of q on [0.05, 0.95] (width 0.09, right-closed)",
    band_edges_q=edges.tolist(),
    pooled=dict(q=pooled.q.tolist(), p=pooled.p.tolist(), n=pooled.n.tolist()),
    per_model={mname: dict(q=g.sort_values("band").q.tolist(), p=g.sort_values("band").p.tolist(), n=g.sort_values("band").n.tolist()) for mname, g in rb.groupby("model")},
    provisional=PROV))
summary["fig_freeciv_reliability"] = dict(band_edges_q=edges.tolist(), pooled=dict(q=pooled.q.tolist(), p=pooled.p.tolist(), n=pooled.n.tolist()),
                                          per_model_bias_bank={m: float((bank.p - bank.q)[bank.model == m].mean()) for m in models},
                                          per_model_bias_bank_note="item-level mean of (p - q) over the 750 bank items per model (not the unweighted mean over bands)")

# ================================================================ Figure 4: natcond update (appendix)
nc = items[items.set == "natcond"].copy()
nc = nc.dropna(subset=["move", "target"])
assert np.allclose(nc.move, nc.p2 - nc.p1, atol=1e-6) and np.allclose(nc.target, nc.p_given - nc.q, atol=1e-6)
per_model_corr = {}
for mname, g in nc.groupby("model"):
    r = pearsonr(g.move, g.target).statistic
    rs = spearmanr(g.move, g.target).statistic
    slope = np.polyfit(g.target, g.move, 1)[0]
    big = g[g.target.abs() >= 0.05]
    per_model_corr[mname] = dict(pearson=float(r), spearman=float(rs), ols_slope_move_on_target=float(slope),
                                 mean_abs_move=float(g.move.abs().mean()), mean_abs_target=float(g.target.abs().mean()),
                                 sign_agree_abs_target_ge_0p05=float((np.sign(big.move) == np.sign(big.target)).mean()), n_cells=int(len(g)), n_big=int(len(big)))
pc = np.array([per_model_corr[m]["pearson"] for m in models])
rho_c, p_c, lo_c, hi_c = spearman_boot(ECI, pc)
pooled_r = pearsonr(nc.move, nc.target).statistic
pooled_slope = np.polyfit(nc.target, nc.move, 1)[0]
# pooled panel: one point per cell = mean move over the models that answered it, against the true shift
assert nc.groupby("item").target.nunique().max() == 1
cell = nc.groupby("item").agg(target=("target", "first"), move=("move", "mean"), n_models=("move", "size"),
                              T=("T", "first"), block=("block", "first")).reset_index()
cell_r = pearsonr(cell.move, cell.target).statistic
cell_slope = float(np.polyfit(cell.target, cell.move, 1)[0])
bins = np.array([-0.6, -0.3, -0.2, -0.15, -0.1, -0.06, -0.03, -0.01, 0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.6])
cell["tbin"] = pd.cut(cell.target, bins)
bm = cell.groupby("tbin", observed=True).agg(target=("target", "mean"), move=("move", "mean"), n=("move", "size"),
                                              se=("move", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan)).reset_index()
fig, ax = plt.subplots(figsize=(3.2, 3.2))
fig.subplots_adjust(left=0.18, right=0.97, top=0.95, bottom=0.15)
lim = 0.62
ax.axhline(0, color=LGREY, lw=0.6, zorder=1); ax.axvline(0, color=LGREY, lw=0.6, zorder=1)
ax.plot([-lim, lim], [-lim, lim], color=GREY, ls="--", dashes=(4, 2), lw=0.8, zorder=2, label="move = true shift")
ax.scatter(nc.target, nc.move, s=2, color=LGREY, alpha=0.4, linewidths=0, zorder=2)
ax.scatter([], [], s=8, color=LGREY, linewidths=0, label="one model, one cell")  # legend proxy, visible size
ax.scatter(cell.target, cell.move, s=7, color=GREEN, alpha=0.8, linewidths=0, zorder=3, label="cell: mean over 24 models")
ax.errorbar(bm.target, bm.move, yerr=1.96 * bm.se, color=ORANGE, lw=1.3, marker="o", ms=3, capsize=0, zorder=4,
            label="cell means by true-shift bin ($\\pm$1.96 s.e.)")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xticks([-0.6, -0.3, 0, 0.3, 0.6]); ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
ax.set_xlabel("True shift $p(Y\\,|\\,X) - p(Y)$")
ax.set_ylabel("Model move $p_2 - p_1$ after the reveal")
ax.set_aspect("equal")
leg = ax.legend(loc="lower right", fontsize=5.8, handlelength=1.8, frameon=True, framealpha=0.9, edgecolor="none",
                fancybox=False, borderaxespad=0.3, labelspacing=0.35)
ax.text(0.03, 0.84, f"{len(cell)} cells, {len(nc):,} model-cell pairs\ncell means: Pearson $r = {cell_r:.2f}$, OLS slope {cell_slope:.2f}",
        transform=ax.transAxes, fontsize=6.5, color=GREEN, ha="left", va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
ax.text(0.02, 0.98, PROV, transform=ax.transAxes, fontsize=5.5, color=GREY, ha="left", va="top", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.0))
pooled_stats = dict(
    cells=dict(n=int(len(cell)), pearson=float(cell_r), ols_slope=cell_slope,
               mean_abs_move=float(cell.move.abs().mean()), mean_abs_target=float(cell.target.abs().mean()),
               what="one point per natcond cell: x = p(Y|X) - p(Y) from the 1,000 replays, y = mean over models of p2 - p1"),
    all_pairs=dict(n=int(len(nc)), pearson=float(pooled_r), ols_slope=float(pooled_slope),
                   mean_abs_move=float(nc.move.abs().mean()), mean_abs_target=float(nc.target.abs().mean()),
                   what="every (model, cell) pair with a parsed turn-2 answer; grey background points"),
)
save(fig, "fig_freeciv_natcond_update", dict(
    source=rel(ITEMS), pooled=pooled_stats,
    cells=dict(item=cell.item.tolist(), T=cell["T"].astype(int).tolist(), block=cell.block.tolist(),
               target=cell.target.round(6).tolist(), mean_move=cell.move.round(6).tolist(), n_models=cell.n_models.tolist()),
    background_points=dict(
        what="grey points: (model, cell) pairs; model_index indexes `models`, cell_index indexes `cells.item`; x = cells.target[cell_index], y = move",
        models=models, model_index=[models.index(m) for m in nc.model],
        cell_index=cell.reset_index().set_index("item")["index"].loc[nc.item].astype(int).tolist(),
        move=nc.move.round(4).tolist()),
    binned=dict(bin_edges=bins.tolist(), target_mean=bm.target.tolist(), move_mean=bm.move.tolist(), n=bm.n.tolist(), se=bm.se.tolist()),
    per_model=per_model_corr,
    per_model_pearson_vs_eci=dict(spearman_rho=rho_c, p=p_c, ci95_models=[lo_c, hi_c],
                                  what="Spearman of each model's Pearson corr(move, target) against ECI"),
    provisional=PROV))
summary["fig_freeciv_natcond_update"] = dict(
    pooled=pooled_stats, binned=dict(bin_edges=bins.tolist(), target_mean=bm.target.tolist(), move_mean=bm.move.tolist(), n=bm.n.tolist(), se=bm.se.tolist()),
    per_model=per_model_corr, per_model_pearson_vs_eci=dict(spearman_rho=rho_c, p=p_c, ci95_models=[lo_c, hi_c]))
print(f"natcond update: cells r={cell_r:.3f} slope={cell_slope:.3f}; all pairs r={pooled_r:.3f} slope={pooled_slope:.3f}; "
      f"per-model r range {pc.min():.3f}-{pc.max():.3f}; rho(r,ECI)={rho_c:+.3f} p={p_c:.3f} CI=[{lo_c:+.2f},{hi_c:+.2f}]")

# ------------------------------------------------------------- reference constants for the text
bank_u = uniq[uniq.set == "bank"]; tails_u = uniq[uniq.set == "tails"]
summary["reference_forecasters"] = dict(
    bank_flat_0p5_excess=float(np.mean((0.5 - bank_u.q) ** 2)),
    bank_var_q=float(np.var(bank_u.q)),
    tails_flat_0p05_bits=float(np.mean(kl_bits(tails_u.q.values, 0.05))),
    tails_flat_0p5_bits=float(np.mean(kl_bits(tails_u.q.values, 0.5))),
    n_bank_items=int(len(bank_u)), n_tails_items=int(len(tails_u)),
)
with open(DATADIR / "freeciv_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print("wrote", DATADIR / "freeciv_summary.json")
