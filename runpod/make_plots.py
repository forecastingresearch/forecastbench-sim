"""Generate plots for the SFT/RL writeup. Reads from tmp/rl_session/.

Outputs PNGs to runpod/plots/.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tmp" / "rl_session"
OUT = ROOT / "runpod" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# Hardcoded baseline numbers from console output (not re-pulled from pod).
BASELINES = {
    "Qwen3-8B greedy":      {"fb": (0.1822, 146/401),  "freeciv": (0.1970, 55/57)},
    "Qwen3-8B t=0.6":       {"fb": (0.1735, 372/401),  "freeciv": (0.2297, 48/57)},
    "R1-Distill-Qwen-7B":   {"fb": (0.2042, 389/401),  "freeciv": (0.2469, 39/57)},
}

# Load SFT/RL eval results
ep1 = json.load(open(DATA / "post_sft" / "ep1" / "eval.json"))
ep4 = json.load(open(DATA / "post_sft" / "ep4" / "eval.json"))
rl  = json.load(open(DATA / "post_rl" / "eval.json"))


def stats(eval_json, suite):
    s = eval_json["results"][suite]["stats"]
    return s["brier"], s["n_parsed"] / s["n"]


SFT_RL = {
    "SFT ep1": {"fb": stats(ep1, "forecastbench"), "freeciv": stats(ep1, "freeciv_val")},
    "SFT ep4": {"fb": stats(ep4, "forecastbench"), "freeciv": stats(ep4, "freeciv_val")},
    "Post-RL":  {"fb": stats(rl, "forecastbench"),  "freeciv": stats(rl, "freeciv_val")},
}

# =============================================================================
# PLOT 1: Headline Brier comparison (per-suite, full-set numbers)
# =============================================================================

conditions = ["Qwen3-8B\ngreedy", "Qwen3-8B\nt=0.6", "R1-Distill\nt=0.6",
              "SFT ep1", "SFT ep4", "Post-RL"]
fb_brier   = [BASELINES["Qwen3-8B greedy"]["fb"][0], BASELINES["Qwen3-8B t=0.6"]["fb"][0],
              BASELINES["R1-Distill-Qwen-7B"]["fb"][0],
              SFT_RL["SFT ep1"]["fb"][0], SFT_RL["SFT ep4"]["fb"][0], SFT_RL["Post-RL"]["fb"][0]]
fc_brier   = [BASELINES["Qwen3-8B greedy"]["freeciv"][0], BASELINES["Qwen3-8B t=0.6"]["freeciv"][0],
              BASELINES["R1-Distill-Qwen-7B"]["freeciv"][0],
              SFT_RL["SFT ep1"]["freeciv"][0], SFT_RL["SFT ep4"]["freeciv"][0],
              SFT_RL["Post-RL"]["freeciv"][0]]
fb_parse   = [BASELINES["Qwen3-8B greedy"]["fb"][1], BASELINES["Qwen3-8B t=0.6"]["fb"][1],
              BASELINES["R1-Distill-Qwen-7B"]["fb"][1],
              SFT_RL["SFT ep1"]["fb"][1], SFT_RL["SFT ep4"]["fb"][1], SFT_RL["Post-RL"]["fb"][1]]
fc_parse   = [BASELINES["Qwen3-8B greedy"]["freeciv"][1], BASELINES["Qwen3-8B t=0.6"]["freeciv"][1],
              BASELINES["R1-Distill-Qwen-7B"]["freeciv"][1],
              SFT_RL["SFT ep1"]["freeciv"][1], SFT_RL["SFT ep4"]["freeciv"][1],
              SFT_RL["Post-RL"]["freeciv"][1]]

# constant-baseline Brier for the headline bars (taken from the actual eval JSON for each)
# For Freeciv, the constant-baseline is roughly stable at ~0.150; for FB ~0.158 (best baseline)
FB_CONST = 0.158       # from Qwen3-8B t=0.6
FC_CONST = 0.151       # from Qwen3-8B t=0.6 / SFT ep4

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
colors = ["#888", "#4a8", "#888", "#fbb", "#d44", "#26a"]

for ax, brs, parses, title, const in [
    (axes[0], fb_brier, fb_parse, "ForecastBench (n=401)", FB_CONST),
    (axes[1], fc_brier, fc_parse, "Freeciv-val held-out (n=57)", FC_CONST),
]:
    bars = ax.bar(range(len(conditions)), brs, color=colors, edgecolor="#222", linewidth=0.8)
    ax.axhline(const, color="black", linestyle="--", linewidth=1.0,
               label=f"constant-baseline Brier = {const:.3f}")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Brier (lower is better)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.set_ylim(0, max(brs) * 1.25)
    # Annotate bars with parse rate
    for b, br, pr in zip(bars, brs, parses):
        ax.annotate(f"{br:.3f}\n{pr:.0%} parsed", xy=(b.get_x() + b.get_width()/2, br),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5)
fig.suptitle("Brier across conditions — full-suite numbers (different parsed subsets)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "01_headline_brier.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / '01_headline_brier.png'}")

# =============================================================================
# PLOT 2: Apples-to-apples Brier on common-parsed subset
# =============================================================================

# Compute common parsed sets across SFT ep1, ep4, RL
def common_brier(d_list, suite):
    rowsets = []
    rowmaps = []
    for d in d_list:
        rs = d["results"][suite]["rows"]
        rowsets.append(set(r["qid"] for r in rs if r["pred"] is not None))
        rowmaps.append({r["qid"]: r for r in rs})
    common = rowsets[0].intersection(*rowsets[1:])
    targets = []
    out = []
    for d_idx, rm in enumerate(rowmaps):
        brs = [(rm[q]["pred"] - rm[q]["target"]) ** 2 for q in common]
        out.append(sum(brs) / len(brs))
        if d_idx == 0:
            targets = [rm[q]["target"] for q in common]
    base_rate = sum(targets) / len(targets)
    const = sum((base_rate - t) ** 2 for t in targets) / len(targets)
    return out, const, len(common)


fb_common, fb_const_c, fb_n = common_brier([ep1, ep4, rl], "forecastbench")
fc_common, fc_const_c, fc_n = common_brier([ep1, ep4, rl], "freeciv_val")

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
labels = ["SFT ep1", "SFT ep4", "Post-RL"]
colors2 = ["#fbb", "#d44", "#26a"]

for ax, vals, const, n, title in [
    (axes[0], fb_common, fb_const_c, fb_n, "ForecastBench common-parsed"),
    (axes[1], fc_common, fc_const_c, fc_n, "Freeciv-val common-parsed"),
]:
    bars = ax.bar(labels, vals, color=colors2, edgecolor="#222", linewidth=0.8)
    ax.axhline(const, color="black", linestyle="--", linewidth=1.0,
               label=f"constant-baseline = {const:.3f}")
    ax.set_ylabel("Brier")
    ax.set_title(f"{title}\n(n={n})")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", xy=(b.get_x() + b.get_width()/2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(max(vals), const) * 1.2)

fig.suptitle("Apples-to-apples: Brier on common-parsed subset across SFT/RL conditions", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "02_common_parsed_brier.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / '02_common_parsed_brier.png'}")

# =============================================================================
# PLOT 3: SFT training curve (loss + eval_loss)
# =============================================================================

sft_metrics = [json.loads(l) for l in open(DATA / "sft" / "sft_metrics.jsonl") if l.strip()]
train_steps = [e["step"] for e in sft_metrics if "loss" in e and "eval_loss" not in e]
train_loss  = [e["loss"]  for e in sft_metrics if "loss" in e and "eval_loss" not in e]
eval_entries = [e for e in sft_metrics if "eval_loss" in e]
eval_steps = [e["step"] for e in eval_entries]
eval_loss  = [e["eval_loss"] for e in eval_entries]

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(train_steps, train_loss, color="#26a", linewidth=1.0, alpha=0.7,
        label="train loss (per-step)")
ax.set_yscale("log")
ax.set_xlabel("SFT step")
ax.set_ylabel("loss (log scale)")
ax.set_title("SFT loss curve: train loss collapses to ~0 by epoch 2; val CE rises (overfit signal)")

# Eval loss on secondary y-axis
ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
ax2.plot(eval_steps, eval_loss, color="#d44", marker="o", linewidth=2.0, label="val CE (per epoch)")
ax2.set_ylabel("val cross-entropy")
ax2.set_ylim(0, max(eval_loss) * 1.2)

# Mark epoch boundaries
n_steps = max(train_steps)
for ep_step in [69, 138, 207, 276]:
    ax.axvline(ep_step, color="#aaa", linestyle=":", linewidth=0.7)
    ax.text(ep_step, ax.get_ylim()[1] * 0.8, f"ep{ep_step // 69}", fontsize=7,
            ha="right", color="#555")

# Annotate val CE values
for s, e in zip(eval_steps, eval_loss):
    ax2.annotate(f"{e:.3f}", xy=(s, e), xytext=(5, 5), textcoords="offset points",
                 color="#d44", fontsize=9, weight="bold")

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "03_sft_curves.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / '03_sft_curves.png'}")

# =============================================================================
# PLOT 4: RL reward curve
# =============================================================================

rl_state = json.load(open(DATA / "rl" / "checkpoint-300" / "trainer_state.json"))
hist = rl_state["log_history"]
rl_steps = [e["step"] for e in hist if "reward" in e]
rl_reward = [e["reward"] for e in hist if "reward" in e]
rl_reward_std = [e.get("reward_std", 0) for e in hist if "reward" in e]
rl_kl = [e.get("kl", 0) for e in hist if "reward" in e]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(rl_steps, rl_reward, color="#26a", marker="o", markersize=3, linewidth=1.2,
        label="reward (avg per log step)")
# fill +/- 1 std
import numpy as np
rl_steps_a = np.array(rl_steps)
rl_reward_a = np.array(rl_reward)
rl_std_a = np.array(rl_reward_std)
ax.fill_between(rl_steps_a, rl_reward_a - rl_std_a, rl_reward_a + rl_std_a, color="#26a", alpha=0.15,
                label="reward ± 1 std")
ax.axhline(0, color="black", linestyle="--", linewidth=0.7, label="0 (perfect Brier)")
ax.set_xlabel("RL step")
ax.set_ylabel("reward (-Brier × weight)")
ax.set_title("GRPO reward over 300 steps — essentially flat")
ax.legend(loc="lower right", fontsize=8)

ax = axes[1]
ax.plot(rl_steps, rl_kl, color="#d44", marker="o", markersize=3, linewidth=1.2)
ax.set_yscale("symlog")
ax.set_xlabel("RL step")
ax.set_ylabel("KL to SFT reference (symlog)")
ax.set_title("KL divergence from SFT reference policy")

fig.tight_layout()
fig.savefig(OUT / "04_rl_reward_kl.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / '04_rl_reward_kl.png'}")

# =============================================================================
# PLOT 5: Per-template Brier on Freeciv-val (SFT ep4 vs Post-RL)
# =============================================================================

def per_template(d, suite="freeciv_val"):
    rows = d["results"][suite]["rows"]
    pt = defaultdict(list)
    for r in rows:
        if r["pred"] is None:
            continue
        tmpl = r.get("meta", {}).get("template_id", "?")
        pt[tmpl].append((r["pred"] - r["target"]) ** 2)
    return {k: (sum(v)/len(v), len(v)) for k, v in pt.items()}


pt_ep4 = per_template(ep4)
pt_rl  = per_template(rl)
# Use union of templates
tmpls = sorted(set(pt_ep4) | set(pt_rl))
ep4_briers = [pt_ep4.get(t, (None, 0))[0] for t in tmpls]
rl_briers  = [pt_rl.get(t, (None, 0))[0] for t in tmpls]
ep4_ns = [pt_ep4.get(t, (None, 0))[1] for t in tmpls]

fig, ax = plt.subplots(figsize=(11, 4.5))
x = list(range(len(tmpls)))
width = 0.35
b1 = ax.bar([xi - width/2 for xi in x], ep4_briers, width, color="#d44",
            edgecolor="#222", linewidth=0.7, label="SFT ep4")
b2 = ax.bar([xi + width/2 for xi in x], rl_briers, width, color="#26a",
            edgecolor="#222", linewidth=0.7, label="Post-RL")
ax.axhline(0.151, color="black", linestyle="--", linewidth=1.0,
           label="constant baseline (0.151)")
ax.set_xticks(x)
ax.set_xticklabels([t.replace("_", "\n") for t in tmpls], fontsize=8, rotation=0)
ax.set_ylabel("Brier")
ax.set_title("Freeciv-val per-template Brier — SFT ep4 vs Post-RL")
ax.legend(loc="upper left", fontsize=8)

# Annotate n
for xi, n in zip(x, ep4_ns):
    ax.annotate(f"n={n}", xy=(xi, 0), xytext=(0, -22), textcoords="offset points",
                ha="center", fontsize=7, color="#666")
fig.tight_layout()
fig.savefig(OUT / "05_freeciv_per_template.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / '05_freeciv_per_template.png'}")

# =============================================================================
# PLOT 6: Prediction calibration scatter — SFT ep4 on Freeciv-val
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, d, name in [
    (axes[0], ep4, "SFT ep4"),
    (axes[1], rl,  "Post-RL"),
]:
    rows = d["results"]["freeciv_val"]["rows"]
    parsed = [r for r in rows if r["pred"] is not None]
    preds = [r["pred"] for r in parsed]
    targets = [r["target"] for r in parsed]
    # Color by template
    tmpls = sorted(set(r.get("meta",{}).get("template_id","?") for r in parsed))
    cmap = plt.colormaps["tab10"]
    for i, tmpl in enumerate(tmpls):
        ti = [j for j, r in enumerate(parsed)
              if r.get("meta",{}).get("template_id","?") == tmpl]
        ax.scatter([preds[j] for j in ti], [targets[j] for j in ti],
                   color=cmap(i % 10), label=tmpl, s=40, alpha=0.7,
                   edgecolor="#222", linewidth=0.4)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect calibration")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("model prediction")
    ax.set_ylabel("p_mc (target)")
    ax.set_title(f"{name} predictions vs targets on Freeciv-val (n={len(parsed)})")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.12), ncol=5, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "06_calibration_scatter.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / '06_calibration_scatter.png'}")

print("\nAll plots saved to:", OUT)

# =============================================================================
# SESSION 2 PLOTS
# =============================================================================

S2 = ROOT / "tmp" / "session2" / "results"
OUT2 = OUT / "v2"


def _load_eval(path: str):
    return json.load(open(S2 / path))


def _suite_stats(eval_json, suite: str):
    result = eval_json.get("results", {}).get(suite)
    if not result:
        return None
    stats = result["stats"]
    return {
        "brier": stats["brier"],
        "const": stats["brier_constant_baseline"],
        "parse": stats["n_parsed"] / stats["n"],
        "n": stats["n"],
        "n_parsed": stats["n_parsed"],
    }


def _plot_session2():
    OUT2.mkdir(parents=True, exist_ok=True)

    key_conditions = [
        ("Base\nthink t=0", "E1/chat_t0/eval.json"),
        ("Base\nthink t=.6", "E1/chat_t06/eval.json"),
        ("Base no-think\n+ guided", "E4/E1_nothink_guided/chat_t06/eval.json"),
        ("E2 SFT\nbest", "E2E3/E2/eval_best_chat_t06.json"),
        ("E3 RL\nfrom SFT", "E2E3/E3/eval_chat_t06.json"),
        ("E5 RL\nfrom base", "E5/E5/eval_chat_t06.json"),
    ]
    loaded = [(label, _load_eval(path)) for label, path in key_conditions]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, suite, title in [
        (axes[0], "forecastbench", "ForecastBench"),
        (axes[1], "freeciv_val", "Freeciv-val"),
    ]:
        labels, briers, consts, parses = [], [], [], []
        for label, data in loaded:
            stat = _suite_stats(data, suite)
            if stat is None:
                continue
            labels.append(label)
            briers.append(stat["brier"])
            consts.append(stat["const"])
            parses.append(stat["parse"])
        colors = ["#777", "#999", "#2a9d8f", "#e76f51", "#8d5a97", "#457b9d"][:len(labels)]
        bars = ax.bar(range(len(labels)), briers, color=colors, edgecolor="#222", linewidth=0.7)
        ax.axhline(consts[-1], color="black", linestyle="--", linewidth=1.0,
                   label=f"constant baseline = {consts[-1]:.3f}")
        ax.set_title(title)
        ax.set_ylabel("Brier (lower is better)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, max(max(briers), consts[-1]) * 1.25)
        ax.legend(loc="upper left", fontsize=8)
        for bar, brier, parse in zip(bars, briers, parses):
            ax.annotate(f"{brier:.3f}\n{parse:.0%}",
                        xy=(bar.get_x() + bar.get_width() / 2, brier),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("Session 2 headline: parse fixed, calibration still worse than constant", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT2 / "01_session2_headline.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT2 / '01_session2_headline.png'}")

    e2_ckpts = [69, 138, 207, 276]
    e2_vals = []
    e2_const = None
    for ckpt in e2_ckpts:
        stat = _suite_stats(_load_eval(f"E2E3/E2/eval_checkpoint-{ckpt}/eval.json"), "freeciv_val")
        e2_vals.append(stat["brier"])
        e2_const = stat["const"]

    e4_ckpts = [56, 112, 168, 224]
    e4_vals = []
    e4_const = None
    for ckpt in e4_ckpts:
        stat = _suite_stats(_load_eval(f"E4/E4/eval_checkpoint-{ckpt}/eval.json"), "freeciv_val")
        e4_vals.append(stat["brier"])
        e4_const = stat["const"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot([1, 2, 3, 4], e2_vals, marker="o", color="#e76f51", linewidth=2)
    axes[0].axhline(e2_const, color="black", linestyle="--", linewidth=1.0,
                    label=f"constant = {e2_const:.3f}")
    axes[0].set_title("E2 SFT Freeciv-val Brier")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("Brier")
    axes[0].set_xticks([1, 2, 3, 4])
    axes[0].legend(fontsize=8)
    for x, y in zip([1, 2, 3, 4], e2_vals):
        axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6),
                         ha="center", fontsize=8)

    axes[1].plot([1, 2, 3, 4], e4_vals, marker="o", color="#457b9d", linewidth=2)
    axes[1].axhline(e4_const, color="black", linestyle="--", linewidth=1.0,
                    label=f"constant = {e4_const:.3f}")
    axes[1].set_title("E4 held-out-template Brier")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Brier")
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].legend(fontsize=8)
    for x, y in zip([1, 2, 3, 4], e4_vals):
        axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6),
                         ha="center", fontsize=8)
    fig.suptitle("SFT curves under no-thinking/guided parse-fixed eval", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT2 / "02_sft_brier_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT2 / '02_sft_brier_curves.png'}")

    e4_matrix = defaultdict(dict)
    e4_counts = defaultdict(dict)
    e4_consts = defaultdict(dict)
    for ckpt in e4_ckpts:
        data = _load_eval(f"E4/E4/eval_checkpoint-{ckpt}/eval.json")
        rows = data["results"]["freeciv_val"]["rows"]
        by_template = defaultdict(list)
        by_target = defaultdict(list)
        for row in rows:
            template = row.get("meta", {}).get("template_id", "?")
            if row["pred"] is None:
                continue
            by_template[template].append((row["pred"] - row["target"]) ** 2)
            by_target[template].append(row["target"])
        for template, vals in by_template.items():
            e4_matrix[template][ckpt] = sum(vals) / len(vals)
            e4_counts[template][ckpt] = len(vals)
            base_rate = sum(by_target[template]) / len(by_target[template])
            e4_consts[template][ckpt] = sum((base_rate - t) ** 2 for t in by_target[template]) / len(by_target[template])

    templates = sorted(e4_matrix)
    x = list(range(len(templates)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    colors = ["#84a59d", "#f6bd60", "#f28482", "#6d597a"]
    for idx, ckpt in enumerate(e4_ckpts):
        vals = [e4_matrix[t][ckpt] for t in templates]
        offs = [xi + (idx - 1.5) * width for xi in x]
        ax.bar(offs, vals, width=width, color=colors[idx], edgecolor="#222", linewidth=0.6,
               label=f"epoch {idx + 1}")
    const_vals = [e4_consts[t][56] for t in templates]
    ax.scatter(x, const_vals, color="black", marker="_", s=180, linewidths=2.0,
               label="template constant")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{t.replace('_', chr(10))}\nn={e4_counts[t][56]}" for t in templates],
        fontsize=8,
    )
    ax.set_ylabel("Brier")
    ax.set_title("E4 held-out-template Brier by template")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT2 / "03_e4_per_template.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT2 / '03_e4_per_template.png'}")

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for label, path, color in [
        ("E3 from SFT", "E2E3/E3/checkpoint-300/trainer_state.json", "#8d5a97"),
        ("E5 from base", "E5/E5/checkpoint-300/trainer_state.json", "#457b9d"),
    ]:
        state = json.load(open(S2 / path))
        points = [(e["step"], e["reward"]) for e in state["log_history"] if "reward" in e]
        ax.plot([p[0] for p in points], [p[1] for p in points],
                marker="o", markersize=3, linewidth=1.4, color=color, label=label)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("logged reward")
    ax.set_title("Session 2 RL reward curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT2 / "04_rl_rewards.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT2 / '04_rl_rewards.png'}")

    print("Session 2 plots saved to:", OUT2)


if S2.exists():
    _plot_session2()
