"""Fig 1 — ForecastBench-Sim benchmark schematic.

Top row is the pipeline: simulate → world report → (LLM forecaster) →
forecast questions → resolution. Both forecast questions and resolution
flow down into Scoring (a 5th pipeline step). Matched inputs and Controlled
variants sit in the bottom row as features (not steps), styled distinctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.paper._style import REPO_ROOT, apply_style, save_figure


BOX = "#F8FAFC"
EDGE = "#334155"
MUTED = "#64748B"
BLUE = "#DBEAFE"
GREEN = "#DCFCE7"
AMBER = "#FEF3C7"
RED = "#FEE2E2"
PURPLE = "#F3E8FF"

BOT_ICON_PATH = REPO_ROOT / "assets" / "icons" / "bot.png"


def _step_box(ax, xy, w, h, title, body, face, title_offset=0.045):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.0, edgecolor=EDGE, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - title_offset, title,
            ha="center", va="top", fontweight="bold", fontsize=13, color="#0F172A")
    ax.text(x + w / 2, y + h / 2 - 0.045, body,
            ha="center", va="center", fontsize=11, linespacing=1.4, color="#0F172A")


def _feature_box(ax, xy, w, h, title, body, face):
    """Visually distinct from step boxes: dashed border, muted title."""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.020",
        linewidth=1.0, edgecolor=MUTED, facecolor=face, linestyle=(0, (4, 2)),
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.05, title,
            ha="center", va="top", fontweight="bold", fontsize=11.5, color=MUTED)
    ax.text(x + w / 2, y + h / 2 - 0.04, body,
            ha="center", va="center", fontsize=10.5, linespacing=1.35, color="#334155")


def _arrow(ax, start, end, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.2, color=EDGE,
        connectionstyle=f"arc3,rad={rad}",
    ))


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(13.5, 4.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ----- Top row: pipeline steps -------------------------------------------
    # Every step-to-step arrow is the same length `g`. Between boxes 2 and 3
    # the gap is exactly two arrows + the bot icon's width.
    step_w = 0.165   # slightly narrower so the rounded edges of boxes 1 and 4
                     # are not clipped at the left/right edges of the figure.
    step_h = 0.28
    step_y = 0.555
    g = 0.055
    bot_half = 0.028                             # half-width of the bot icon
    bot_gap = 2 * g + 2 * bot_half               # 0.166

    box1_x = 0.035   # small left margin so the box edge is not clipped
    box2_x = box1_x + step_w + g                # 0.245
    box3_x = box2_x + step_w + bot_gap          # 0.581
    box4_x = box3_x + step_w + g                # 0.806

    _step_box(ax, (box1_x, step_y), step_w, step_h,
              "1. Simulated world",
              "Freeciv game rollouts\nwith multiple civilizations\nand hidden future states",
              BLUE)
    _step_box(ax, (box2_x, step_y), step_w, step_h,
              "2. World report",
              "Turn-60 snapshot:\nstate tables, histories,\nmap/report evidence",
              GREEN)
    _step_box(ax, (box3_x, step_y), step_w, step_h,
              "3. Forecast questions",
              "Binary events and\ncontinuous quantities\nat H1 ... H7",
              AMBER)
    _step_box(ax, (box4_x, step_y), step_w, step_h,
              "4. Resolution",
              "Continue simulation\nto future turns and\nread ground truth",
              RED)

    # Arrow y is the vertical centre of the step boxes.
    arrow_y = step_y + step_h / 2

    # 1 -> 2 and 3 -> 4: standard short arrows in the small gap.
    _arrow(ax, (box1_x + step_w, arrow_y), (box2_x, arrow_y))
    _arrow(ax, (box3_x + step_w, arrow_y), (box4_x, arrow_y))

    # ----- Bot icon between boxes 2 and 3 ------------------------------------
    bot_cx = (box2_x + step_w + box3_x) / 2

    bot_img = mpimg.imread(str(BOT_ICON_PATH))
    imagebox = OffsetImage(bot_img, zoom=0.55)
    imagebox.image.axes = ax
    ab = AnnotationBbox(
        imagebox, (bot_cx, arrow_y),
        frameon=False, pad=0.0, xycoords="data",
    )
    ax.add_artist(ab)
    ax.text(bot_cx, arrow_y - 0.10, "LLM forecaster",
            ha="center", va="top", fontsize=10, color=MUTED, fontweight="bold")

    # 2 -> bot and bot -> 3 arrows, same length as the 1↔2 / 3↔4 arrows.
    _arrow(ax, (box2_x + step_w, arrow_y), (bot_cx - bot_half, arrow_y))
    _arrow(ax, (bot_cx + bot_half, arrow_y), (box3_x, arrow_y))

    # ----- Bottom row: features + Scoring step -------------------------------
    feat_h = 0.25
    feat_y = 0.14   # bottom row sits a bit higher; diagonal arrows are allowed
                    # to intrude slightly into the top of the Scoring box.

    # Scoring sits horizontally between boxes 3 and 4 in the top row.
    # Title offset is smaller than the default so the "5. Scoring" heading
    # sits a little higher inside the (shorter) box.
    scoring_w = 0.25
    scoring_x = (box3_x + step_w / 2 + box4_x + step_w / 2) / 2 - scoring_w / 2
    _step_box(ax, (scoring_x, feat_y), scoring_w, feat_h,
              "5. Scoring",
              "Brier for binary forecasts\nCRPS for quantile forecasts\nH0 checks report reading",
              PURPLE, title_offset=0.030)

    # Diagonal arrows from boxes 3 and 4 into Scoring (now shorter because
    # the bottom row is closer to the top row).
    box3_bottom = (box3_x + step_w / 2, step_y)
    box4_bottom = (box4_x + step_w / 2, step_y)
    # End the arrows slightly inside the top of the scoring box so they
    # visibly reach into it rather than stopping at the edge.
    arrow_end_y = feat_y + feat_h - 0.020
    scoring_top_left = (scoring_x + scoring_w * 0.30, arrow_end_y)
    scoring_top_right = (scoring_x + scoring_w * 0.70, arrow_end_y)
    _arrow(ax, box3_bottom, scoring_top_left, rad=0.10)
    _arrow(ax, box4_bottom, scoring_top_right, rad=-0.10)

    # Features: same width as top-row boxes, same height as Scoring, with
    # a visible gap between them. Shifted right by half a top-box width so
    # they sit between boxes 1/2 and the Scoring block rather than directly
    # under boxes 1 and 2.
    feat_w = step_w           # 0.17
    shift = step_w * 0.5      # half a top-box width
    matched_x = box1_x + shift
    controlled_x = box2_x + shift
    _feature_box(ax, (matched_x, feat_y), feat_w, feat_h,
                 "Matched inputs",
                 "Same report format\nand question templates\nacross model comparisons",
                 BOX)
    _feature_box(ax, (controlled_x, feat_y), feat_w, feat_h,
                 "Controlled variants",
                 "Mutate savegames for\nconditional / causal\npaired rollouts",
                 "#F0F9FF")

    # "Benchmark features" header, centred between the two feature boxes,
    # positioned a bit higher above them.
    features_centre_x = (matched_x + feat_w / 2 + controlled_x + feat_w / 2) / 2
    ax.text(features_centre_x, feat_y + feat_h + 0.040, "Benchmark features",
            ha="center", va="bottom", fontsize=12,
            fontweight="bold", color=MUTED, style="italic")

    # ----- Title -------------------------------------------------------------
    ax.text(
        0.5, 0.965,
        "ForecastBench-Sim evaluates forecasting in controlled simulated worlds",
        ha="center", va="top", fontsize=19, fontweight="bold",
    )

    out = save_figure(fig, "fig1_benchmark_schematic")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
