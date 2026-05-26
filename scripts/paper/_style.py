"""Shared matplotlib style and model metadata for paper figures.

Paper figures live under `scripts/paper/`. They share font sizes, colors,
model display names, and model ordering through this module so a single
edit re-styles everything consistently.

Output format is PNG (the user will insert into Overleaf manually).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = REPO_ROOT / "data" / "evaluations" / "plots" / "paper"
DATA_DIR = REPO_ROOT / "data"

FIGURE_FORMAT = "png"
FIGURE_DPI = 200


# The curated set is the 9 models that have full coverage across every run
# (binary uncond, continuous uncond, H0 binary, H0 continuous, archived
# republic + gold500 conditional). They are the 10 models in the archived
# conditional runs minus Claude Opus 4.5 (which is missing from binary/H0).
CURATED_MODELS: list[str] = [
    "anthropic/claude-sonnet-4-5-20250929",
    "openai/gpt-5.1-2025-11-13",
    "openai/gpt-5-2025-08-07",
    "openai/gpt-5-mini-2025-08-07",
    "openai/gpt-4.1-2025-04-14",
    "openai/o3-2025-04-16",
    "google/gemini-3-pro-preview",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
]


# Distinct color per curated model so all 9 lines are readable on Fig 3.
# Within each lab the hues stay related; across labs they're distinct.
CURATED_LINE_COLORS: dict[str, str] = {
    "anthropic/claude-sonnet-4-5-20250929": "#C2410C",  # burnt orange
    "openai/gpt-5.1-2025-11-13":            "#0F766E",  # dark teal
    "openai/gpt-5-2025-08-07":              "#10B981",  # emerald
    "openai/gpt-5-mini-2025-08-07":         "#6EE7B7",  # mint
    "openai/gpt-4.1-2025-04-14":            "#B45309",  # amber
    "openai/o3-2025-04-16":                 "#A855F7",  # purple (reasoning)
    "google/gemini-3-pro-preview":          "#1E3A8A",  # navy
    "google/gemini-2.5-pro":                "#3B82F6",  # royal blue
    "google/gemini-2.5-flash":              "#93C5FD",  # sky blue
}


MODEL_DISPLAY_NAMES: dict[str, str] = {
    "anthropic/claude-opus-4-5-20251101": "Claude Opus 4.5",
    "anthropic/claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "anthropic/claude-sonnet-4-20250514": "Claude Sonnet 4",
    "anthropic/claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "anthropic/claude-3-haiku-20240307": "Claude 3 Haiku",
    "openai/gpt-5-2025-08-07": "GPT-5",
    "openai/gpt-5.1-2025-11-13": "GPT-5.1",
    "openai/gpt-5-mini-2025-08-07": "GPT-5 mini",
    "openai/gpt-5-nano-2025-08-07": "GPT-5 nano",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "openai/gpt-4o": "GPT-4o",
    "openai/gpt-3.5-turbo-0125": "GPT-3.5 Turbo",
    "openai/o3-2025-04-16": "o3",
    "openai/o3-mini-2025-01-31": "o3-mini",
    "openai/o4-mini-2025-04-16": "o4-mini",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "google/gemini-2.0-flash-lite-001": "Gemini 2.0 Flash Lite",
    "xai/grok-4-0709": "Grok 4",
    "xai/grok-4-fast-reasoning": "Grok 4 Fast (reasoning)",
    "xai/grok-4-fast-non-reasoning": "Grok 4 Fast",
    "xai/grok-4-1-fast-reasoning": "Grok 4.1 Fast (reasoning)",
    "xai/grok-4-1-fast-non-reasoning": "Grok 4.1 Fast",
    "together_ai/deepseek-ai/DeepSeek-V3": "DeepSeek V3",
    "together_ai/deepseek-ai/DeepSeek-V3.1": "DeepSeek V3.1",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama 3.3 70B",
    "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B",
    "mistral/mistral-large-2407": "Mistral Large 2407",
    "mistral/mistral-large-2411": "Mistral Large 2411",
    "mistral/mistral-large-latest": "Mistral Large (latest)",
}


# Lab-grouped color palette. We re-use the same hue family for models from
# the same lab so labs are visually distinguishable on multi-model charts.
LAB_COLORS: dict[str, str] = {
    "anthropic": "#D97757",   # warm orange
    "openai":    "#10A37F",   # green
    "google":    "#4285F4",   # blue
    "xai":       "#7E57C2",   # purple
    "deepseek":  "#E53935",   # red
    "meta":      "#8B5A2B",   # brown
    "mistral":   "#616161",   # gray
}


def lab_for(model_id: str) -> str:
    if model_id.startswith("anthropic/"):
        return "anthropic"
    if model_id.startswith("openai/"):
        return "openai"
    if model_id.startswith("google/"):
        return "google"
    if model_id.startswith("xai/"):
        return "xai"
    if "deepseek" in model_id.lower():
        return "deepseek"
    if "llama" in model_id.lower() or "meta-llama" in model_id.lower():
        return "meta"
    if "mistral" in model_id.lower() or "mixtral" in model_id.lower():
        return "mistral"
    return "mistral"


def color_for(model_id: str) -> str:
    return LAB_COLORS[lab_for(model_id)]


def line_color_for(model_id: str) -> str:
    """Per-curated-model line color. Falls back to lab color for non-curated."""
    return CURATED_LINE_COLORS.get(model_id, color_for(model_id))


def display_name(model_id: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


def apply_style() -> None:
    """Apply consistent matplotlib styling for paper figures."""
    mpl.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "lines.linewidth": 2.2,
        "lines.markersize": 6.5,
    })


def save_figure(fig, name: str) -> Path:
    """Save a figure to PLOTS_DIR/{name}.{FIGURE_FORMAT}."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(out)
    plt.close(fig)
    return out
