#!/usr/bin/env python3
"""
Compare ARC-AGI vs ForecastBench as predictors of CivBench rankings.

Supports either:
- precomputed CivBench CSV input, or
- on-the-fly weighted CivBench scoring from eval JSON + learned (horizon,template) weights.

Also produces per-seed rank heatmaps (unweighted and weighted) when eval JSON is provided.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# Model name mapping: CivBench ID -> (ForecastBench name, ARC-AGI best variant name)
# ARC-AGI uses best-performing variant per model family (including CoT)
MODEL_MAPPING = {
    "openai/gpt-3.5-turbo-0125": ("GPT-3.5-Turbo-0125", None),
    "openai/gpt-4o": ("GPT-4o-2024-05-13", "GPT-4o"),
    "openai/gpt-4.1-2025-04-14": ("GPT-4.1-2025-04-14", "GPT-4.1"),
    "openai/gpt-5-2025-08-07": ("GPT-5-2025-08-07", "GPT-5 (High)"),
    "openai/gpt-5-mini-2025-08-07": ("GPT-5-Mini-2025-08-07", "GPT-5 Mini (High)"),
    "openai/gpt-5-nano-2025-08-07": ("GPT-5-Nano-2025-08-07", "GPT-5 Nano (High)"),
    "openai/gpt-5.1-2025-11-13": ("GPT-5.1-2025-11-13", "GPT-5.1 (Thinking, High)"),
    "openai/o3-2025-04-16": ("O3-2025-04-16", "o3 (High)"),
    "openai/o3-mini-2025-01-31": ("O3-Mini-2025-01-31", "o3-mini (High)"),
    "openai/o4-mini-2025-04-16": ("O4-Mini-2025-04-16", "o4-mini (High)"),
    "anthropic/claude-3-haiku-20240307": ("Claude-3-Haiku-20240307", None),
    "anthropic/claude-haiku-4-5-20251001": ("Claude-Haiku-4-5-20251001", "Claude Haiku 4.5 (Thinking 32K)"),
    "google/gemini-2.0-flash-lite-001": ("Gemini-2.0-Flash-Lite-001", None),
    "google/gemini-2.5-flash": ("Gemini-2.5-Flash", "Gemini 2.5 Flash (Preview) (Thinking 16K)"),
    "xai/grok-4-fast-reasoning": ("Grok-4-Fast-Reasoning", "Grok 4 (Fast Reasoning)"),
    "xai/grok-4-1-fast-reasoning": ("Grok-4-1-Fast-Reasoning", "Grok 4 (Fast Reasoning)"),
    "mistral/mistral-large-2411": ("Mistral-Large-2411", None),
    "mistral/mistral-large-latest": ("Mistral-Large-Latest", None),
    "anthropic/claude-opus-4-6": ("Claude-Opus-4-6", None),
}

# ForecastBench score variants: key -> (score column, CI column, display label)
FORECASTBENCH_METRICS = {
    "overall": ("Overall (N)", "Overall 95% CI", "Overall (N)"),
    "market": ("Market (N)", "Market 95% CI", "Market (N)"),
    "dataset": ("Dataset (N)", "Dataset 95% CI", "Dataset (N)"),
}


def seed_sort_key(seed: str) -> tuple[int, int | str]:
    s = str(seed)
    m = re.fullmatch(r"seed(\d+)", s, flags=re.IGNORECASE)
    if m:
        return 0, int(m.group(1))
    m = re.search(r"(\d+)$", s)
    if m:
        return 1, int(m.group(1))
    return 2, s


def parse_seed_csv(seed_csv: str | None) -> list[str]:
    if not seed_csv:
        return []
    return [s.strip() for s in seed_csv.split(",") if s.strip()]


def short_model_name(model_id: str) -> str:
    return model_id.split("/", 1)[-1]


def load_forecastbench(filepath: str, metric: str = "overall") -> pd.DataFrame:
    """Load ForecastBench CSV and extract zero-shot model scores with CIs for one metric."""
    if metric not in FORECASTBENCH_METRICS:
        valid = ", ".join(sorted(FORECASTBENCH_METRICS.keys()))
        raise ValueError(f"Unknown ForecastBench metric '{metric}'. Valid options: {valid}")

    score_col, ci_col, _ = FORECASTBENCH_METRICS[metric]

    df = pd.read_csv(filepath)
    model_col = "Model"

    df = df[df[model_col].astype(str).str.contains(r"\(zero shot\)", case=False, regex=True)]

    results = []
    for _, row in df.iterrows():
        model_name = str(row[model_col]).strip()

        score_str = str(row[score_col])
        match = re.match(r"([\d.]+)\s*\(([\d,]+)\)", score_str)
        if not match:
            continue
        score = float(match.group(1))

        ci_str = str(row[ci_col])
        ci_match = re.match(r"\[([\d.]+),\s*([\d.]+)\]", ci_str)
        if ci_match:
            ci_lower = float(ci_match.group(1))
            ci_upper = float(ci_match.group(2))
        else:
            ci_lower = ci_upper = score

        base_name = re.sub(r"\s*\(zero shot\)\s*$", "", model_name, flags=re.IGNORECASE).strip()

        results.append(
            {
                "model_name": base_name,
                "score": score,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

    return pd.DataFrame(results)


def load_arcagi(filepath: str) -> pd.DataFrame:
    """Load ARC-AGI CSV (all variants, we'll match by exact name)."""
    df = pd.read_csv(filepath)

    results = []
    for _, row in df.iterrows():
        model_name = str(row["AI System"]).strip()

        score_str = str(row["ARC-AGI-1"])
        if score_str == "N/A" or pd.isna(score_str):
            continue

        match = re.match(r"([\d.]+)%", score_str)
        if not match:
            continue
        score = float(match.group(1)) / 100

        results.append(
            {
                "model_name": model_name,
                "score": score,
            }
        )

    return pd.DataFrame(results)


def load_civbench(filepath: str) -> pd.DataFrame:
    """Load CivBench results CSV. Accepts either brier_score or score columns."""
    df = pd.read_csv(filepath)

    if "brier_score" in df.columns:
        out = df.rename(columns={"model": "model_name", "brier_score": "score"})
    elif "score" in df.columns and "model_name" in df.columns:
        out = df.copy()
    elif "weighted_brier" in df.columns and "model" in df.columns:
        out = df.rename(columns={"model": "model_name", "weighted_brier": "score"})
    else:
        raise ValueError(
            f"Unsupported CivBench CSV schema in {filepath}. "
            "Expected columns including model+brier_score or model_name+score."
        )

    if "ci_lower" not in out.columns:
        out["ci_lower"] = out["score"]
    if "ci_upper" not in out.columns:
        out["ci_upper"] = out["score"]

    return out


def load_weight_map(weights_csv_path: str) -> dict[tuple[str, str], float]:
    df = pd.read_csv(weights_csv_path)
    required = {"horizon", "template_id", "weight"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Weights CSV {weights_csv_path} must have columns: horizon, template_id, weight"
        )

    weight_map: dict[tuple[str, str], float] = {}
    for _, row in df.iterrows():
        pair = (str(row["horizon"]), str(row["template_id"]))
        w = float(row["weight"])
        if w < 0:
            raise ValueError("Weights CSV contains negative weights, which are not allowed.")
        weight_map[pair] = w

    total = sum(weight_map.values())
    if total <= 0:
        raise ValueError("Weights CSV has non-positive total weight.")

    for k in list(weight_map.keys()):
        weight_map[k] /= total

    return weight_map


def extract_eval_brier_data(
    eval_json_path: str,
    include_h0: bool = False,
    seed_filter: set[str] | None = None,
) -> dict[str, object]:
    data = json.loads(Path(eval_json_path).read_text())
    questions = data.get("questions", [])

    models = data.get("metadata", {}).get("models", [])
    if models:
        models = sorted(str(m) for m in models)
    else:
        models = sorted(
            {
                str(model_id)
                for q in questions
                for model_id in q.get("predictions", {}).keys()
            }
        )

    sums_seed_model_pair: dict[tuple[str, str, tuple[str, str]], float] = defaultdict(float)
    cnts_seed_model_pair: dict[tuple[str, str, tuple[str, str]], int] = defaultdict(int)

    sums_model_pair: dict[tuple[str, tuple[str, str]], float] = defaultdict(float)
    cnts_model_pair: dict[tuple[str, tuple[str, str]], int] = defaultdict(int)

    values_model_pair: dict[tuple[str, tuple[str, str]], list[float]] = defaultdict(list)
    counts_seed_pair: dict[tuple[str, tuple[str, str]], int] = defaultdict(int)

    seeds: set[str] = set()
    pairs: set[tuple[str, str]] = set()

    for q in questions:
        template_id = str(q.get("template_id", ""))
        if not include_h0 and template_id.startswith("h0_"):
            continue

        gt_raw = q.get("ground_truth")
        if gt_raw is None:
            continue

        seed = str(q.get("game_id", "unknown"))
        if seed_filter is not None and seed not in seed_filter:
            continue

        horizon = str(q.get("horizon", "?"))
        pair = (horizon, template_id)

        seeds.add(seed)
        pairs.add(pair)
        counts_seed_pair[(seed, pair)] += 1

        gt = 1.0 if bool(gt_raw) else 0.0

        for model in models:
            pred = q.get("predictions", {}).get(model)
            if not pred:
                continue
            prob = pred.get("probability")
            if prob is None:
                continue

            brier = (float(prob) - gt) ** 2

            sums_seed_model_pair[(seed, model, pair)] += brier
            cnts_seed_model_pair[(seed, model, pair)] += 1

            sums_model_pair[(model, pair)] += brier
            cnts_model_pair[(model, pair)] += 1
            values_model_pair[(model, pair)].append(brier)

    seeds_sorted = sorted(seeds, key=seed_sort_key)
    pairs_sorted = sorted(pairs, key=lambda x: (x[0], x[1]))

    seed_idx = {s: i for i, s in enumerate(seeds_sorted)}
    model_idx = {m: i for i, m in enumerate(models)}
    pair_idx = {p: i for i, p in enumerate(pairs_sorted)}

    S, M, P = len(seeds_sorted), len(models), len(pairs_sorted)
    mean_seed_model_pair = np.full((S, M, P), np.nan, dtype=float)
    mean_model_pair = np.full((M, P), np.nan, dtype=float)
    counts_sp = np.zeros((S, P), dtype=float)

    for (seed, model, pair), total in sums_seed_model_pair.items():
        i = seed_idx[seed]
        j = model_idx[model]
        k = pair_idx[pair]
        mean_seed_model_pair[i, j, k] = total / cnts_seed_model_pair[(seed, model, pair)]

    for (model, pair), total in sums_model_pair.items():
        j = model_idx[model]
        k = pair_idx[pair]
        mean_model_pair[j, k] = total / cnts_model_pair[(model, pair)]

    for (seed, pair), n in counts_seed_pair.items():
        i = seed_idx[seed]
        k = pair_idx[pair]
        counts_sp[i, k] = n

    return {
        "models": models,
        "seeds": seeds_sorted,
        "pairs": pairs_sorted,
        "mean_seed_model_pair": mean_seed_model_pair,
        "mean_model_pair": mean_model_pair,
        "counts_seed_pair": counts_sp,
        "values_model_pair": {
            k: np.asarray(v, dtype=float) for k, v in values_model_pair.items()
        },
    }


def resolve_pair_weights(
    pairs: list[tuple[str, str]],
    weight_map: dict[tuple[str, str], float] | None,
) -> np.ndarray:
    if weight_map is None:
        return np.full(len(pairs), 1.0 / len(pairs), dtype=float)

    w = np.array([weight_map.get(pair, 0.0) for pair in pairs], dtype=float)
    if np.all(w <= 0):
        raise ValueError("No overlapping positive weights between weights CSV and eval pairs.")
    w /= w.sum()
    return w


def compute_weighted_civbench_from_eval(
    eval_json_path: str,
    output_csv_path: str,
    weights_csv_path: str | None,
    include_h0: bool = False,
    bootstrap_samples: int = 2000,
    seed_filter: set[str] | None = None,
) -> pd.DataFrame:
    eval_data = extract_eval_brier_data(eval_json_path, include_h0=include_h0, seed_filter=seed_filter)

    models: list[str] = eval_data["models"]
    pairs: list[tuple[str, str]] = eval_data["pairs"]
    mean_model_pair: np.ndarray = eval_data["mean_model_pair"]
    values_model_pair: dict[tuple[str, tuple[str, str]], np.ndarray] = eval_data["values_model_pair"]

    weight_map = load_weight_map(weights_csv_path) if weights_csv_path else None
    weights = resolve_pair_weights(pairs, weight_map)

    rng = np.random.default_rng(42)
    rows = []

    for m_idx, model in enumerate(models):
        model_pair_means = mean_model_pair[m_idx, :]
        available = np.isfinite(model_pair_means) & (weights > 0)

        if not np.any(available):
            continue

        w = weights[available]
        w /= w.sum()
        point = float(np.dot(w, model_pair_means[available]))

        if bootstrap_samples > 0:
            boot = np.zeros(bootstrap_samples, dtype=float)
            total_questions = 0
            avail_indices = np.where(available)[0]

            for local_idx, p_idx in enumerate(avail_indices):
                pair = pairs[p_idx]
                vals = values_model_pair.get((model, pair))
                if vals is None or len(vals) == 0:
                    continue
                total_questions += int(len(vals))
                draw_idx = rng.integers(0, len(vals), size=(bootstrap_samples, len(vals)))
                sampled_means = vals[draw_idx].mean(axis=1)
                boot += w[local_idx] * sampled_means

            ci_lower = float(np.percentile(boot, 2.5))
            ci_upper = float(np.percentile(boot, 97.5))
        else:
            total_questions = int(np.sum([len(values_model_pair.get((model, pairs[p_idx]), [])) for p_idx in np.where(available)[0]]))
            ci_lower = point
            ci_upper = point

        rows.append(
            {
                "model": model,
                "brier_score": point,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_questions": total_questions,
            }
        )

    df = pd.DataFrame(rows).sort_values("brier_score")
    df.to_csv(output_csv_path, index=False)
    return df


def create_seed_rank_heatmap(
    eval_json_path: str,
    output_path: str,
    title: str,
    weights_csv_path: str | None = None,
    include_h0: bool = False,
    seed_filter: set[str] | None = None,
    train_seeds: list[str] | None = None,
    test_seeds: list[str] | None = None,
) -> Path:
    eval_data = extract_eval_brier_data(eval_json_path, include_h0=include_h0, seed_filter=seed_filter)

    models: list[str] = eval_data["models"]
    seeds: list[str] = eval_data["seeds"]
    pairs: list[tuple[str, str]] = eval_data["pairs"]
    mean_seed_model_pair: np.ndarray = eval_data["mean_seed_model_pair"]
    counts_sp: np.ndarray = eval_data["counts_seed_pair"]

    weight_map = load_weight_map(weights_csv_path) if weights_csv_path else None

    S, M, _ = mean_seed_model_pair.shape
    seed_scores = np.full((S, M), np.nan, dtype=float)

    if weight_map is None:
        # Unweighted per-seed score: question-count-weighted average over available pairs.
        for s_idx in range(S):
            pair_counts = counts_sp[s_idx, :]
            for m_idx in range(M):
                means = mean_seed_model_pair[s_idx, m_idx, :]
                avail = np.isfinite(means) & (pair_counts > 0)
                if not np.any(avail):
                    continue
                w_local = pair_counts[avail].astype(float)
                w_local /= w_local.sum()
                seed_scores[s_idx, m_idx] = float(np.dot(w_local, means[avail]))
    else:
        base_w = resolve_pair_weights(pairs, weight_map)
        for s_idx in range(S):
            pair_counts = counts_sp[s_idx, :]
            for m_idx in range(M):
                means = mean_seed_model_pair[s_idx, m_idx, :]
                avail = np.isfinite(means) & (pair_counts > 0) & (base_w > 0)
                if not np.any(avail):
                    continue
                w_local = base_w[avail].astype(float)
                w_local /= w_local.sum()
                seed_scores[s_idx, m_idx] = float(np.dot(w_local, means[avail]))

    if np.isnan(seed_scores).any():
        # Fallback to keep plot renderable; these should be rare after availability-normalization.
        finite_vals = seed_scores[np.isfinite(seed_scores)]
        fill = float(np.max(finite_vals) + 1.0) if finite_vals.size else float(M)
        seed_scores = np.where(np.isfinite(seed_scores), seed_scores, fill)

    rank_matrix = np.zeros_like(seed_scores)
    for s_idx in range(seed_scores.shape[0]):
        rank_matrix[s_idx] = stats.rankdata(seed_scores[s_idx], method="average")

    rank_matrix = rank_matrix.T  # models x seeds

    # Reorder seed columns to show train seeds first and test seeds second when provided.
    display_train = train_seeds or []
    display_test = test_seeds or []
    seed_to_idx = {s: i for i, s in enumerate(seeds)}

    ordered_seed_names: list[str] = []
    seen: set[str] = set()
    for s in display_train + display_test:
        if s in seed_to_idx and s not in seen:
            ordered_seed_names.append(s)
            seen.add(s)
    for s in seeds:
        if s not in seen:
            ordered_seed_names.append(s)
            seen.add(s)

    ordered_seed_idx = [seed_to_idx[s] for s in ordered_seed_names]
    rank_matrix = rank_matrix[:, ordered_seed_idx]
    seeds = ordered_seed_names
    test_seed_set = set(display_test)
    train_count = sum(1 for s in seeds if s in set(display_train))

    # Sort models top-to-bottom by best (lowest) average rank across displayed seeds.
    mean_rank = rank_matrix.mean(axis=1)
    order = np.argsort(mean_rank)
    rank_sorted = rank_matrix[order, :]
    model_labels = [short_model_name(models[i]) for i in order]

    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    im = ax.imshow(rank_sorted, aspect="auto", cmap="viridis", vmin=1, vmax=len(models))
    ax.set_xticks(np.arange(len(seeds)))
    xticklabels = [f"{s}*" if s in test_seed_set else s for s in seeds]
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(model_labels)))
    ax.set_yticklabels(model_labels)
    ax.set_title(title)

    if display_train and display_test and 0 < train_count < len(seeds):
        ax.axvline(train_count - 0.5, color="white", linestyle="--", linewidth=1.5, alpha=0.9)

    for tick_label, seed_name in zip(ax.get_xticklabels(), seeds):
        if seed_name in test_seed_set:
            tick_label.set_color("crimson")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rank (1 = best)")

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def match_all_datasets(civbench_df: pd.DataFrame,
                       forecastbench_df: pd.DataFrame,
                       arcagi_df: pd.DataFrame) -> dict:
    """Match models across all three datasets."""
    fb_lookup = {row["model_name"].lower(): row.to_dict()
                 for _, row in forecastbench_df.iterrows()}
    arc_lookup = {row["model_name"].lower(): row["score"]
                  for _, row in arcagi_df.iterrows()}

    cb_fb_matches = []
    cb_arc_matches = []
    all_three_matches = []

    for _, cb_row in civbench_df.iterrows():
        civbench_id = cb_row["model_name"]
        civbench_score = cb_row["score"]
        civbench_ci_lower = cb_row["ci_lower"]
        civbench_ci_upper = cb_row["ci_upper"]

        fb_name, arc_name = MODEL_MAPPING.get(civbench_id, (None, None))

        fb_data = None
        arc_score = None

        if fb_name and fb_name.lower() in fb_lookup:
            fb_data = fb_lookup[fb_name.lower()]
            cb_fb_matches.append(
                {
                    "model": civbench_id,
                    "civbench_score": civbench_score,
                    "civbench_ci_lower": civbench_ci_lower,
                    "civbench_ci_upper": civbench_ci_upper,
                    "forecastbench_score": fb_data["score"],
                    "forecastbench_ci_lower": fb_data["ci_lower"],
                    "forecastbench_ci_upper": fb_data["ci_upper"],
                }
            )

        if arc_name and arc_name.lower() in arc_lookup:
            arc_score = arc_lookup[arc_name.lower()]
            cb_arc_matches.append(
                {
                    "model": civbench_id,
                    "civbench_score": civbench_score,
                    "civbench_ci_lower": civbench_ci_lower,
                    "civbench_ci_upper": civbench_ci_upper,
                    "arcagi_score": arc_score,
                }
            )

        if fb_data is not None and arc_score is not None:
            all_three_matches.append(
                {
                    "model": civbench_id,
                    "civbench_score": civbench_score,
                    "civbench_ci_lower": civbench_ci_lower,
                    "civbench_ci_upper": civbench_ci_upper,
                    "forecastbench_score": fb_data["score"],
                    "forecastbench_ci_lower": fb_data["ci_lower"],
                    "forecastbench_ci_upper": fb_data["ci_upper"],
                    "arcagi_score": arc_score,
                }
            )

    return {
        "civbench_forecastbench": pd.DataFrame(cb_fb_matches),
        "civbench_arcagi": pd.DataFrame(cb_arc_matches),
        "all_three": pd.DataFrame(all_three_matches),
    }


def ci_to_std(ci_lower: float, ci_upper: float) -> float:
    """Convert 95% CI to standard deviation."""
    return (ci_upper - ci_lower) / (2 * 1.96)


def sample_from_ci(score: float, ci_lower: float, ci_upper: float,
                   rng: np.random.Generator) -> float:
    """Sample from CI distribution (assuming normal)."""
    std = ci_to_std(ci_lower, ci_upper)
    if std <= 0:
        return score
    return rng.normal(score, std)


def bootstrap_rank_correlation(x: np.ndarray, y: np.ndarray,
                               x_ci_lower: np.ndarray | None = None,
                               x_ci_upper: np.ndarray | None = None,
                               y_ci_lower: np.ndarray | None = None,
                               y_ci_upper: np.ndarray | None = None,
                               n_bootstrap: int = 10000,
                               x_higher_is_better: bool = False,
                               y_higher_is_better: bool = False) -> dict:
    """
    Bootstrap rank correlation accounting for CIs.

    x: predictor scores
    y: target scores (CivBench)
    If CIs are provided, samples from them; otherwise uses point estimates.
    """
    n = len(x)
    rng = np.random.default_rng(42)

    has_x_ci = x_ci_lower is not None and x_ci_upper is not None
    has_y_ci = y_ci_lower is not None and y_ci_upper is not None

    bootstrap_rhos = []

    for _ in range(n_bootstrap):
        if has_x_ci:
            x_sample = np.array([sample_from_ci(x[i], x_ci_lower[i], x_ci_upper[i], rng) for i in range(n)])
        else:
            x_sample = x

        if has_y_ci:
            y_sample = np.array([sample_from_ci(y[i], y_ci_lower[i], y_ci_upper[i], rng) for i in range(n)])
        else:
            y_sample = y

        if x_higher_is_better:
            x_ranks = stats.rankdata(-x_sample)
        else:
            x_ranks = stats.rankdata(x_sample)

        if y_higher_is_better:
            y_ranks = stats.rankdata(-y_sample)
        else:
            y_ranks = stats.rankdata(y_sample)

        rho, _ = stats.spearmanr(x_ranks, y_ranks)
        bootstrap_rhos.append(rho)

    bootstrap_rhos = np.array(bootstrap_rhos)

    if x_higher_is_better:
        x_ranks_point = stats.rankdata(-x)
    else:
        x_ranks_point = stats.rankdata(x)

    if y_higher_is_better:
        y_ranks_point = stats.rankdata(-y)
    else:
        y_ranks_point = stats.rankdata(y)

    rho_point, p_value = stats.spearmanr(x_ranks_point, y_ranks_point)

    return {
        "point_estimate": rho_point,
        "p_value": p_value,
        "bootstrap_mean": np.mean(bootstrap_rhos),
        "ci_lower": np.percentile(bootstrap_rhos, 2.5),
        "ci_upper": np.percentile(bootstrap_rhos, 97.5),
        "prob_positive": np.mean(bootstrap_rhos > 0),
        "n": n,
    }


def create_rank_plots(
    matched_data: dict,
    output_dir: str,
    forecastbench_label: str = "ForecastBench",
    civbench_label: str = "CivBench",
    output_stem: str = "rank_comparison",
):
    """Create scatter plots of ranks."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cb_fb = matched_data["civbench_forecastbench"]
    if len(cb_fb) >= 3:
        ax = axes[0]
        cb_ranks = stats.rankdata(cb_fb["civbench_score"].values)
        fb_ranks = stats.rankdata(cb_fb["forecastbench_score"].values)
        rho_fb, _ = stats.spearmanr(fb_ranks, cb_ranks)

        ax.scatter(fb_ranks, cb_ranks)
        ax.plot([0, len(cb_fb) + 1], [0, len(cb_fb) + 1], "r--", alpha=0.5, label="Perfect agreement")

        ax.set_xlabel(f"{forecastbench_label} Rank")
        ax.set_ylabel(f"{civbench_label} Rank")
        ax.set_title(f"{forecastbench_label} vs {civbench_label} (n={len(cb_fb)}, rho={rho_fb:.3f})")

        for i, row in cb_fb.iterrows():
            ax.annotate(short_model_name(row["model"])[:16], (fb_ranks[i], cb_ranks[i]), fontsize=6, alpha=0.7)
        ax.legend()

    cb_arc = matched_data["civbench_arcagi"]
    if len(cb_arc) >= 3:
        ax = axes[1]
        cb_ranks = stats.rankdata(cb_arc["civbench_score"].values)
        arc_ranks = stats.rankdata(-cb_arc["arcagi_score"].values)
        rho_arc, _ = stats.spearmanr(arc_ranks, cb_ranks)

        ax.scatter(arc_ranks, cb_ranks)
        ax.plot([0, len(cb_arc) + 1], [0, len(cb_arc) + 1], "r--", alpha=0.5, label="Perfect agreement")

        ax.set_xlabel("ARC-AGI Rank (1 = highest score)")
        ax.set_ylabel(f"{civbench_label} Rank (1 = lowest Brier)")
        ax.set_title(f"ARC-AGI vs {civbench_label} (n={len(cb_arc)}, rho={rho_arc:.3f})")

        for i, row in cb_arc.iterrows():
            ax.annotate(short_model_name(row["model"])[:16], (arc_ranks[i], cb_ranks[i]), fontsize=6, alpha=0.7)
        ax.legend()
    else:
        axes[1].text(0.5, 0.5, f"Only {len(cb_arc)} models matched",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("ARC-AGI vs CivBench Rankings")

    plt.tight_layout()
    output_path = Path(output_dir) / f"{output_stem}.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nRank plots saved to {output_path}")
    plt.close()


def _plot_rank_scatter(
    ax: plt.Axes,
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    x_label: str,
    y_label: str,
    title_prefix: str,
    model_labels: list[str],
    x_higher_is_better: bool = False,
    y_higher_is_better: bool = False,
) -> None:
    if x_higher_is_better:
        x_ranks = stats.rankdata(-x_scores)
    else:
        x_ranks = stats.rankdata(x_scores)

    if y_higher_is_better:
        y_ranks = stats.rankdata(-y_scores)
    else:
        y_ranks = stats.rankdata(y_scores)

    rho, _ = stats.spearmanr(x_ranks, y_ranks)
    n = len(x_scores)

    ax.scatter(x_ranks, y_ranks)
    ax.plot([0, n + 1], [0, n + 1], "r--", alpha=0.5, label="Perfect agreement")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_prefix} (n={n}, rho={rho:.3f})")

    for i, label in enumerate(model_labels):
        ax.annotate(short_model_name(label)[:16], (x_ranks[i], y_ranks[i]), fontsize=6, alpha=0.7)
    ax.legend()


def create_split_agreement_plot(
    metric_label: str,
    matched_train: dict,
    matched_test: dict,
    output_dir: str,
    civbench_label: str,
    output_stem: str,
    train_seeds: list[str] | None = None,
    test_seeds: list[str] | None = None,
) -> None:
    """
    Create a 3-panel split-aware agreement figure:
    1) ForecastBench vs CivBench (train seeds)
    2) ForecastBench vs CivBench (test seeds)
    3) ARC-AGI vs CivBench (test seeds)
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    cb_fb_train = matched_train["civbench_forecastbench"]
    if len(cb_fb_train) >= 3:
        _plot_rank_scatter(
            ax=axes[0],
            x_scores=cb_fb_train["forecastbench_score"].values,
            y_scores=cb_fb_train["civbench_score"].values,
            x_label=f"ForecastBench [{metric_label}] Rank",
            y_label=f"{civbench_label} Train Rank",
            title_prefix="Train Agreement",
            model_labels=cb_fb_train["model"].tolist(),
            x_higher_is_better=False,
            y_higher_is_better=False,
        )
    else:
        axes[0].text(0.5, 0.5, f"Only {len(cb_fb_train)} models", ha="center", va="center",
                     transform=axes[0].transAxes)
        axes[0].set_title("Train Agreement")

    cb_fb_test = matched_test["civbench_forecastbench"]
    if len(cb_fb_test) >= 3:
        _plot_rank_scatter(
            ax=axes[1],
            x_scores=cb_fb_test["forecastbench_score"].values,
            y_scores=cb_fb_test["civbench_score"].values,
            x_label=f"ForecastBench [{metric_label}] Rank",
            y_label=f"{civbench_label} Test Rank",
            title_prefix="Test Agreement",
            model_labels=cb_fb_test["model"].tolist(),
            x_higher_is_better=False,
            y_higher_is_better=False,
        )
    else:
        axes[1].text(0.5, 0.5, f"Only {len(cb_fb_test)} models", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_title("Test Agreement")

    cb_arc_test = matched_test["civbench_arcagi"]
    if len(cb_arc_test) >= 3:
        _plot_rank_scatter(
            ax=axes[2],
            x_scores=cb_arc_test["arcagi_score"].values,
            y_scores=cb_arc_test["civbench_score"].values,
            x_label="ARC-AGI Rank (1 = highest score)",
            y_label=f"{civbench_label} Test Rank",
            title_prefix="ARC-AGI vs CivBench (Test)",
            model_labels=cb_arc_test["model"].tolist(),
            x_higher_is_better=True,
            y_higher_is_better=False,
        )
    else:
        axes[2].text(0.5, 0.5, f"Only {len(cb_arc_test)} models", ha="center", va="center",
                     transform=axes[2].transAxes)
        axes[2].set_title("ARC-AGI vs CivBench (Test)")

    if train_seeds or test_seeds:
        fig.suptitle(
            f"Train seeds: {','.join(train_seeds or [])} | Test seeds: {','.join(test_seeds or [])}",
            fontsize=10,
            y=1.02,
        )
    plt.tight_layout()
    output_path = Path(output_dir) / f"{output_stem}.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSplit agreement plot saved to {output_path}")
    plt.close()


def predict_civbench(
    forecastbench_path: str,
    arcagi_path: str,
    civbench_path: str,
    output_dir: str | None = None,
    n_bootstrap: int = 10000,
    forecastbench_metrics: list[str] | None = None,
    eval_json_path: str | None = None,
    weights_csv_path: str | None = None,
    weighted_output_csv: str | None = None,
    weighted_bootstrap_samples: int = 2000,
    include_h0: bool = False,
    seed_filter: set[str] | None = None,
    train_seeds: list[str] | None = None,
    test_seeds: list[str] | None = None,
):
    """Main function to compare predictors of CivBench rankings."""
    if forecastbench_metrics is None:
        forecastbench_metrics = ["overall"]

    invalid = [m for m in forecastbench_metrics if m not in FORECASTBENCH_METRICS]
    if invalid:
        valid = ", ".join(sorted(FORECASTBENCH_METRICS.keys()))
        raise ValueError(f"Invalid ForecastBench metrics: {invalid}. Valid options: {valid}")

    if weights_csv_path and not eval_json_path:
        raise ValueError("--weights-csv requires --eval-json so weighted CivBench can be computed.")

    print("=" * 70)
    print("COMPARING PREDICTORS OF CIVBENCH RANKINGS")
    print("(Using best variant per model family for ARC-AGI)")
    print("=" * 70)

    cb_label = "CivBench"
    cb_source_path = civbench_path
    cb_train_path: str | None = None
    cb_test_path: str | None = None
    cb_train_unweighted_path: str | None = None
    cb_test_unweighted_path: str | None = None

    if weights_csv_path:
        if weighted_output_csv is None:
            weighted_output_csv = str(Path(output_dir or "data") / "civbench_results_weighted.csv")

        print("\nComputing weighted CivBench Brier scores from eval JSON...")
        weighted_df = compute_weighted_civbench_from_eval(
            eval_json_path=eval_json_path,
            output_csv_path=weighted_output_csv,
            weights_csv_path=weights_csv_path,
            include_h0=include_h0,
            bootstrap_samples=weighted_bootstrap_samples,
            seed_filter=seed_filter,
        )
        print(f"  Weighted CivBench CSV saved: {weighted_output_csv} ({len(weighted_df)} models)")
        cb_source_path = weighted_output_csv
        cb_label = "CivBench (Weighted)"

        if train_seeds and test_seeds:
            weighted_base = Path(weighted_output_csv)
            train_csv = weighted_base.with_name(f"{weighted_base.stem}_train{weighted_base.suffix}")
            test_csv = weighted_base.with_name(f"{weighted_base.stem}_test{weighted_base.suffix}")

            weighted_train_df = compute_weighted_civbench_from_eval(
                eval_json_path=eval_json_path,
                output_csv_path=str(train_csv),
                weights_csv_path=weights_csv_path,
                include_h0=include_h0,
                bootstrap_samples=weighted_bootstrap_samples,
                seed_filter=set(train_seeds),
            )
            weighted_test_df = compute_weighted_civbench_from_eval(
                eval_json_path=eval_json_path,
                output_csv_path=str(test_csv),
                weights_csv_path=weights_csv_path,
                include_h0=include_h0,
                bootstrap_samples=weighted_bootstrap_samples,
                seed_filter=set(test_seeds),
            )
            cb_train_path = str(train_csv)
            cb_test_path = str(test_csv)
            print(f"  Weighted TRAIN CivBench CSV saved: {train_csv} ({len(weighted_train_df)} models)")
            print(f"  Weighted TEST CivBench CSV saved: {test_csv} ({len(weighted_test_df)} models)")

            # Also produce unweighted split CSVs for baseline split-agreement plots.
            unweighted_base = Path(output_dir or "data") / "civbench_results_unweighted_from_eval.csv"
            train_unweighted_csv = unweighted_base.with_name(
                f"{unweighted_base.stem}_train{unweighted_base.suffix}"
            )
            test_unweighted_csv = unweighted_base.with_name(
                f"{unweighted_base.stem}_test{unweighted_base.suffix}"
            )

            unweighted_train_df = compute_weighted_civbench_from_eval(
                eval_json_path=eval_json_path,
                output_csv_path=str(train_unweighted_csv),
                weights_csv_path=None,
                include_h0=include_h0,
                bootstrap_samples=weighted_bootstrap_samples,
                seed_filter=set(train_seeds),
            )
            unweighted_test_df = compute_weighted_civbench_from_eval(
                eval_json_path=eval_json_path,
                output_csv_path=str(test_unweighted_csv),
                weights_csv_path=None,
                include_h0=include_h0,
                bootstrap_samples=weighted_bootstrap_samples,
                seed_filter=set(test_seeds),
            )
            cb_train_unweighted_path = str(train_unweighted_csv)
            cb_test_unweighted_path = str(test_unweighted_csv)
            print(
                f"  Unweighted TRAIN CivBench CSV saved: {train_unweighted_csv} "
                f"({len(unweighted_train_df)} models)"
            )
            print(
                f"  Unweighted TEST CivBench CSV saved: {test_unweighted_csv} "
                f"({len(unweighted_test_df)} models)"
            )
    elif eval_json_path and train_seeds and test_seeds:
        # Build split-specific unweighted CivBench from eval JSON.
        unweighted_base = Path(output_dir or "data") / "civbench_results_unweighted_from_eval.csv"
        train_csv = unweighted_base.with_name(f"{unweighted_base.stem}_train{unweighted_base.suffix}")
        test_csv = unweighted_base.with_name(f"{unweighted_base.stem}_test{unweighted_base.suffix}")

        train_df = compute_weighted_civbench_from_eval(
            eval_json_path=eval_json_path,
            output_csv_path=str(train_csv),
            weights_csv_path=None,
            include_h0=include_h0,
            bootstrap_samples=weighted_bootstrap_samples,
            seed_filter=set(train_seeds),
        )
        test_df = compute_weighted_civbench_from_eval(
            eval_json_path=eval_json_path,
            output_csv_path=str(test_csv),
            weights_csv_path=None,
            include_h0=include_h0,
            bootstrap_samples=weighted_bootstrap_samples,
            seed_filter=set(test_seeds),
        )
        cb_train_path = str(train_csv)
        cb_test_path = str(test_csv)
        print(f"  Unweighted TRAIN CivBench CSV saved: {train_csv} ({len(train_df)} models)")
        print(f"  Unweighted TEST CivBench CSV saved: {test_csv} ({len(test_df)} models)")

    print("\nLoading datasets...")
    fb_by_metric = {}
    for metric in forecastbench_metrics:
        fb_df = load_forecastbench(forecastbench_path, metric)
        fb_by_metric[metric] = fb_df
        _, _, metric_label = FORECASTBENCH_METRICS[metric]
        print(f"  ForecastBench [{metric_label}]: {len(fb_df)} zero-shot models")

    arc_df = load_arcagi(arcagi_path)
    print(f"  ARC-AGI: {len(arc_df)} models (all variants)")

    cb_df = load_civbench(cb_source_path)
    print(f"  {cb_label}: {len(cb_df)} models")

    if output_dir and eval_json_path:
        try:
            heatmap_seed_filter = seed_filter
            if train_seeds or test_seeds:
                split_union = set((train_seeds or []) + (test_seeds or []))
                heatmap_seed_filter = split_union if heatmap_seed_filter is None else (heatmap_seed_filter | split_union)
                print(f"  Heatmap train seeds: {train_seeds or []}")
                print(f"  Heatmap test seeds: {test_seeds or []}")

            split_suffix = "_split" if (train_seeds or test_seeds) else ""
            split_title = ""
            if train_seeds or test_seeds:
                split_title = (
                    f" | train={','.join(train_seeds or [])} "
                    f"| test={','.join(test_seeds or [])}"
                )

            unweighted_path = create_seed_rank_heatmap(
                eval_json_path=eval_json_path,
                output_path=str(Path(output_dir) / f"seed_rank_heatmap_unweighted{split_suffix}.png"),
                title=f"Per-seed rank heatmap (unweighted; lower rank = better){split_title}",
                weights_csv_path=None,
                include_h0=include_h0,
                seed_filter=heatmap_seed_filter,
                train_seeds=train_seeds,
                test_seeds=test_seeds,
            )
            print(f"  Seed heatmap saved: {unweighted_path}")

            if weights_csv_path:
                weighted_path = create_seed_rank_heatmap(
                    eval_json_path=eval_json_path,
                    output_path=str(Path(output_dir) / f"seed_rank_heatmap_weighted{split_suffix}.png"),
                    title=f"Per-seed rank heatmap (weighted; lower rank = better){split_title}",
                    weights_csv_path=weights_csv_path,
                    include_h0=include_h0,
                    seed_filter=heatmap_seed_filter,
                    train_seeds=train_seeds,
                    test_seeds=test_seeds,
                )
                print(f"  Seed heatmap saved: {weighted_path}")
        except Exception as exc:
            print(f"  Warning: failed to generate seed heatmaps: {exc}")

    print("\nMatching models across datasets...")
    matched_by_metric = {}
    matched_train_by_metric = {}
    matched_test_by_metric = {}
    matched_train_unweighted_by_metric = {}
    matched_test_unweighted_by_metric = {}

    cb_df_train = load_civbench(cb_train_path) if cb_train_path else None
    cb_df_test = load_civbench(cb_test_path) if cb_test_path else None
    cb_df_train_unweighted = load_civbench(cb_train_unweighted_path) if cb_train_unweighted_path else None
    cb_df_test_unweighted = load_civbench(cb_test_unweighted_path) if cb_test_unweighted_path else None

    for metric in forecastbench_metrics:
        matched = match_all_datasets(cb_df, fb_by_metric[metric], arc_df)
        matched_by_metric[metric] = matched
        _, _, metric_label = FORECASTBENCH_METRICS[metric]
        cb_fb = matched["civbench_forecastbench"]
        cb_arc = matched["civbench_arcagi"]
        all_three = matched["all_three"]
        print(f"  CivBench-ForecastBench matches [{metric_label}]: {len(cb_fb)}")
        print(f"  CivBench-ARC-AGI matches [{metric_label}]: {len(cb_arc)}")
        print(f"  All three datasets [{metric_label}]: {len(all_three)}")

        if cb_df_train is not None:
            matched_train = match_all_datasets(cb_df_train, fb_by_metric[metric], arc_df)
            matched_train_by_metric[metric] = matched_train
            print(
                f"    Train split matches [{metric_label}]: "
                f"FB={len(matched_train['civbench_forecastbench'])}, ARC={len(matched_train['civbench_arcagi'])}"
            )
        if cb_df_test is not None:
            matched_test = match_all_datasets(cb_df_test, fb_by_metric[metric], arc_df)
            matched_test_by_metric[metric] = matched_test
            print(
                f"    Test split matches [{metric_label}]: "
                f"FB={len(matched_test['civbench_forecastbench'])}, ARC={len(matched_test['civbench_arcagi'])}"
            )
        if cb_df_train_unweighted is not None:
            matched_train_unweighted_by_metric[metric] = match_all_datasets(
                cb_df_train_unweighted, fb_by_metric[metric], arc_df
            )
        if cb_df_test_unweighted is not None:
            matched_test_unweighted_by_metric[metric] = match_all_datasets(
                cb_df_test_unweighted, fb_by_metric[metric], arc_df
            )

    primary_metric = forecastbench_metrics[0]
    matched_primary = matched_by_metric[primary_metric]
    cb_arc = matched_primary["civbench_arcagi"]

    print("\n" + "-" * 70)
    print("MATCHED MODELS")
    print("-" * 70)

    if len(cb_arc) > 0:
        print("\nCivBench <-> ARC-AGI matches:")
        for _, row in cb_arc.sort_values("civbench_score").iterrows():
            arc_pct = row["arcagi_score"] * 100
            print(f"  {row['model']}: CB={row['civbench_score']:.3f}, ARC={arc_pct:.1f}%")

    forecastbench_results = {}
    for metric in forecastbench_metrics:
        _, _, metric_label = FORECASTBENCH_METRICS[metric]
        cb_fb = matched_by_metric[metric]["civbench_forecastbench"]

        print("\n" + "-" * 70)
        print(f"FORECASTBENCH [{metric_label}] RANK CORRELATION WITH {cb_label.upper()}")
        print("-" * 70)

        if len(cb_fb) >= 3:
            fb_result = bootstrap_rank_correlation(
                x=cb_fb["forecastbench_score"].values,
                y=cb_fb["civbench_score"].values,
                x_ci_lower=cb_fb["forecastbench_ci_lower"].values,
                x_ci_upper=cb_fb["forecastbench_ci_upper"].values,
                y_ci_lower=cb_fb["civbench_ci_lower"].values,
                y_ci_upper=cb_fb["civbench_ci_upper"].values,
                n_bootstrap=n_bootstrap,
                x_higher_is_better=False,
                y_higher_is_better=False,
            )

            print(f"\nSpearman's rho (rank correlation): {fb_result['point_estimate']:.4f}")
            print(f"  Bootstrap mean: {fb_result['bootstrap_mean']:.4f}")
            print(f"  95% CI: [{fb_result['ci_lower']:.4f}, {fb_result['ci_upper']:.4f}]")
            print(f"  P(rho > 0): {fb_result['prob_positive']:.1%}")
            print(f"  n_models: {fb_result['n']}")
        else:
            print("\nInsufficient data (need >= 3 matched models)")
            fb_result = None

        forecastbench_results[metric] = fb_result

    print("\n" + "-" * 70)
    print(f"ARC-AGI RANK CORRELATION WITH {cb_label.upper()}")
    print("-" * 70)

    if len(cb_arc) >= 3:
        arc_result = bootstrap_rank_correlation(
            x=cb_arc["arcagi_score"].values,
            y=cb_arc["civbench_score"].values,
            y_ci_lower=cb_arc["civbench_ci_lower"].values,
            y_ci_upper=cb_arc["civbench_ci_upper"].values,
            n_bootstrap=n_bootstrap,
            x_higher_is_better=True,
            y_higher_is_better=False,
        )

        print(f"\nSpearman's rho (rank correlation): {arc_result['point_estimate']:.4f}")
        print(f"  Bootstrap mean: {arc_result['bootstrap_mean']:.4f}")
        print(f"  95% CI: [{arc_result['ci_lower']:.4f}, {arc_result['ci_upper']:.4f}]")
        print(f"  P(rho > 0): {arc_result['prob_positive']:.1%}")
        print(f"  n_models: {arc_result['n']}")
    else:
        print(f"\nInsufficient data: only {len(cb_arc)} matched models (need >= 3)")
        arc_result = None

    if output_dir:
        for metric in forecastbench_metrics:
            _, _, metric_label = FORECASTBENCH_METRICS[metric]
            stem_suffix = "weighted" if weights_csv_path else "unweighted"

            if metric in matched_train_by_metric and metric in matched_test_by_metric:
                create_split_agreement_plot(
                    metric_label=metric_label,
                    matched_train=matched_train_by_metric[metric],
                    matched_test=matched_test_by_metric[metric],
                    output_dir=output_dir,
                    civbench_label=cb_label,
                    output_stem=f"rank_agreement_split_{metric}_{stem_suffix}",
                    train_seeds=train_seeds,
                    test_seeds=test_seeds,
                )

            if (
                metric in matched_train_unweighted_by_metric
                and metric in matched_test_unweighted_by_metric
                and (weights_csv_path is not None)
            ):
                create_split_agreement_plot(
                    metric_label=metric_label,
                    matched_train=matched_train_unweighted_by_metric[metric],
                    matched_test=matched_test_unweighted_by_metric[metric],
                    output_dir=output_dir,
                    civbench_label="CivBench (Unweighted)",
                    output_stem=f"rank_agreement_split_{metric}_unweighted",
                    train_seeds=train_seeds,
                    test_seeds=test_seeds,
                )

    primary_result = forecastbench_results.get("overall")
    if primary_result is None and forecastbench_metrics:
        primary_result = forecastbench_results.get(forecastbench_metrics[0])

    return {
        "civbench_source_path": cb_source_path,
        "civbench_train_path": cb_train_path,
        "civbench_test_path": cb_test_path,
        "matched_by_metric": matched_by_metric,
        "matched_train_by_metric": matched_train_by_metric,
        "matched_test_by_metric": matched_test_by_metric,
        "forecastbench_results": forecastbench_results,
        "forecastbench_result": primary_result,
        "arcagi_result": arc_result,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare ARC-AGI vs ForecastBench as CivBench predictors")
    parser.add_argument("--forecastbench", default="data/ForecastBench.csv")
    parser.add_argument("--arcagi", default="data/ARC-AGI.csv")
    parser.add_argument("--civbench", default="data/civbench_results.csv")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument(
        "--forecastbench-metrics",
        default="overall,market,dataset",
        help="Comma-separated ForecastBench metrics: overall,market,dataset",
    )

    parser.add_argument(
        "--eval-json",
        default="",
        help="Optional eval JSON for seed heatmaps and/or weighted CivBench scoring.",
    )
    parser.add_argument(
        "--weights-csv",
        default="",
        help="Optional learned weights CSV (horizon,template,weight) to compute weighted CivBench scores.",
    )
    parser.add_argument(
        "--weighted-output-csv",
        default="",
        help="Output CSV path for weighted CivBench scores (used when --weights-csv is set).",
    )
    parser.add_argument(
        "--weighted-bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples for weighted CivBench CI estimation.",
    )
    parser.add_argument(
        "--include-h0",
        action="store_true",
        help="Include H0 templates when computing weighted CivBench/heatmaps.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated seed filter for eval-json based computations.",
    )
    parser.add_argument(
        "--train-seeds",
        default="",
        help="Optional comma-separated train seeds (e.g., seed1,...,seed8) for split agreement plots.",
    )
    parser.add_argument(
        "--test-seeds",
        default="",
        help="Optional comma-separated test seeds (e.g., seed9,seed10) for split agreement plots.",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    def resolve(p: str) -> Path:
        return project_root / p if p and not Path(p).is_absolute() else Path(p)

    metrics = [m.strip().lower() for m in args.forecastbench_metrics.split(",") if m.strip()]

    eval_json = str(resolve(args.eval_json)) if args.eval_json else None
    weights_csv = str(resolve(args.weights_csv)) if args.weights_csv else None
    weighted_output_csv = str(resolve(args.weighted_output_csv)) if args.weighted_output_csv else None
    seed_filter_vals = parse_seed_csv(args.seeds)
    seed_filter = set(seed_filter_vals) if seed_filter_vals else None
    train_seeds = parse_seed_csv(args.train_seeds)
    test_seeds = parse_seed_csv(args.test_seeds)

    predict_civbench(
        str(resolve(args.forecastbench)),
        str(resolve(args.arcagi)),
        str(resolve(args.civbench)),
        str(resolve(args.output_dir)),
        args.n_bootstrap,
        metrics,
        eval_json_path=eval_json,
        weights_csv_path=weights_csv,
        weighted_output_csv=weighted_output_csv,
        weighted_bootstrap_samples=args.weighted_bootstrap_samples,
        include_h0=args.include_h0,
        seed_filter=seed_filter,
        train_seeds=train_seeds if train_seeds else None,
        test_seeds=test_seeds if test_seeds else None,
    )
