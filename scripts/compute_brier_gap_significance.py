"""
Compute bootstrap 95% CIs and permutation-test p-values for the
conditional Brier gap (conditional - baseline) for each model,
across Republic and Gold500 interventions.
"""

import json
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

INTERVENTIONS = {
    "Republic": {
        "baseline": DATA_DIR / "republic_baseline_binary_all.json",
        "conditional": DATA_DIR / "republic_conditional_binary_all.json",
    },
    "Gold500": {
        "baseline": DATA_DIR / "gold500_baseline_binary_all.json",
        "conditional": DATA_DIR / "gold500_conditional_binary_all.json",
    },
}

N_BOOTSTRAP = 10_000
N_PERMUTATIONS = 10_000
SEED = 42
CI_LOWER = 2.5
CI_UPPER = 97.5

# Short display names for models
SHORT_NAMES = {
    "anthropic/claude-opus-4-5-20251101": "Claude Opus 4.5",
    "anthropic/claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "openai/gpt-5-2025-08-07": "GPT-5",
    "openai/gpt-5-mini-2025-08-07": "GPT-5 Mini",
    "openai/gpt-5.1-2025-11-13": "GPT-5.1",
    "openai/o3-2025-04-16": "o3",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_brier_scores(filepath: Path, model_id: str) -> np.ndarray:
    """Return per-question Brier scores for a model from one results file."""
    with open(filepath) as f:
        data = json.load(f)
    scores = []
    for q in data["questions"]:
        if q["question_type"] != "binary":
            continue
        pred = q["predictions"][model_id]
        prob = pred["probability"]
        gt = float(q["ground_truth"])
        brier = (prob - gt) ** 2
        scores.append(brier)
    return np.array(scores)


def bootstrap_ci(
    baseline: np.ndarray,
    conditional: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap 95% CI on the gap (conditional_mean - baseline_mean)."""
    gaps = np.empty(n_boot)
    n_base = len(baseline)
    n_cond = len(conditional)
    for i in range(n_boot):
        b_sample = baseline[rng.integers(0, n_base, size=n_base)]
        c_sample = conditional[rng.integers(0, n_cond, size=n_cond)]
        gaps[i] = c_sample.mean() - b_sample.mean()
    lo = np.percentile(gaps, CI_LOWER)
    hi = np.percentile(gaps, CI_UPPER)
    return lo, hi


def permutation_test(
    baseline: np.ndarray,
    conditional: np.ndarray,
    observed_gap: float,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """
    Two-sample permutation test.

    Null: baseline and conditional Brier scores come from the same
    distribution.  Test statistic: gap = mean(conditional) - mean(baseline).

    Returns the fraction of permutations where the gap >= observed gap
    (one-sided: conditional is worse).
    """
    combined = np.concatenate([baseline, conditional])
    n_base = len(baseline)
    n_total = len(combined)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n_total)
        perm_base = combined[perm[:n_base]]
        perm_cond = combined[perm[n_base:]]
        perm_gap = perm_cond.mean() - perm_base.mean()
        if perm_gap >= observed_gap:
            count += 1
    return count / n_perm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    for intervention_name, paths in INTERVENTIONS.items():
        # Discover models from baseline file
        with open(paths["baseline"]) as f:
            data = json.load(f)
        model_ids = list(data["model_results"].keys())

        results = []
        for model_id in model_ids:
            base_scores = load_brier_scores(paths["baseline"], model_id)
            cond_scores = load_brier_scores(paths["conditional"], model_id)

            base_mean = base_scores.mean()
            cond_mean = cond_scores.mean()
            gap = cond_mean - base_mean

            ci_lo, ci_hi = bootstrap_ci(base_scores, cond_scores, N_BOOTSTRAP, rng)
            p_value = permutation_test(base_scores, cond_scores, gap, N_PERMUTATIONS, rng)

            short = SHORT_NAMES.get(model_id, model_id)
            results.append(
                {
                    "model": short,
                    "baseline": base_mean,
                    "conditional": cond_mean,
                    "gap": gap,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "p_value": p_value,
                }
            )

        # Sort by gap descending (largest degradation first)
        results.sort(key=lambda r: r["gap"], reverse=True)

        # ----- Pretty table -----
        print("=" * 100)
        print(f"  {intervention_name} Intervention: Conditional Brier Gap Significance")
        print(
            f"  (n_baseline={len(base_scores)}, n_conditional={len(cond_scores)}, "
            f"bootstrap={N_BOOTSTRAP}, permutations={N_PERMUTATIONS})"
        )
        print("=" * 100)
        header = (
            f"{'Model':<22} {'Baseline':>10} {'Conditional':>12} "
            f"{'Gap':>8} {'95% CI':>20} {'p-value':>10}"
        )
        print(header)
        print("-" * 100)
        for r in results:
            ci_str = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
            p_str = f"{r['p_value']:.4f}" if r["p_value"] > 0 else "<0.0001"
            print(
                f"{r['model']:<22} {r['baseline']:>10.4f} {r['conditional']:>12.4f} "
                f"{r['gap']:>+8.4f} {ci_str:>20} {p_str:>10}"
            )
        print()

        # ----- Markdown table -----
        print(f"### {intervention_name} -- Markdown Table")
        print()
        print(
            "| Model | Baseline Brier | Conditional Brier | Gap | 95% CI | p-value |"
        )
        print(
            "|-------|---------------:|------------------:|----:|-------:|--------:|"
        )
        for r in results:
            ci_str = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
            p_str = f"{r['p_value']:.4f}" if r["p_value"] > 0 else "<0.0001"
            print(
                f"| {r['model']} | {r['baseline']:.4f} | {r['conditional']:.4f} "
                f"| {r['gap']:+.4f} | {ci_str} | {p_str} |"
            )
        print()
        print()


if __name__ == "__main__":
    main()
