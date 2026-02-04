#!/usr/bin/env python3
"""
Analyze and compare Opus 4.5 Brier scores across different conditions:
- Unconditional (null conditional)
- Republic conditional
- Gold +5000 conditional (as available proxy for +500)

Produces:
- Bar chart of Brier scores
- Statistical significance tests between pairs
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

# Paths to evaluation files
DATA_DIR = Path(__file__).parent.parent / "data" / "evaluations"

EVAL_FILES = {
    "Unconditional": DATA_DIR / "baseline_opus45_republic_eval.json",
    "Republic": DATA_DIR / "conditional_opus45_republic_eval.json",
    "Gold +500": DATA_DIR / "gold500_opus45_eval.json",
}


def load_evaluation(filepath: Path) -> dict:
    """Load evaluation results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def extract_question_briers(eval_data: dict, model_id: str = "anthropic/claude-opus-4-5-20251101") -> list[float]:
    """Extract per-question squared errors (Brier components) for a model."""
    briers = []
    for q in eval_data["questions"]:
        pred = q["predictions"].get(model_id)
        if pred and pred.get("probability") is not None and pred.get("error") is None:
            prob = pred["probability"]
            truth = 1.0 if q["ground_truth"] else 0.0
            squared_error = (prob - truth) ** 2
            briers.append(squared_error)
    return briers


def bootstrap_ci(data: list[float], n_bootstrap: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    data = np.array(data)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    return lower, upper


def permutation_test(data1: list[float], data2: list[float], n_permutations: int = 10000) -> float:
    """
    Permutation test for difference in means between two groups.
    Returns p-value (two-tailed).
    """
    data1 = np.array(data1)
    data2 = np.array(data2)

    observed_diff = np.mean(data1) - np.mean(data2)
    combined = np.concatenate([data1, data2])
    n1 = len(data1)

    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    return count / n_permutations


def main():
    np.random.seed(42)  # For reproducibility

    # Load all evaluations
    print("=" * 70)
    print("Loading evaluation data...")
    print("=" * 70)

    results = {}
    for condition, filepath in EVAL_FILES.items():
        if filepath.exists():
            eval_data = load_evaluation(filepath)
            briers = extract_question_briers(eval_data)

            results[condition] = {
                "briers": briers,
                "mean_brier": np.mean(briers),
                "std_brier": np.std(briers, ddof=1),
                "n_questions": len(briers),
                "metadata": eval_data.get("metadata", {}),
            }

            print(f"\n{condition}:")
            print(f"  File: {filepath.name}")
            print(f"  Questions: {len(briers)}")
            print(f"  Brier Score: {np.mean(briers):.4f} ± {np.std(briers, ddof=1):.4f}")
        else:
            print(f"\n{condition}: FILE NOT FOUND ({filepath})")

    # Compute confidence intervals
    print("\n" + "=" * 70)
    print("Bootstrap 95% Confidence Intervals")
    print("=" * 70)

    for condition, data in results.items():
        ci_lower, ci_upper = bootstrap_ci(data["briers"])
        results[condition]["ci_lower"] = ci_lower
        results[condition]["ci_upper"] = ci_upper
        print(f"\n{condition}:")
        print(f"  Brier: {data['mean_brier']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Statistical significance tests between all pairs
    print("\n" + "=" * 70)
    print("Statistical Significance Tests (Permutation Test)")
    print("=" * 70)

    conditions = list(results.keys())
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            cond1, cond2 = conditions[i], conditions[j]

            briers1 = results[cond1]["briers"]
            briers2 = results[cond2]["briers"]

            # Permutation test
            p_value = permutation_test(briers1, briers2)

            # Also compute t-test for reference
            t_stat, t_pvalue = stats.ttest_ind(briers1, briers2, equal_var=False)

            diff = results[cond1]["mean_brier"] - results[cond2]["mean_brier"]

            print(f"\n{cond1} vs {cond2}:")
            print(f"  Difference: {diff:+.4f} (Brier)")
            print(f"  Permutation test p-value: {p_value:.4f}")
            print(f"  Welch's t-test p-value: {t_pvalue:.4f}")
            print(f"  Significant at α=0.05: {'YES' if p_value < 0.05 else 'NO'}")

    # Create bar chart
    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))

    conditions_sorted = sorted(results.keys(), key=lambda x: results[x]["mean_brier"])
    x_pos = np.arange(len(conditions_sorted))

    means = [results[c]["mean_brier"] for c in conditions_sorted]
    errors_lower = [results[c]["mean_brier"] - results[c]["ci_lower"] for c in conditions_sorted]
    errors_upper = [results[c]["ci_upper"] - results[c]["mean_brier"] for c in conditions_sorted]

    # Use different colors for each condition
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # green, blue, red

    bars = ax.bar(
        x_pos, means,
        yerr=[errors_lower, errors_upper],
        capsize=5,
        color=colors[:len(conditions_sorted)],
        edgecolor='black',
        linewidth=1.2
    )

    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Brier Score (lower is better)', fontsize=12)
    ax.set_title('Opus 4.5 Brier Scores Across Conditions\n(with 95% Bootstrap CI)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions_sorted, fontsize=11)

    # Add value labels on bars
    for bar, mean, n in zip(bars, means, [results[c]["n_questions"] for c in conditions_sorted]):
        ax.annotate(
            f'{mean:.3f}\n(n={n})',
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha='center', va='bottom',
            fontsize=10
        )

    # Add horizontal line for uninformed baseline (0.25)
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.7, label='Uninformed baseline (0.25)')
    ax.legend(loc='upper right')

    ax.set_ylim(0, max(means) * 1.3)

    plt.tight_layout()

    output_path = DATA_DIR / "opus45_condition_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")

    # Also save as PDF
    output_pdf = DATA_DIR / "opus45_condition_comparison.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Saved plot to: {output_pdf}")

    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nBrier Scores (lower is better):")
    for c in conditions_sorted:
        print(f"  {c}: {results[c]['mean_brier']:.4f} (n={results[c]['n_questions']})")

    print("\nKey findings:")
    best = conditions_sorted[0]
    worst = conditions_sorted[-1]
    print(f"  - Best performing: {best} ({results[best]['mean_brier']:.4f})")
    print(f"  - Worst performing: {worst} ({results[worst]['mean_brier']:.4f})")


if __name__ == "__main__":
    main()
