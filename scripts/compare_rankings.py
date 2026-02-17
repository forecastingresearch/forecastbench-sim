#!/usr/bin/env python3
"""
Compare model rankings between ForecastBench and CivBench.

Uses CI-aware analysis: rankings are only considered definitive when CIs don't overlap.
Bootstraps from CI distributions to propagate uncertainty into correlation estimates.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


# Model name mapping: CivBench ID -> ForecastBench name
CIVBENCH_TO_FORECASTBENCH = {
    "openai/gpt-3.5-turbo-0125": "GPT-3.5-Turbo-0125",
    "openai/gpt-4o": "GPT-4o-2024-05-13",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1-2025-04-14",
    "openai/gpt-5-2025-08-07": "GPT-5-2025-08-07",
    "openai/gpt-5-mini-2025-08-07": "GPT-5-Mini-2025-08-07",
    "openai/gpt-5-nano-2025-08-07": "GPT-5-Nano-2025-08-07",
    "openai/gpt-5.1-2025-11-13": "GPT-5.1-2025-11-13",
    "openai/o3-2025-04-16": "O3-2025-04-16",
    "openai/o3-mini-2025-01-31": "O3-Mini-2025-01-31",
    "openai/o4-mini-2025-04-16": "O4-Mini-2025-04-16",
    "anthropic/claude-3-haiku-20240307": "Claude-3-Haiku-20240307",
    "anthropic/claude-haiku-4-5-20251001": "Claude-Haiku-4-5-20251001",
    "google/gemini-2.0-flash-lite-001": "Gemini-2.0-Flash-Lite-001",
    "google/gemini-2.5-flash": "Gemini-2.5-Flash",
    "xai/grok-4-fast-reasoning": "Grok-4-Fast-Reasoning",
    "xai/grok-4-1-fast-reasoning": "Grok-4-1-Fast-Reasoning",
    "mistral/mistral-large-2411": "Mistral-Large-2411",
    "mistral/mistral-large-latest": "Mistral-Large-Latest",
}


def load_forecastbench(filepath: str) -> pd.DataFrame:
    """Load ForecastBench CSV and extract zero-shot model scores with CIs."""
    df = pd.read_csv(filepath)
    model_col = 'Model'

    df = df[df[model_col].astype(str).str.contains(r'\(zero shot\)', case=False, regex=True)]

    results = []
    for _, row in df.iterrows():
        model_name = str(row[model_col]).strip()

        # Parse Overall (N) column - format: "0.150 (562)" or "0.150 (1,270)"
        overall_str = str(row['Overall (N)'])
        match = re.match(r'([\d.]+)\s*\(([\d,]+)\)', overall_str)
        if not match:
            continue
        overall_score = float(match.group(1))

        # Parse CI
        ci_str = str(row['Overall 95% CI'])
        ci_match = re.match(r'\[([\d.]+),\s*([\d.]+)\]', ci_str)
        if ci_match:
            ci_lower = float(ci_match.group(1))
            ci_upper = float(ci_match.group(2))
        else:
            ci_lower = ci_upper = overall_score

        base_name = re.sub(r'\s*\(zero shot\)\s*$', '', model_name, flags=re.IGNORECASE).strip()

        results.append({
            'model_name': base_name,
            'score': overall_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        })

    return pd.DataFrame(results)


def load_civbench(filepath: str) -> pd.DataFrame:
    """Load CivBench results CSV."""
    df = pd.read_csv(filepath)
    return df.rename(columns={'model': 'model_name', 'brier_score': 'score'})


def match_models(civbench_df: pd.DataFrame, forecastbench_df: pd.DataFrame) -> pd.DataFrame:
    """Match models between CivBench and ForecastBench using the mapping."""
    fb_lookup = {row['model_name'].lower(): row.to_dict()
                 for _, row in forecastbench_df.iterrows()}

    matched = []
    unmatched = []

    for _, cb_row in civbench_df.iterrows():
        civbench_id = cb_row['model_name']
        fb_name = CIVBENCH_TO_FORECASTBENCH.get(civbench_id)

        if fb_name and fb_name.lower() in fb_lookup:
            fb_data = fb_lookup[fb_name.lower()]
            matched.append({
                'model': civbench_id,
                'civbench_score': cb_row['score'],
                'civbench_ci_lower': cb_row['ci_lower'],
                'civbench_ci_upper': cb_row['ci_upper'],
                'forecastbench_score': fb_data['score'],
                'forecastbench_ci_lower': fb_data['ci_lower'],
                'forecastbench_ci_upper': fb_data['ci_upper'],
            })
        else:
            unmatched.append(civbench_id)

    if unmatched:
        print(f"\nUnmatched CivBench models ({len(unmatched)}):")
        for m in unmatched:
            print(f"  {m}")

    return pd.DataFrame(matched)


def ci_to_std(ci_lower: float, ci_upper: float) -> float:
    """Convert 95% CI to standard deviation (assuming normal distribution)."""
    return (ci_upper - ci_lower) / (2 * 1.96)


def sample_from_ci(score: float, ci_lower: float, ci_upper: float,
                   rng: np.random.Generator) -> float:
    """Sample a value from the CI distribution (assuming normal)."""
    std = ci_to_std(ci_lower, ci_upper)
    if std <= 0:
        return score
    return rng.normal(score, std)


def cis_overlap(ci1_lower: float, ci1_upper: float,
                ci2_lower: float, ci2_upper: float) -> bool:
    """Check if two confidence intervals overlap."""
    return not (ci1_upper < ci2_lower or ci2_upper < ci1_lower)


def analyze_pairwise_with_ci(matched_df: pd.DataFrame) -> dict:
    """
    Analyze pairwise ranking consistency accounting for CI overlap.

    For each pair:
    - "definite_consistent": Both benchmarks agree on direction AND neither CI overlaps
    - "definite_inconsistent": Both benchmarks disagree AND neither CI overlaps
    - "uncertain": At least one benchmark has overlapping CIs for this pair
    """
    n = len(matched_df)
    definite_consistent = 0
    definite_inconsistent = 0
    uncertain = 0
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            row_i = matched_df.iloc[i]
            row_j = matched_df.iloc[j]

            # CivBench comparison
            cb_overlap = cis_overlap(
                row_i['civbench_ci_lower'], row_i['civbench_ci_upper'],
                row_j['civbench_ci_lower'], row_j['civbench_ci_upper']
            )
            cb_direction = np.sign(row_i['civbench_score'] - row_j['civbench_score'])

            # ForecastBench comparison
            fb_overlap = cis_overlap(
                row_i['forecastbench_ci_lower'], row_i['forecastbench_ci_upper'],
                row_j['forecastbench_ci_lower'], row_j['forecastbench_ci_upper']
            )
            fb_direction = np.sign(row_i['forecastbench_score'] - row_j['forecastbench_score'])

            if cb_overlap or fb_overlap:
                status = 'uncertain'
                uncertain += 1
            elif cb_direction == fb_direction:
                status = 'definite_consistent'
                definite_consistent += 1
            else:
                status = 'definite_inconsistent'
                definite_inconsistent += 1

            pairs.append({
                'model_i': row_i['model'],
                'model_j': row_j['model'],
                'civbench_diff': row_i['civbench_score'] - row_j['civbench_score'],
                'forecastbench_diff': row_i['forecastbench_score'] - row_j['forecastbench_score'],
                'civbench_overlap': cb_overlap,
                'forecastbench_overlap': fb_overlap,
                'status': status,
            })

    total = definite_consistent + definite_inconsistent + uncertain
    definite_total = definite_consistent + definite_inconsistent

    return {
        'definite_consistent': definite_consistent,
        'definite_inconsistent': definite_inconsistent,
        'uncertain': uncertain,
        'total_pairs': total,
        'definite_consistency_rate': definite_consistent / definite_total if definite_total > 0 else None,
        'pairs': pairs
    }


def bootstrap_rank_correlation(matched_df: pd.DataFrame,
                               n_bootstrap: int = 10000) -> dict:
    """
    Bootstrap rank correlation by sampling from CI distributions.

    For each bootstrap iteration:
    1. Sample each model's score from its CI distribution (both benchmarks)
    2. Compute ranks
    3. Compute Spearman correlation between ranks

    Returns distribution of correlations.
    """
    n = len(matched_df)
    rng = np.random.default_rng(42)

    bootstrap_rhos = []

    for _ in range(n_bootstrap):
        # Sample scores from CIs
        cb_samples = np.array([
            sample_from_ci(row['civbench_score'], row['civbench_ci_lower'], row['civbench_ci_upper'], rng)
            for _, row in matched_df.iterrows()
        ])
        fb_samples = np.array([
            sample_from_ci(row['forecastbench_score'], row['forecastbench_ci_lower'], row['forecastbench_ci_upper'], rng)
            for _, row in matched_df.iterrows()
        ])

        # Compute ranks (lower Brier = better = rank 1)
        cb_ranks = stats.rankdata(cb_samples)
        fb_ranks = stats.rankdata(fb_samples)

        # Compute correlation
        rho, _ = stats.spearmanr(cb_ranks, fb_ranks)

        bootstrap_rhos.append(rho)

    bootstrap_rhos = np.array(bootstrap_rhos)

    # Point estimate using original scores
    cb_ranks_orig = stats.rankdata(matched_df['civbench_score'].values)
    fb_ranks_orig = stats.rankdata(matched_df['forecastbench_score'].values)
    rho_point, rho_pval = stats.spearmanr(cb_ranks_orig, fb_ranks_orig)

    return {
        'spearman': {
            'point_estimate': rho_point,
            'p_value': rho_pval,
            'bootstrap_mean': np.mean(bootstrap_rhos),
            'bootstrap_median': np.median(bootstrap_rhos),
            'ci_lower': np.percentile(bootstrap_rhos, 2.5),
            'ci_upper': np.percentile(bootstrap_rhos, 97.5),
            'prob_positive': np.mean(bootstrap_rhos > 0),
        },
        'n_models': n,
    }


def permutation_test_spearman_rho(matched_df: pd.DataFrame,
                                  n_permutations: int = 10000,
                                  seed: int = 42) -> dict:
    """
    Permutation test for Spearman's rho by permuting labels in one benchmark.

    Uses point estimates (not CI sampling) to build the null distribution.
    """
    rng = np.random.default_rng(seed)

    cb_scores = matched_df['civbench_score'].values
    fb_scores = matched_df['forecastbench_score'].values

    cb_ranks = stats.rankdata(cb_scores)
    fb_ranks = stats.rankdata(fb_scores)

    rho_obs, _ = stats.spearmanr(cb_ranks, fb_ranks)

    perm_rhos = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        permuted = rng.permutation(fb_scores)
        perm_rho, _ = stats.spearmanr(cb_ranks, stats.rankdata(permuted))
        perm_rhos[i] = perm_rho

    # One-sided p-value for positive association (rho >= observed).
    p_value_one_sided = (np.sum(perm_rhos >= rho_obs) + 1) / (n_permutations + 1)

    return {
        'rho_observed': rho_obs,
        'p_value_one_sided': p_value_one_sided,
        'null_mean': float(np.mean(perm_rhos)),
        'null_median': float(np.median(perm_rhos)),
        'null_ci_lower': float(np.percentile(perm_rhos, 2.5)),
        'null_ci_upper': float(np.percentile(perm_rhos, 97.5)),
        'perm_rhos': perm_rhos,
        'n_permutations': n_permutations,
    }


def compare_rankings(forecastbench_path: str, civbench_path: str, n_bootstrap: int = 10000,
                     n_permutations: int = 50000):
    """Main function to compare rankings between ForecastBench and CivBench."""

    print("=" * 70)
    print("COMPARING MODEL RANKINGS: ForecastBench vs CivBench")
    print("(CI-Aware Analysis)")
    print("=" * 70)

    # Load data
    print("\nLoading ForecastBench data...")
    fb_df = load_forecastbench(forecastbench_path)
    print(f"  Found {len(fb_df)} zero-shot models")

    print("\nLoading CivBench data...")
    cb_df = load_civbench(civbench_path)
    print(f"  Found {len(cb_df)} models")

    # Match models
    print("\nMatching models...")
    matched_df = match_models(cb_df, fb_df)
    print(f"  Matched {len(matched_df)} models")

    if len(matched_df) < 3:
        print("\nERROR: Too few matched models for meaningful analysis")
        return

    # Print matched models with rankings
    print("\n" + "-" * 70)
    print("MATCHED MODELS (sorted by CivBench rank)")
    print("-" * 70)
    matched_sorted = matched_df.sort_values('civbench_score')
    print(f"{'Model':<40} {'CB Rank':<8} {'FB Rank':<8}")
    print("-" * 70)

    cb_ranks = stats.rankdata(matched_df['civbench_score'].values)
    fb_ranks = stats.rankdata(matched_df['forecastbench_score'].values)
    matched_df['cb_rank'] = cb_ranks
    matched_df['fb_rank'] = fb_ranks

    for _, row in matched_df.sort_values('cb_rank').iterrows():
        model_short = row['model'].split('/')[-1][:35]
        cb_r = int(row['cb_rank'])
        fb_r = int(row['fb_rank'])
        cb_score = f"{row['civbench_score']:.3f}"
        fb_score = f"{row['forecastbench_score']:.3f}"
        print(f"{model_short:<35} #{cb_r:<3} ({cb_score})  #{fb_r:<3} ({fb_score})")

    # CI-aware pairwise analysis
    print("\n" + "-" * 70)
    print("CI-AWARE PAIRWISE RANKING ANALYSIS")
    print("-" * 70)

    pairwise = analyze_pairwise_with_ci(matched_df)
    print(f"\nTotal pairs: {pairwise['total_pairs']}")
    print(f"  Definite consistent:   {pairwise['definite_consistent']:3d} "
          f"(CIs non-overlapping, same ranking direction)")
    print(f"  Definite inconsistent: {pairwise['definite_inconsistent']:3d} "
          f"(CIs non-overlapping, opposite ranking direction)")
    print(f"  Uncertain:             {pairwise['uncertain']:3d} "
          f"(at least one benchmark has overlapping CIs)")

    if pairwise['definite_consistency_rate'] is not None:
        print(f"\nAmong definite rankings: {pairwise['definite_consistency_rate']:.1%} consistent")

    # Show definite inconsistencies
    inconsistent = [p for p in pairwise['pairs'] if p['status'] == 'definite_inconsistent']
    if inconsistent:
        print(f"\nDefinite ranking disagreements ({len(inconsistent)}):")
        for p in inconsistent[:10]:  # Show first 10
            mi = p['model_i'].split('/')[-1]
            mj = p['model_j'].split('/')[-1]
            cb_better = mi if p['civbench_diff'] < 0 else mj
            fb_better = mi if p['forecastbench_diff'] < 0 else mj
            print(f"  CivBench: {cb_better} better | ForecastBench: {fb_better} better")

    # Bootstrap correlation analysis
    print("\n" + "-" * 70)
    print("RANK CORRELATION (Bootstrapped from CI distributions)")
    print("-" * 70)

    corr_results = bootstrap_rank_correlation(matched_df, n_bootstrap)

    print(f"\nSpearman's rho (rank correlation):")
    spearman = corr_results['spearman']
    print(f"  Point estimate: {spearman['point_estimate']:.4f}")
    print(f"  Bootstrap mean: {spearman['bootstrap_mean']:.4f}")
    print(f"  95% CI (from CI sampling): [{spearman['ci_lower']:.4f}, {spearman['ci_upper']:.4f}]")
    print(f"  P(rho > 0): {spearman['prob_positive']:.1%}")

    # Permutation test (null distribution by permuting ForecastBench labels)
    print("\n" + "-" * 70)
    print("PERMUTATION TEST (Null via label permutation)")
    print("-" * 70)

    perm_results = permutation_test_spearman_rho(matched_df, n_permutations)
    print(f"\nObserved Spearman's rho (point estimates): {perm_results['rho_observed']:.4f}")
    print(f"Null mean rho: {perm_results['null_mean']:.4f}")
    print(f"Null 95% CI: [{perm_results['null_ci_lower']:.4f}, {perm_results['null_ci_upper']:.4f}]")
    print(f"One-sided permutation p-value (rho >= observed): {perm_results['p_value_one_sided']:.6f}")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if spearman['ci_lower'] > 0:
        print("\nThe rank correlation is SIGNIFICANTLY POSITIVE.")
        print("Models that rank better on ForecastBench tend to rank better on CivBench.")
    elif spearman['ci_upper'] < 0:
        print("\nThe rank correlation is SIGNIFICANTLY NEGATIVE.")
        print("Models that rank better on ForecastBench tend to rank worse on CivBench.")
    else:
        print("\nThe rank correlation CI INCLUDES ZERO.")
        print("We cannot conclude whether the benchmarks rank models similarly or differently.")

    print(f"\nNote: The bootstrap CI accounts for uncertainty in both benchmarks' scores.")
    print(f"      {pairwise['uncertain']}/{pairwise['total_pairs']} pairs "
          f"({100*pairwise['uncertain']/pairwise['total_pairs']:.0f}%) have uncertain rankings due to CI overlap.")
    print("      The permutation test compares the observed rho to a null distribution")
    print("      from permuting ForecastBench labels (point estimates).")

    return {
        'matched_df': matched_df,
        'pairwise': pairwise,
        'correlations': corr_results,
        'permutation': perm_results,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare ForecastBench vs CivBench rankings (CI-aware)')
    parser.add_argument('--forecastbench', default='data/ForecastBench.csv')
    parser.add_argument('--civbench', default='data/civbench_results.csv')
    parser.add_argument('--n-bootstrap', type=int, default=10000)
    parser.add_argument('--n-permutations', type=int, default=50000)

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    def resolve(p):
        return project_root / p if not Path(p).is_absolute() else Path(p)

    compare_rankings(str(resolve(args.forecastbench)), str(resolve(args.civbench)),
                     args.n_bootstrap, args.n_permutations)
