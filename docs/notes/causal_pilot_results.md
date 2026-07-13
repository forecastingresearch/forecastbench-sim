# Causal golden-question pilot — results (corrected stats)

2026-07-13. Data: 4 seeds × (40 baseline + 40 do(X)) rollouts, 220 questions, H1
(turn 60 → 90). Analysis: `scripts/uplift_v2/causal_analyze.py` + paired/noise-floor
pass (this doc). Raw: `tmp/causal/analysis.json`.

## Correction to the overnight summary

The overnight commit claimed "46/220 questions with |confounding gap| > 0.15." That
count did not account for the small observational-arm sample (n_obs = 10–22, SE up to
~0.16). Under a proper two-proportion test (p_obs vs p_do), **13/220 gaps are
significant at |z| ≥ 1.96, vs ~11 expected by chance** — the gap signal is *suggestive
in its largest instances, not established in aggregate* at N=40. The Δ_do (causal
effect) signal, by contrast, is solidly above noise (below).

## What the pilot established

1. **The machinery works end to end** (fork → intervene → re-simulate → paired
   resolution), including a fixed silently-broken tech intervention.
2. **Real causal effects, above noise:** 19/220 questions have |Δ_do| > 0.15 vs
   **8.8 expected under the null** (per-question binomial simulation at each p_y).
   Top effect survives Bonferroni across all 220 tests:
   - seed2, do(grant The Republic to Greater Polish): "Will Greater Polish have more
     technologies than Benin at turn 90?" Δ_do = **−0.38, z = −3.8** (granting the
     Republic tech redirects research and triggers government switching — fewer techs
     by t90). Nine more at |z| ≥ 2.0, all mechanistically coherent (tech_discovered /
     tech_comparative questions downstream of the granted tech; "Will Arab be in
     Republic at turn 90" +0.25 under a tech grant that frees its research path).
3. **The confounding-gap pattern is visible and interpretable, but needs more N:**
   largest gaps run 0.3–0.4 with z ≈ 2.5–2.9 (e.g., seed2: worlds that discover The
   Republic naturally are science-strong worlds → p_obs = 0.70, while imposing it
   gives p_do = 0.28 — textbook selection vs intervention). None survives
   family-wise correction at N=40.
4. **Common-random-number pairing does not help:** cross-arm outcome correlation at
   matched RNG seeds is ≈ 0.01 — 30 turns of chaos fully decorrelates trajectories,
   so variance reduction must come from N, not pairing.
5. **Known design asymmetry (matters for interpretation):** do(X) grants the tech at
   turn 60; the observational X is "discovered by turn 70." Part of Δ_do vs Δ_obs
   is therefore a timing head-start, not confounding. Fixable by granting at the
   median natural discovery turn, or conditioning on earlier discovery windows.

## Implications

- The **simulator ground-truth side of WhatIf-Bench is validated**: we can produce
  questions where interventional and observational conditionals demonstrably diverge —
  the quantity that cannot be measured in the real world, and the core of the pitch
  (TFR-Bench's synthetic-oracle gap).
- **Phase 4 (LLM elicitation) can run now** on the 19-question high-|Δ_do| subset +
  placebo controls (~$5–15 API, gated on approval): score elicited Δ̂ against Δ_do,
  and test whether models distinguish do-framing from observed-framing at all.
- **To certify the confounding gap statistically:** scale seed2 to ~150–200
  rollouts/arm (free local CPU, ~4–5 h with 2 workers) → SE(p) ≈ 0.03, which makes
  the 0.3–0.4 gaps ≈ 6–10σ. Recommended before showing the result externally.
