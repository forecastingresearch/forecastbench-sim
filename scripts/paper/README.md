# Paper figures (ICML 2026 Forecasting Workshop)

Self-contained figure pipeline for the CivBench workshop submission.
Every script reads from `data/`, writes to
`data/evaluations/plots/paper/`, and shares styling via
`_style.py` and data loaders via `_data.py`.

## Output format

PNG at 200 DPI. The user inserts into Overleaf manually.

## Normalization (continuous CRPS)

Continuous templates are scored in their native units (cities, techs,
gold), but the units differ by orders of magnitude. To compare across
templates we divide CRPS and MAE by a fixed per-template value range:

| Template family | Value range |
|-----------------|-------------|
| `cities`        | 40          |
| `techs`         | 60          |
| `treasury`      | 2000        |

These ranges match the human-pilot bin edges in
`data/human_baseline/bin_schema.json`, so the same normalization
applies cleanly to both human and model scores.

A model's reported "normalized CRPS" is the *unweighted mean* of
per-question normalized CRPS across whichever slice is being shown
(e.g. all continuous predictions, or just H1 cities). Templates with
more questions therefore get more weight by question count, not by
template count. Raw CRPS is reported alongside in the appendix.

Brier (binary) is in [0, 1] and is not normalized.

## Curated model list (9)

The curated set is the 9 models that have full coverage across every
run (binary uncond, continuous uncond, H0 binary, H0 continuous, archived
republic + gold500 conditional). All 9 are highlighted in every figure;
non-curated models appear in light grey when they are part of the
backdrop set.

| Model id                                | Display name      |
|-----------------------------------------|-------------------|
| `anthropic/claude-sonnet-4-5-20250929`  | Claude Sonnet 4.5 |
| `openai/gpt-5.1-2025-11-13`             | GPT-5.1           |
| `openai/gpt-5-2025-08-07`               | GPT-5             |
| `openai/gpt-5-mini-2025-08-07`          | GPT-5 mini        |
| `openai/gpt-4.1-2025-04-14`             | GPT-4.1           |
| `openai/o3-2025-04-16`                  | o3                |
| `google/gemini-3-pro-preview`           | Gemini 3 Pro      |
| `google/gemini-2.5-pro`                 | Gemini 2.5 Pro    |
| `google/gemini-2.5-flash`               | Gemini 2.5 Flash  |

The figure-level "out of N" denominator differs by metric:

- **Binary** figures denominate by **30** (the main run model count).
- **Continuous** figures denominate by **31** = 30 main run +
  Claude Opus 4.5 spliced in from the archived `cont_uncond_all`. Opus
  is not curated (no binary / H0 forecasts), but it is shown in the
  continuous backdrop since the data exists and is on the same 2310
  question set as the main run.
- **Conditional** (Fig 4) is restricted to the curated 9 since that's
  where coverage is uniform.

## Figures and tables

Main paper:

| Script                          | Output                              |
|---------------------------------|-------------------------------------|
| `fig1_benchmark_schematic.py`   | `fig1_benchmark_schematic.png`      |
| `fig2_model_validation_compact.py` | `fig2_model_validation_compact.png` |
| `fig3_horizon_curve.py`         | `fig3_horizon_curve.png`            |
| `table1_benchmark_composition.py` | `table1_benchmark_composition.tex` |

Appendix:

| Script                          | Output                                  |
|---------------------------------|-----------------------------------------|
| `fig2_leaderboard.py`           | `fig2_leaderboard.png`                  |
| `fig4_intervention_gap.py`      | `fig4_intervention_gap.png`             |
| `figA4_null_conditional.py`     | `figA4_null_conditional.png`            |
| `figA1_human_vs_models.py`      | `figA1_human_vs_models.png`             |
| `figA2_template_heatmap.py`     | `figA2_template_heatmap.png`            |
| `figA3_reliability.py`          | `figA3_reliability.png`                 |
| external appendix asset         | `figA5_tail_risk.png`                   |
| `table1_benchmark_composition.py` | `table1_benchmark_composition.csv`    |
| `table2_headline_numbers.py`    | `table2_headline_numbers.csv`           |
| `tableA1_h0_comprehension.py`   | `tableA1_h0_comprehension.tex`          |

## Running

```bash
uv run python scripts/paper/fig1_benchmark_schematic.py
uv run python scripts/paper/fig2_model_validation_compact.py
uv run python scripts/paper/fig3_horizon_curve.py
# ...
```

Each script is self-contained (no required CLI flags) and writes to a
deterministic output path, so re-running overwrites prior PNGs.

## Decisions locked in

- Curated set = the 10 models in archived republic + gold500 conditional
  runs (Anthropic Opus / Sonnet, OpenAI 5.1 / 5 / 5-mini / 4.1 / o3,
  Google Gemini 3 Pro / 2.5 Pro / 2.5 Flash).
- Continuous CRPS is normalized by per-template value range (cities=40,
  techs=60, treasury=2000), matching the human-pilot bin widths. Raw
  numbers also reported in appendix.
- Brier (binary) is not normalized.
- Main-paper Fig 2 is compact and uses only curated models, with full-run
  rank annotations. The full 30/31-model leaderboard is appendix-only.
- Fig 3 main panel is H1–H7; H0 lives in appendix only.
- Fig 4 is appendix-only for the workshop paper. It uses metric (a):
  Brier-gap between conditional and baseline
  framings on matched questions, versus the observed ground-truth
  Brier-gap, plus a small accompanying scatter/bar.
- PNG output at 200 DPI; user inserts into Overleaf manually.
