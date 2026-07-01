# Session 2 Distribution Analysis

**Purpose**: explain how the base model's forecast distribution compares to the realized target distribution, how SFT/RL changed that distribution, and why the next recommended experiment is SFT-only before more RL.

## Data availability

The Session 2 eval JSONs include per-question `target`, `pred`, `meta`, and `raw_tail` fields. Session 1 SFT/RL eval rows are also available locally. The original Session 1 raw base-model rows were not pulled, so this analysis uses the Session 2 base evals for base-model distribution analysis.

The most comparable base condition is **base no-thinking + guided, t=0.6**, because it parsed all 401 ForecastBench rows and all 57 Freeciv-val rows.

## ForecastBench: actual distribution vs model predictions

ForecastBench outcomes are binary. In the full parse-fixed set, the realized YES rate is `0.192`.

The `freeze_market_value` column is highly predictive, but it is **not included in the model prompt**. The prompt includes freeze date, source, question, background, and resolution criteria, but not the market probability ([04_eval.py](04_eval.py#L41-L55)). The value is stored only in `meta` after prompt construction ([04_eval.py](04_eval.py#L67-L70)).

This matters because the market-value distribution is highly skewed:

| quantity | value |
|---|---:|
| outcome base rate | 0.192 |
| mean freeze market value | 0.219 |
| market-value Brier | 0.077 |
| constant-baseline Brier | 0.155 |
| examples with market value < 0.10 | 220/401 |
| actual YES rate when market < 0.10 | 0.009 |

The model is not given the strongest available base-rate signal. It therefore overpredicts low-probability ForecastBench events.

### ForecastBench prediction summary

| condition | n | target mean | pred mean | bias | pred std | median | p90 | >0.5 | corr | ECE10 | Brier | const |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 thinking t=0.6 | 248 | 0.157 | 0.276 | +0.119 | 0.188 | 0.250 | 0.480 | 10.1% | 0.297 | 0.119 | 0.141 | 0.133 |
| Base no-thinking + guided | 401 | 0.192 | 0.254 | +0.062 | 0.223 | 0.150 | 0.550 | 10.2% | 0.242 | 0.117 | 0.166 | 0.155 |
| E2 SFT best | 401 | 0.192 | 0.321 | +0.129 | 0.238 | 0.350 | 0.650 | 17.7% | 0.156 | 0.164 | 0.199 | 0.155 |
| E3 RL from SFT | 401 | 0.192 | 0.322 | +0.130 | 0.242 | 0.300 | 0.650 | 20.2% | 0.200 | 0.158 | 0.193 | 0.155 |
| E5 RL from base | 401 | 0.192 | 0.267 | +0.075 | 0.227 | 0.150 | 0.500 | 10.0% | 0.250 | 0.123 | 0.168 | 0.155 |
| S1 SFT ep4 | 322 | 0.208 | 0.320 | +0.112 | 0.260 | 0.250 | 0.700 | 22.4% | 0.165 | 0.183 | 0.210 | 0.165 |
| S1 post-RL | 393 | 0.196 | 0.284 | +0.088 | 0.233 | 0.250 | 0.650 | 17.6% | 0.237 | 0.126 | 0.176 | 0.158 |

The main distributional effect of SFT is an upward shift:

- Base no-thinking guided mean prediction: `0.254`.
- E2 SFT mean prediction: `0.321`.
- E3 RL from SFT mean prediction: `0.322`.
- E5 RL from base mean prediction: `0.267`.

E2 and E3 move the prediction distribution away from the realized ForecastBench base rate. E5 stays close to the base distribution.

### ForecastBench by market-value bin

This table groups examples by `freeze_market_value`, then reports actual outcome rate and mean predictions. The market value was not available to the model in the prompt, but it diagnoses where errors came from.

| market bin | n | actual YES | market mean | base pred | E2 pred | E3 pred | E5 pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0-0.1 | 220 | 0.009 | 0.024 | 0.211 | 0.293 | 0.283 | 0.224 |
| 0.1-0.2 | 50 | 0.140 | 0.138 | 0.190 | 0.262 | 0.258 | 0.205 |
| 0.2-0.3 | 22 | 0.136 | 0.249 | 0.308 | 0.370 | 0.373 | 0.296 |
| 0.3-0.4 | 18 | 0.278 | 0.336 | 0.303 | 0.383 | 0.394 | 0.352 |
| 0.4-0.5 | 18 | 0.389 | 0.435 | 0.317 | 0.431 | 0.481 | 0.342 |
| 0.5-0.6 | 12 | 0.417 | 0.548 | 0.288 | 0.229 | 0.350 | 0.375 |
| 0.6-0.7 | 12 | 0.667 | 0.645 | 0.363 | 0.450 | 0.417 | 0.396 |
| 0.7-0.8 | 16 | 0.625 | 0.751 | 0.384 | 0.413 | 0.381 | 0.316 |
| 0.8-0.9 | 12 | 0.750 | 0.842 | 0.233 | 0.293 | 0.331 | 0.277 |
| 0.9-1.0 | 21 | 1.000 | 0.945 | 0.526 | 0.486 | 0.519 | 0.526 |

Two errors dominate:

1. For very low-market questions, the base model predicts far too high: `0.211` against `0.009` actual YES rate. SFT worsens this to `0.293`.
2. For high-market questions, the models predict too low: in the `0.9-1.0` bin, actual YES rate is `1.000`, market mean is `0.945`, but model predictions are about `0.49-0.53`.

So the ForecastBench failure is not just "too high" globally. It is compression toward a broad LLM prior: rare events are overpredicted, high-confidence events are underpredicted, and SFT shifts the low-probability mass upward.

## Why SFT shifted ForecastBench upward

The Freeciv SFT target distribution is much higher than the ForecastBench outcome distribution, especially after weighting.

| dataset | n | mean target | weighted mean target | target std | target hist by decile |
|---|---:|---:|---:|---:|---|
| Freeciv train | 641 | 0.378 | 0.495 | 0.401 | `[274, 43, 25, 31, 23, 22, 23, 25, 36, 139]` |
| Freeciv-val | 57 | 0.561 | 0.546 | 0.388 | `[12, 2, 4, 3, 5, 0, 3, 2, 6, 20]` |
| E4 train | 506 | 0.419 | 0.507 | 0.405 | `[190, 33, 25, 24, 18, 18, 19, 20, 36, 123]` |
| E4 heldout | 192 | 0.323 | 0.480 | 0.389 | `[96, 12, 4, 10, 10, 4, 7, 7, 6, 36]` |

The SFT run uses example weighting, so the effective target mean is near `0.50`, not the unweighted train mean `0.378`. ForecastBench's realized base rate is `0.192`. A learned upward prior from Freeciv is therefore harmful on ForecastBench.

This is directly visible in the prediction deltas:

| comparison | suite | mean pred delta | median delta | Brier delta | >0.05 up | <-0.05 down | within +/-0.05 |
|---|---|---:|---:|---:|---:|---:|---:|
| E2 - base | ForecastBench | +0.068 | +0.030 | +0.033 | 38.7% | 12.0% | 49.4% |
| E3 - base | ForecastBench | +0.069 | +0.030 | +0.027 | 38.9% | 12.7% | 48.4% |
| E5 - base | ForecastBench | +0.013 | +0.000 | +0.002 | 21.4% | 17.2% | 61.3% |

E2 and E3 substantially move the ForecastBench distribution upward. E5 barely moves it.

## Freeciv-val: actual distribution vs model predictions

Freeciv-val has soft targets `p_mc`, not binary labels. Its target mean is `0.561`, with a large number of high-probability events.

| condition | n | target mean | pred mean | bias | pred std | median | p90 | >0.5 | corr | ECE10 | Brier | const |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base no-thinking + guided | 57 | 0.561 | 0.516 | -0.045 | 0.406 | 0.650 | 0.950 | 56.1% | 0.429 | 0.241 | 0.182 | 0.150 |
| E2 SFT best | 57 | 0.561 | 0.564 | +0.003 | 0.391 | 0.700 | 0.950 | 59.6% | 0.394 | 0.257 | 0.184 | 0.150 |
| E3 RL from SFT | 57 | 0.561 | 0.605 | +0.044 | 0.390 | 0.750 | 0.966 | 66.7% | 0.380 | 0.277 | 0.189 | 0.150 |
| E5 RL from base | 57 | 0.561 | 0.481 | -0.081 | 0.420 | 0.200 | 0.950 | 47.4% | 0.498 | 0.234 | 0.171 | 0.150 |
| S1 SFT ep4 | 56 | 0.558 | 0.531 | -0.027 | 0.352 | 0.600 | 0.950 | 58.9% | 0.471 | 0.169 | 0.147 | 0.152 |
| S1 post-RL | 56 | 0.554 | 0.462 | -0.091 | 0.423 | 0.500 | 0.990 | 46.4% | 0.372 | 0.310 | 0.215 | 0.149 |

For Session 2 Freeciv-val, the problem is not primarily mean bias. E2's mean prediction is almost exactly the target mean (`0.564` vs `0.561`), yet its Brier is worse than constant. The problem is low per-example discrimination/calibration: correlation and ECE do not improve.

E5 improves over base on Freeciv-val by making fewer changes and improving correlation (`0.498` vs `0.429`), but the final Brier is still worse than constant.

## Freeciv by-template behavior

Per-template results show that the aggregate Freeciv score hides very different regimes.

| template | n | target mean | base pred | base Brier | E2 pred | E2 Brier | E5 pred | E5 Brier | const |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| city_count_comparative | 4 | 0.700 | 0.675 | 0.004 | 0.675 | 0.004 | 0.675 | 0.004 | 0.166 |
| government_at | 10 | 0.305 | 0.375 | 0.325 | 0.505 | 0.355 | 0.375 | 0.247 | 0.128 |
| population_comparative | 4 | 0.713 | 0.662 | 0.011 | 0.512 | 0.124 | 0.725 | 0.002 | 0.170 |
| score_comparative | 4 | 0.812 | 0.800 | 0.118 | 0.825 | 0.083 | 0.688 | 0.236 | 0.047 |
| score_rank_1 | 5 | 0.200 | 0.232 | 0.022 | 0.330 | 0.031 | 0.240 | 0.005 | 0.124 |
| tech_comparative | 4 | 0.738 | 0.637 | 0.274 | 0.738 | 0.096 | 0.537 | 0.161 | 0.139 |
| tech_discovered | 15 | 0.560 | 0.439 | 0.193 | 0.603 | 0.171 | 0.349 | 0.209 | 0.106 |
| territory_comparative | 4 | 0.550 | 0.717 | 0.068 | 0.512 | 0.161 | 0.725 | 0.069 | 0.134 |
| treasury_comparative | 4 | 0.750 | 0.950 | 0.074 | 0.812 | 0.079 | 0.925 | 0.067 | 0.034 |
| wonder_completed | 3 | 0.833 | 0.050 | 0.642 | 0.033 | 0.672 | 0.040 | 0.654 | 0.029 |

The sharpest standard-val failure is `wonder_completed`: all Session 2 models predict near zero, but the three standard-val examples average `0.833`. This is not mysterious after checking the data distribution:

- Freeciv train `wonder_completed`: `n=101`, mean target `0.127`.
- Freeciv-val `wonder_completed`: `n=3`, mean target `0.833`.

So the standard validation split contains a small but severe seed-level distribution shift for `wonder_completed`. This explains why all models are badly wrong there. It also warns against overinterpreting the 57-example Freeciv-val aggregate.

## E4 template-holdout root cause

E4 is more informative than the 57-example Freeciv-val split because the held-out set has 192 examples.

E4 did not fail only because of mean bias. For the held-out templates, the learned model often had roughly plausible mean predictions but poor ordering/correlation.

At epoch 1:

| template | n | target mean | pred mean | bias | corr |
|---|---:|---:|---:|---:|---:|
| tech_comparative | 44 | 0.489 | 0.648 | +0.159 | 0.274 |
| territory_comparative | 44 | 0.574 | 0.754 | +0.180 | 0.335 |
| wonder_completed | 104 | 0.147 | 0.110 | -0.038 | -0.033 |

At epoch 4:

| template | n | target mean | pred mean | bias | corr |
|---|---:|---:|---:|---:|---:|
| tech_comparative | 44 | 0.489 | 0.670 | +0.182 | -0.035 |
| territory_comparative | 44 | 0.574 | 0.699 | +0.125 | 0.052 |
| wonder_completed | 104 | 0.147 | 0.154 | +0.007 | 0.023 |

For `wonder_completed`, epoch 4 gets the mean almost exactly right but still has worse Brier than constant because it cannot rank examples within the template. For `tech_comparative` and `territory_comparative`, it both overpredicts and loses correlation by epoch 4.

This supports the template-memorization interpretation more strongly than the standard Freeciv-val split: absent the template during training, the model does not learn a useful per-example mapping for that template.

## Why I recommended SFT-only before more RL

This is not because SFT is currently working. It is because SFT is the cheaper and cleaner diagnostic for the remaining confound.

The distribution data shows:

1. **SFT produces a large, measurable distribution shift.** On ForecastBench, E2 raises mean prediction by `+0.068` relative to base and worsens Brier by `+0.033`.
2. **RL from SFT mostly inherits the SFT shift.** E3 has almost the same ForecastBench mean as E2 (`0.322` vs `0.321`) and remains badly overpredicted.
3. **RL from base barely moves the distribution.** E5 changes ForecastBench mean by only `+0.013`; 61.3% of predictions stay within `+/-0.05` of base. It improves Freeciv-val Brier by only `0.011` and still loses to constant.
4. **E3 reward was flat.** First four logged rewards averaged `-0.1496`; last four averaged `-0.1520`.
5. **E5 reward moved mildly, but not enough to matter.** First four logged rewards averaged `-0.1300`; last four averaged `-0.1187`, with final Brier still worse than constant.

Therefore the immediate question is not "which RL hyperparameter should we try?" The immediate question is whether the supervised target can be learned and evaluated under a matched format at all.

A matched no-thinking SFT run answers that question cheaply:

- If aligned SFT cannot beat constant on standard Freeciv-val and E4 held-out templates, then RL on the same data is unlikely to be a productive next step.
- If aligned SFT does beat those gates, then RL becomes meaningful again: it can be tested as a refinement step starting from a demonstrably useful supervised policy.

So the recommendation is **SFT-only as a diagnostic gate**, not SFT as the final method.

## Concrete next analysis/experiment

1. Run base no-thinking + guided on `data/training_E4/val.jsonl` to compare E4 SFT against base, not only constant.
2. Run an aligned no-thinking SFT where `enable_thinking=False` is used consistently in training prompt construction and eval.
3. Evaluate that SFT on standard Freeciv-val, E4 heldout, and ForecastBench with both guided decoding and unguided/two-pass parsing.
4. Only run RL if SFT beats constant on Freeciv-val and at least one template-holdout split.
